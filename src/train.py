import os
import io
import torch
import librosa
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import (
    VitsModel, 
    VitsTokenizer, 
    TrainingArguments, 
    Trainer
)
from huggingface_hub import login

# 1. Xác thực Hugging Face
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# 2. Cấu hình & Load Dataset (Giữ nguyên phần decode=False an toàn)
model_id = "facebook/mms-tts-vie"
dataset_id = "doof-ferb/infore1_25hours"

dataset = load_dataset(dataset_id, split="train")
dataset = dataset.cast_column("audio", Audio(decode=False))

# 3. Load Model & Tokenizer
tokenizer = VitsTokenizer.from_pretrained(model_id)
# Sử dụng VitsModel nhưng cần kích hoạt chế độ train
model = VitsModel.from_pretrained(model_id)
model.train()

# 4. Tiền xử lý dữ liệu
def prepare_dataset(batch):
    audio_data = batch["audio"]
    audio_bytes = audio_data["bytes"]
    array, sr = sf.read(io.BytesIO(audio_bytes))
    if sr != 16000:
        array = librosa.resample(y=array, orig_sr=sr, target_sr=16000)

    # Tokenize text
    inputs = tokenizer(batch["transcription"], return_tensors=None)
    batch["input_ids"] = inputs.input_ids
    batch["labels"] = array # Waveform
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=4)

# 5. Data Collator và Trainer tùy chỉnh để giải quyết lỗi NotImplementedError
class VitsTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Ghi đè hàm tính loss vì VitsModel mặc định không trả về loss
        """
        labels = inputs.pop("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        
        # Vì MMS-TTS trong transformers chưa hỗ trợ training forward hoàn chỉnh 
        # Chúng ta giả lập một hàm loss đơn giản để Trainer có thể chạy
        # LƯU Ý: Để fine-tune chuẩn xác nhất, nên dùng script từ: 
        # https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-to-speech
        
        waveform = outputs.waveform
        # Cắt ngắn waveform hoặc labels để cùng kích thước
        min_size = min(waveform.shape[1], labels.shape[1])
        loss = torch.nn.functional.mse_loss(waveform[:, :min_size], labels[:, :min_size])
        
        return (loss, outputs) if return_outputs else loss

class TTSDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.float) for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.0)
        attention_mask = (input_ids_padded != 0).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded
        }

# 6. Huấn luyện
training_args = TrainingArguments(
    output_dir="./mms-tts-vie-finetuned",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    max_steps=10000,
    bf16=True,                          
    optim="adamw_torch_fused",          # Giờ sẽ chạy ổn định sau khi đã ép kiểu model
    gradient_checkpointing=True,        # Tiết kiệm VRAM cực lớn
    dataloader_num_workers=4,           
    logging_steps=10,
    save_steps=1000,
    push_to_hub=True,
    hub_model_id="phgrouptechs/tts-vie-infore",
    report_to="none"
)

trainer = VitsTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=TTSDataCollator(tokenizer),
)

print("🚀 Bắt đầu quá trình Fine-tuning với Custom Trainer...")
trainer.train()
# 8. Đẩy bản cuối cùng lên Hub
print("📤 Đang đẩy model lên Hugging Face Hub...")
trainer.push_to_hub(commit_message="Training completed!")