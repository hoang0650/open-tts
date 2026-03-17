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

# 1. Xác thực Hugging Face (Sửa lại đoạn biến môi trường bị thiếu)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print("✅ Đã tìm thấy HF_TOKEN. Đang tiến hành đăng nhập...")
    login(token=hf_token)
else:
    print("❌ Không tìm thấy HF_TOKEN. Vui lòng kiểm tra lại biến môi trường.")

# 2. Cấu hình & Load Dataset
model_id = "facebook/mms-tts-vie"
dataset_id = "doof-ferb/infore1_25hours"

print(f"📦 Đang tải dataset: {dataset_id}...")
dataset = load_dataset(dataset_id, split="train")

# QUAN TRỌNG: Tắt tính năng tự động giải mã (decode=False)
# Thay vì cố gắng giải mã, thư viện chỉ trả về raw bytes của file audio
dataset = dataset.cast_column("audio", Audio(decode=False))

# 3. Load Model & Tokenizer
tokenizer = VitsTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id)

# 4. Tiền xử lý dữ liệu (Tự giải mã thủ công cực kỳ an toàn)
def prepare_dataset(batch):
    audio_data = batch["audio"]
    
    # Đọc dữ liệu nhị phân (bytes) thành mảng numpy bằng soundfile
    audio_bytes = audio_data["bytes"]
    array, sr = sf.read(io.BytesIO(audio_bytes))
    
    # Resample về 16000Hz nếu file gốc khác 16kHz (Bắt buộc cho MMS/VITS)
    if sr != 16000:
        array = librosa.resample(y=array, orig_sr=sr, target_sr=16000)

    # Chuyển văn bản thành ID
    batch["input_ids"] = tokenizer(batch["transcription"], return_tensors=None).input_ids
    
    # Lấy mảng waveform đã được xử lý
    batch["labels"] = array
    
    return batch

print("🛠 Đang tiền xử lý dữ liệu...")
dataset = dataset.map(
    prepare_dataset, 
    remove_columns=dataset.column_names, 
    num_proc=1 # Để 1 luồng để RAM không bị quá tải
)

# 5. Cấu hình Huấn luyện
training_args = TrainingArguments(
    output_dir="./mms-tts-vie-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    max_steps=10000, 
    logging_steps=50,
    save_steps=500,
    eval_strategy="no",
    fp16=torch.cuda.is_available(),
    push_to_hub=True, 
    hub_model_id="phgrouptechs/tts-vie-infore", 
    hub_token=hf_token,
    hub_strategy="every_save",
    report_to="none"
)

# 6. Khởi tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# 7. Chạy Huấn luyện
print("🚀 Bắt đầu quá trình Fine-tuning...")
trainer.train()

# 8. Đẩy bản cuối cùng lên Hub
print("📤 Đang đẩy model lên Hugging Face Hub...")
trainer.push_to_hub(commit_message="Training completed!")