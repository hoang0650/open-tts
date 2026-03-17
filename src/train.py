import os
os.environ["HF_DATASETS_OFFLINE"] = "0"
import torch
from datasets import load_dataset, Audio
from transformers import (
    VitsModel, 
    VitsTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from huggingface_hub import login

# 2. Cấu hình & Load Dataset
model_id = "facebook/mms-tts-vie"
dataset_id = "doof-ferb/infore1_25hours"

print(f"📦 Đang tải dataset: {dataset_id}...")
# Tải dataset nhưng CHƯA giải mã audio ngay (decode=False)
dataset = load_dataset(dataset_id, split="train")

# 3. Load Model & Tokenizer (giữ nguyên)
tokenizer = VitsTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id)

# 4. Tiền xử lý dữ liệu (Sửa lại để tự giải mã bằng soundfile/librosa)
import librosa
import io

def prepare_dataset(batch):
    # Giải mã thủ công từ binary nếu cần, hoặc ép kiểu an toàn
    # Ở đây ta dùng cách cast an toàn sau khi đã gỡ torchcodec
    audio_data = batch["audio"]
    
    # Chuyển văn bản thành ID
    batch["input_ids"] = tokenizer(batch["transcription"], return_tensors=None).input_ids
    
    # Lấy mảng waveform (lúc này datasets sẽ dùng soundfile vì torchcodec đã bị xóa)
    batch["labels"] = audio_data["array"]
    return batch

print("🛠 Đang tiền xử lý dữ liệu...")

# Trước khi map, ép kiểu Audio lần cuối để dùng soundfile backend
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

dataset = dataset.map(
    prepare_dataset, 
    remove_columns=dataset.column_names, 
    num_proc=1 # Để num_proc=1 để dễ debug nếu còn lỗi, sau đó mới tăng lên
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
    hub_model_id="phgrouptechs/tts-vie-infore", # Thay đổi username của bạn
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