import os
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm

# Cố định thư mục chứa data trên RunPod
output_dir = "/workspace/my_dataset"
wav_dir = os.path.join(output_dir, "wavs")
os.makedirs(wav_dir, exist_ok=True)

print("--- Đang tải dataset infore1_25hours từ Hugging Face ---")
dataset = load_dataset("doof-ferb/infore1_25hours", split="train")

metadata_path = os.path.join(output_dir, "metadata.csv")

print(f"--- Đang xử lý {len(dataset)} tệp âm thanh ---")
with open(metadata_path, "w", encoding="utf-8") as f:
    for i, item in enumerate(tqdm(dataset)):
        file_id = f"vi_{i:05d}"
        wav_path = os.path.join(wav_dir, f"{file_id}.wav")
        
        # Lưu file audio
        sf.write(wav_path, item['audio']['array'], item['audio']['sampling_rate'])
        
        # Lưu metadata
        text = item['transcript'].replace("\n", " ")
        f.write(f"{file_id}|{text}\n")

print(f"--- Xong! Dataset lưu tại {output_dir} ---")