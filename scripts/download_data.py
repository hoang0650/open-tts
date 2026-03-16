import os
from datasets import load_dataset, Audio
from tqdm import tqdm

# 1. Định nghĩa thư mục
output_dir = "/workspace/my_dataset"
wav_dir = os.path.join(output_dir, "wavs")
os.makedirs(wav_dir, exist_ok=True)

print("--- Đang tải dataset (Vô hiệu hóa giải mã Audio) ---")

# 2. LOAD DATASET VỚI DECODE = FALSE
# Đây là chìa khóa: decode_audio=False ngăn chặn mọi nỗ lực nạp torchcodec
dataset = load_dataset("doof-ferb/infore1_25hours", split="train")
dataset = dataset.cast_column("audio", Audio(decode=False))

metadata_path = os.path.join(output_dir, "metadata.csv")

print(f"--- Đang trích xuất {len(dataset)} tệp âm thanh ---")

# 3. TRÍCH XUẤT BYTES
with open(metadata_path, "w", encoding="utf-8") as f:
    for i, item in enumerate(tqdm(dataset)):
        file_id = f"vi_{i:05d}"
        wav_path = os.path.join(wav_dir, f"{file_id}.wav")
        
        try:
            # Lúc này item['audio'] là một dict chứa 'bytes' thô
            raw_bytes = item['audio']['bytes']
            
            # Ghi trực tiếp ra file
            with open(wav_path, "wb") as wav_file:
                wav_file.write(raw_bytes)
            
            # Lưu metadata
            text = item['transcript'].replace("\n", " ")
            f.write(f"{file_id}|{text}\n")
            
        except Exception as e:
            continue

print(f"\n--- THÀNH CÔNG! Dữ liệu đã sẵn sàng tại {output_dir} ---")