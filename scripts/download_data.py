import os
import io
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

output_dir = "/workspace/my_dataset"
wav_dir = os.path.join(output_dir, "wavs")
os.makedirs(wav_dir, exist_ok=True)

print("--- Đang tải dataset infore1_25hours (Chế độ Manual Decode) ---")
# Sử dụng cast_column để tránh tự động decode bằng torchcodec
dataset = load_dataset("doof-ferb/infore1_25hours", split="train")

metadata_path = os.path.join(output_dir, "metadata.csv")

print(f"--- Đang xử lý {len(dataset)} tệp âm thanh ---")
with open(metadata_path, "w", encoding="utf-8") as f:
    for i, item in enumerate(tqdm(dataset)):
        file_id = f"vi_{i:05d}"
        wav_path = os.path.join(wav_dir, f"{file_id}.wav")
        
        try:
            # Lấy bytes dữ liệu âm thanh thô
            audio_bytes = item['audio']['bytes']
            
            # Giải mã bằng soundfile thông qua BytesIO
            data, samplerate = sf.read(io.BytesIO(audio_bytes))
            
            # Lưu file wav
            sf.write(wav_path, data, samplerate)
            
            # Lưu metadata
            text = item['transcript'].replace("\n", " ")
            f.write(f"{file_id}|{text}\n")
        except Exception as e:
            print(f"Lỗi tại tệp {i}: {e}")
            continue

print(f"--- Xong! Dataset lưu tại {output_dir} ---")