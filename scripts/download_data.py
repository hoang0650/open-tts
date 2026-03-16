import os
import io
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# Cấu hình thư mục
output_dir = "/workspace/my_dataset"
wav_dir = os.path.join(output_dir, "wavs")
os.makedirs(wav_dir, exist_ok=True)

print("--- Đang tải dataset infore1_25hours (Chế độ Manual Bytes) ---")

# Tải dataset nhưng KHÔNG cho phép tự động decode audio
dataset = load_dataset("doof-ferb/infore1_25hours", split="train")
# Chuyển dataset sang dạng numpy/python để truy cập bytes thô dễ dàng hơn
dataset = dataset.with_format(None) 

metadata_path = os.path.join(output_dir, "metadata.csv")

print(f"--- Đang xử lý {len(dataset)} tệp âm thanh ---")
with open(metadata_path, "w", encoding="utf-8") as f:
    for i, item in enumerate(tqdm(dataset)):
        file_id = f"vi_{i:05d}"
        wav_path = os.path.join(wav_dir, f"{file_id}.wav")
        
        try:
            # Lấy bytes từ cột audio (dữ liệu thô từ file .parquet)
            audio_data = item['audio']
            
            # Nếu audio_data là dict và có key 'bytes'
            if isinstance(audio_data, dict) and 'bytes' in audio_data:
                byte_content = audio_data['bytes']
            else:
                # Trường hợp đặc biệt nếu format trả về khác
                continue
                
            # Đọc mảng byte và ghi trực tiếp ra file .wav
            # Cách này nhanh hơn vì không cần decode rồi encode lại
            with open(wav_path, "wb") as wav_file:
                wav_file.write(byte_content)
            
            # Lưu metadata
            text = item['transcript'].replace("\n", " ")
            f.write(f"{file_id}|{text}\n")
            
        except Exception as e:
            # print(f"Bỏ qua tệp {i} do lỗi: {e}")
            continue

print(f"\n--- Xong! Dataset lưu tại {output_dir} ---")