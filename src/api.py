import torch
import numpy as np
import soundfile as sf
from transformers import VitsTokenizer
# Thay đổi dòng import bị lỗi ở đây:
from optimum.onnxruntime import ORTModelForAudioSequence 

model_dir = "/workspace/open-tts/mms-tts-vie-onnx"

print("🚀 Đang nạp model ONNX để chạy API...")

# 1. Nạp model và tokenizer từ thư mục ONNX
# Nếu dùng GPU (RTX 4090), hãy thêm provider="CUDAExecutionProvider"
model = ORTModelForAudioSequence.from_pretrained(
    model_dir, 
    provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
)
tokenizer = VitsTokenizer.from_pretrained(model_dir)

def text_to_speech_onnx(text, output_path="output.wav"):
    # 2. Tiền xử lý văn bản
    inputs = tokenizer(text, return_tensors="pt")
    
    # 3. Chạy Inference (Cực nhanh trên ONNX)
    with torch.no_grad():
        # ONNX model trả về waveform trực tiếp
        outputs = model(**inputs)
    
    # 4. Lưu file âm thanh (VITS/MMS mặc định sampling_rate là 16000)
    waveform = outputs.waveform[0]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
        
    sf.write(output_path, waveform, 16000)
    return output_path

# Test thử
# text_to_speech_onnx("Xin chào, tôi là trí tuệ nhân tạo chạy trên nền tảng ô en en ích.")