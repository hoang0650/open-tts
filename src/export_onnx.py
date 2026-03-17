import os
from optimum.onnxruntime import ORTModelForTextToSpeech
from transformers import VitsTokenizer

model_id = "phgrouptechs/tts-vie-infore" # Model đã train của bạn
save_dir = "./mms-tts-vie-onnx"

# Tải và chuyển đổi sang ONNX tự động bằng Optimum
print("🔄 Đang chuyển đổi model sang ONNX...")
model = ORTModelForTextToSpeech.from_pretrained(model_id, export=True)
tokenizer = VitsTokenizer.from_pretrained(model_id)

# Lưu cục bộ
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# Đẩy bản ONNX lên một nhánh mới hoặc repo mới trên HF
model.push_to_hub(save_dir, repository_id=model_id, use_auth_token=os.getenv("HF_TOKEN"))
print("✅ Đã đẩy bản ONNX lên Hugging Face!")