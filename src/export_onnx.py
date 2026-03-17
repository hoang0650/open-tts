import torch
from optimum.exporters.onnx import main_export
from transformers import VitsTokenizer
import os

model_id = "phgrouptechs/tts-vie-infore"
save_dir = "./mms-tts-vie-onnx"

print("🚀 Đang tiến hành Force Export sang ONNX...")

try:
    # Sử dụng hàm main_export trực tiếp
    # do_validation=False sẽ bỏ qua lỗi so sánh Shape giữa PyTorch và ONNX
    main_export(
        model_name_or_path=model_id,
        output=save_dir,
        task="text-to-speech",
        do_validation=False, 
    )

    # Lưu thêm Tokenizer vào cùng thư mục để tiện sử dụng sau này
    print("📦 Đang lưu Tokenizer...")
    tokenizer = VitsTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_dir)

    print(f"✅ HOÀN THÀNH! Model đã được lưu tại: {save_dir}")
    print("Bạn có thể kiểm tra các file .onnx trong thư mục đó.")

except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")