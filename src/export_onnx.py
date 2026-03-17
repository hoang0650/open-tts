import os
import torch
from transformers import VitsTokenizer, VitsModel
from optimum.exporters.onnx import export_models
from optimum.exporters.tasks import TasksManager
from pathlib import Path

# Cấu hình đường dẫn
fine_tuned_model_id = "phgrouptechs/tts-vie-infore"
base_mms_id = "facebook/mms-tts-vie" 
save_dir = Path("./mms-tts-vie-onnx")
save_dir.mkdir(parents=True, exist_ok=True)

print("🚀 Đang khởi tạo quá trình chuyển đổi...")

try:
    # 1. Load Tokenizer & Model
    tokenizer = VitsTokenizer.from_pretrained(base_mms_id)
    model = VitsModel.from_pretrained(fine_tuned_model_id)

    # 2. Lấy cấu hình ONNX với task chính xác: 'text-to-audio'
    print("⚙️ Đang cấu hình Backend ONNX cho task 'text-to-audio'...")
    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        model_type="vits",
        exporter="onnx",
        task="text-to-audio", # Đã sửa từ text-to-speech
        library_name="transformers"
    )
    onnx_config = onnx_config_constructor(model.config)

    # 3. Thực hiện export
    print("🔄 Đang ghi file ONNX (Vui lòng đợi trong giây lát)...")
    export_models(
        models_and_onnx_configs={
            "model": (model, onnx_config)
        },
        output_dir=save_dir,
    )

    # 4. Lưu tokenizer
    tokenizer.save_pretrained(save_dir)

    print(f"\n✅ THÀNH CÔNG RỰC RỠ!")
    print(f"📍 Model ONNX lưu tại: {save_dir.absolute()}")
    print(f"📦 Danh sách file: {os.listdir(save_dir)}")

except Exception as e:
    print(f"\n❌ Vẫn còn lỗi: {e}")