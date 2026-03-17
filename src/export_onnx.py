import os
import torch
from transformers import VitsTokenizer, VitsModel
from optimum.exporters.onnx import export_models
from optimum.exporters.tasks import TasksManager
from pathlib import Path

# Nguồn dữ liệu
fine_tuned_model_id = "phgrouptechs/tts-vie-infore"
base_mms_id = "facebook/mms-tts-vie" # Dùng để lấy tokenizer chuẩn
save_dir = Path("./mms-tts-vie-onnx")
save_dir.mkdir(parents=True, exist_ok=True)

print("🚀 Đang tải tokenizer từ base model và model từ bản fine-tune...")
try:
    # 1. Load Tokenizer từ repo gốc (tránh lỗi NoneType vocab)
    tokenizer = VitsTokenizer.from_pretrained(base_mms_id)
    
    # 2. Load Model từ bản bạn đã train
    model = VitsModel.from_pretrained(fine_tuned_model_id)

    # 3. Chuẩn bị cấu hình ONNX
    print("⚙️ Đang chuẩn bị cấu hình ONNX cho VITS...")
    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        model_type="vits",
        exporter="onnx",
        task="text-to-speech",
    )
    onnx_config = onnx_config_constructor(model.config)

    # 4. Thực hiện export
    print("🔄 Đang chuyển đổi sang ONNX (No Validation)...")
    export_models(
        models_and_onnx_configs={
            "model": (model, onnx_config)
        },
        output_dir=save_dir,
    )

    # 5. Lưu tokenizer vào thư mục mới
    tokenizer.save_pretrained(save_dir)

    print(f"✅ HOÀN THÀNH! Model ONNX đã nằm tại: {save_dir}")
    print("Danh sách file:", os.listdir(save_dir))

except Exception as e:
    print(f"❌ Lỗi: {e}")