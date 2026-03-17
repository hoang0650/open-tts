import os
import torch
from transformers import VitsTokenizer, VitsModel
from optimum.exporters.onnx import export_models, OnnxConfigWithPast
from optimum.exporters.tasks import TasksManager
from pathlib import Path

model_id = "phgrouptechs/tts-vie-infore"
save_dir = Path("./mms-tts-vie-onnx")
save_dir.mkdir(parents=True, exist_ok=True)

print("🚀 Đang tải model và chuẩn bị cấu hình...")
# Tải model và tokenizer
tokenizer = VitsTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id)

# Lấy cấu hình ONNX mặc định cho task text-to-speech
onnx_config_constructor = TasksManager.get_exporter_config_constructor(
    model_type="vits",
    exporter="onnx",
    task="text-to-speech",
)
onnx_config = onnx_config_constructor(model.config)

# Thực hiện export thủ công
print("🔄 Đang chuyển đổi sang ONNX (Bỏ qua validation)...")
export_models(
    models_and_onnx_configs={
        "model": (model, onnx_config)
    },
    output_dir=save_dir,
)

# Lưu tokenizer
tokenizer.save_pretrained(save_dir)

print(f"✅ HOÀN THÀNH! Các file đã nằm tại: {save_dir}")
print(os.listdir(save_dir))