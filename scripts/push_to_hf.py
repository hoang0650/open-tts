import os
from huggingface_hub import HfApi

api = HfApi()
repo_id = "phgrouptechs/tts-vi"
HF_TOKEN = os.getenv("HF_TOKEN")

model_onnx = "/workspace/tts-vi.onnx"
model_json = "/workspace/tts-vi.onnx.json"

api.create_repo(repo_id=repo_id, token=HF_TOKEN, exist_ok=True)
api.upload_file(path_or_fileobj=model_onnx, path_in_repo="tts-vi.onnx", repo_id=repo_id, token=HF_TOKEN)
api.upload_file(path_or_fileobj=model_json, path_in_repo="tts-vi.onnx.json", repo_id=repo_id, token=HF_TOKEN)
print(f"Thành công! Xem tại: https://huggingface.co/{repo_id}")