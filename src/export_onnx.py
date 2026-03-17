from huggingface_hub import HfApi
import os

# Cấu hình thông tin
repo_id = "phgrouptechs/tts-vie-infore"  # Repo của bạn
local_folder = "/workspace/open-tts/mms-tts-vie-onnx"
token = os.getenv("HF_TOKEN") # Đảm bảo bạn đã export HF_TOKEN hoặc dán trực tiếp token vào đây

api = HfApi()

print(f"🚀 Đang chuẩn bị tải lên repo: {repo_id}...")

try:
    # Tải toàn bộ thư mục lên một thư mục con tên là 'onnx' trên repo
    # Hoặc bạn có thể để path_in_repo="" nếu muốn đè thẳng vào root
    api.upload_folder(
        folder_path=local_folder,
        repo_id=repo_id,
        path_in_repo="onnx", 
        token=token,
        commit_message="Add ONNX version for faster inference"
    )
    
    print(f"\n✅ ĐÃ PUSH THÀNH CÔNG!")
    print(f"🔗 Xem model tại: https://huggingface.co/{repo_id}/tree/main/onnx")

except Exception as e:
    print(f"❌ Lỗi khi push: {e}")