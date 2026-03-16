# Hướng dẫn Train Vietnamese TTS trên RunPod

### Bước 1: Clone Repository và Cài đặt môi trường
Mở terminal trên RunPod và chạy:
```bash
cd /workspace
# Thay YOUR_GITHUB_REPO bằng link repo của bạn
git clone [https://github.com/YOUR_USERNAME/YOUR_GITHUB_REPO.git](https://github.com/YOUR_USERNAME/YOUR_GITHUB_REPO.git) tts-payment
cd tts-vi

# Cài đặt thư viện hệ thống và Python
apt-get update && apt-get install -y espeak-ng git-lfs wget
pip install -r requirements.txt

# Clone mã nguồn Piper để dùng tool train
cd /workspace
git clone [https://github.com/rhasspy/piper.git](https://github.com/rhasspy/piper.git)
cd piper/src/python
pip install build
pip install -r piper_train/requirements.txt
```
### Bước 2: Tải và Chuẩn bị Dataset
```bash
cd /workspace/open-tts
python scripts/download_data.py
```
### Bước 3: Tiền xử lý (Preprocessing) với Piper
```bash
cd /workspace/piper/src/python
python -m piper_train.preprocess \
    --language vi \
    --input-dir /workspace/my_dataset \
    --output-dir /workspace/training_data \
    --dataset-format ljspeech \
    --sample-rate 22050
```

### Bước 4: Fine-tune model (GPU)
Tải model gốc để train nhanh hơn (Fine-tune):
```bash
# Tải base model
wget [https://huggingface.co/rhasspy/piper-checkpoints/resolve/main/vi/vi_VN/vits1000/medium/epoch%3D1000.ckpt](https://huggingface.co/rhasspy/piper-checkpoints/resolve/main/vi/vi_VN/vits1000/medium/epoch%3D1000.ckpt) -O /workspace/base.ckpt

# Bắt đầu train
cd /workspace/piper/src/python
python -m piper_train \
    --dataset-dir /workspace/training_data \
    --accelerator 'gpu' \
    --devices 1 \
    --batch-size 16 \
    --max_epochs 1100 \
    --resume_from_checkpoint /workspace/base.ckpt
```
(Mẹo: Mở thêm 1 tab terminal khác chạy `tensorboard --logdir` /`workspace/piper/src/python/lightning_logs` để xem biểu đồ hội tụ)
### Bước 5: Xuất sang ONNX & Upload
Sau khi train xong, xuất model:
```bash
cd /workspace/piper/src/python
# Lưu ý: check lại thư mục version_0 hay version_1 tùy số lần bạn chạy train
python -m piper_train.export_onnx \
    /workspace/piper/src/python/lightning_logs/version_0/checkpoints/last.ckpt \
    /workspace/vi_vn_payment.onnx

# Push lên Hugging Face
cd /workspace/tts-payment
python scripts/push_to_hf.py
```
---

### 5. Cách triển khai nhanh trên RunPod
Bạn chỉ cần mở Terminal trên RunPod và gõ các lệnh sau để tạo file tự động:

1.  **Tạo các thư mục:** `mkdir -p scripts`
2.  **Tạo file script:** Dùng lệnh `cat <<EOF > scripts/download_data.py` rồi dán code mục 2 vào, sau đó gõ `EOF`. Làm tương tự cho các file khác.
3.  **Thực thi:** Làm theo các bước trong `README.md`.