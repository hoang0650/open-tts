# Vietnamese MMS-TTS Fine-tuning

Dự án này thực hiện fine-tune model `facebook/mms-tts-vie` với bộ dữ liệu `doof-ferb/infore1_25hours` và triển khai API trên Railway.

## 🚀 Tính năng
- Fine-tune kiến trúc VITS (MMS) cho giọng đọc tiếng Việt.
- Tự động đẩy model lên Hugging Face Hub.
- API inference tối ưu cho CPU sử dụng FastAPI.

## 🛠 Cài đặt
1. Clone repo: `git clone <your-repo-url>`
2. Cài đặt thư viện: `pip install -r requirements.txt`

## 📈 Huấn luyện
Chạy file `src/train.py` hoặc sử dụng notebook trong thư mục `notebooks/`.

## 🌐 Triển khai (Railway)
Dự án được cấu hình để chạy bằng Docker trên Railway, hỗ trợ hoàn toàn CPU.
