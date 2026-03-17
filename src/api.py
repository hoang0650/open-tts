import os
import io
import torch
import scipy.io.wavfile
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from optimum.onnxruntime import ORTModelForTextToSpeech
from transformers import VitsTokenizer

app = FastAPI()

# Cấu hình Model ID từ HF
# Lưu ý: Model ID này phải chứa các file .onnx đã export ở bước trên
MODEL_ID = os.getenv("MODEL_ID", "phgrouptechs/tts-vie-infore")

print("🚀 Đang tải ONNX model lên CPU...")
# Load model ONNX chuyên dụng cho Inference trên CPU
tokenizer = VitsTokenizer.from_pretrained(MODEL_ID)
model = ORTModelForTextToSpeech.from_pretrained(MODEL_ID, provider="CPUExecutionProvider")

@app.get("/tts")
async def tts(text: str):
    try:
        inputs = tokenizer(text, return_tensors="pt")
        
        # Inference với ONNX Runtime
        outputs = model(**inputs)
        waveform = outputs.waveform[0]

        # Chuyển sang buffer audio
        out_io = io.BytesIO()
        sampling_rate = model.config.sampling_rate
        scipy.io.wavfile.write(out_io, sampling_rate, waveform.cpu().numpy())
        out_io.seek(0)
        
        return StreamingResponse(out_io, media_type="audio/wav")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))