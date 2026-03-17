from fastapi import FastAPI
from transformers import VitsModel, VitsTokenizer
import torch
import io
import scipy.io.wavfile

app = FastAPI()

# Load model từ Hugging Face (CPU mode)
model_id = "phgrouptechs/tts-vie-infore"
tokenizer = VitsTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id)

@app.get("/tts")
async def tts(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    # Chuyển đổi sang file wav để trả về
    out_io = io.BytesIO()
    sampling_rate = model.config.sampling_rate
    scipy.io.wavfile.write(out_io, sampling_rate, output.cpu().numpy().squeeze())
    out_io.seek(0)
    
    return StreamingResponse(out_io, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)