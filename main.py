from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import requests
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("base")

class TranscribeRequest(BaseModel):
    url: str

@app.get("/")
def root():
    return {"status": "Whisper API running"}

@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    try:
        response = requests.get(req.url, stream=True, timeout=120)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download file")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        result = model.transcribe(tmp_path, language="vi", task="transcribe")
        os.unlink(tmp_path)
        return {"text": result["text"], "segments": result["segments"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))