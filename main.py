from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisper
import requests
import tempfile
import os

app = FastAPI()
model = whisper.load_model("base")  # use "small" for better Vietnamese accuracy

class TranscribeRequest(BaseModel):
    url: str  # Google Drive direct download URL

@app.get("/")
def root():
    return {"status": "Whisper API running"}

@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    try:
        # Download the file from Google Drive
        response = requests.get(req.url, stream=True, timeout=120)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download file")

        # Save to temp file
        suffix = ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        # Transcribe with Whisper
        result = model.transcribe(tmp_path, language="vi", task="transcribe")
        os.unlink(tmp_path)

        return {
            "text": result["text"],
            "segments": result["segments"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))