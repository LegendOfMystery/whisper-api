from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import requests
import tempfile
import os
import re

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

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def download_drive_file(file_id):
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}
    response = session.get(url, params=params, stream=True)
    token = get_confirm_token(response)
    if token:
        params["confirm"] = token
        response = session.get(url, params=params, stream=True)
    return response

@app.get("/")
def root():
    return {"status": "Whisper API running"}

@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    try:
        match = re.search(r"id=([a-zA-Z0-9_-]+)", req.url)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid Drive URL")
        file_id = match.group(1)

        response = download_drive_file(file_id)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download file")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            for chunk in response.iter_content(chunk_size=32768):
                tmp.write(chunk)
            tmp_path = tmp.name

        result = model.transcribe(tmp_path, language="vi", task="transcribe")
        os.unlink(tmp_path)
        return {"text": result["text"], "segments": result["segments"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))