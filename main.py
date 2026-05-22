from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import requests
import tempfile
import os
import re
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("small")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = """Bạn là chuyên gia huấn luyện bán hàng. Dưới đây là transcript thô từ cuộc gọi bán hàng tiếng Việt.

Bước 1: Định dạng lại transcript thành hội thoại có 2 người:
- Nhân viên: (người chủ động giới thiệu, tư vấn sản phẩm/dịch vụ)
- Khách hàng: (người hỏi, phản đối, hoặc lắng nghe)

Bước 2: Phân tích và trả về JSON thuần (không markdown, không giải thích):
{
  "formatted_transcript": "Nhân viên: ...\nKhách hàng: ...\nNhân viên: ...",
  "customer_needs": ["nhu cầu 1", "nhu cầu 2"],
  "objections": [{"objection": "...", "how_handled": "...", "rating": "good|average|poor"}],
  "salesperson_performance": "nhận xét tổng thể",
  "coaching_tips": ["tip 1", "tip 2", "tip 3"],
  "overall_score": 7
}"""


class TranscribeRequest(BaseModel):
    url: str


class AnalyzeRequest(BaseModel):
    transcript: str


def download_drive_file(file_id):
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}
    response = session.get(url, params=params, stream=True)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            params["confirm"] = value
            response = session.get(url, params=params, stream=True)
            break
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
        suffix = ".m4a" if any(x in req.url for x in ["m4a", "audio"]) else ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in response.iter_content(chunk_size=32768):
                tmp.write(chunk)
            tmp_path = tmp.name
        result = model.transcribe(tmp_path, language="vi", task="transcribe")
        os.unlink(tmp_path)
        return {"text": result["text"], "segments": result["segments"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": req.transcript}],
            },
            timeout=60,
        )
        data = response.json()
        raw = data["content"][0]["text"]
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
