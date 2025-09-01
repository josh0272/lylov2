# app.py

# Make local ffmpeg visible without touching system PATH
import os, sys
ffmpeg_bin = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin")
if os.path.isdir(ffmpeg_bin) and ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI(title="Lylo — Local Transcription + Research Form")

# Choose a Whisper model: tiny, base, or small
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
# Compute type: int8 (CPU friendly), float16 (GPU), etc.
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")

# Load once at startup
model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE)

# Serve the /static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
def home():
    return FileResponse("static/index.html")

@app.get("/research", response_class=FileResponse)
def research():
    return FileResponse("static/research.html")

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), question_id: str = Form(None)):
    # Save upload to a temp file
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Transcribe (English; VAD helps trim silence)
    segments, info = model.transcribe(tmp_path, vad_filter=True, language="en", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()

    # Clean up
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return JSONResponse({"question_id": question_id, "transcript": text})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
