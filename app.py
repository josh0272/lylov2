# app.py

import os, sys, tempfile, shutil, smtplib, ssl
from email.message import EmailMessage
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uvicorn

# --- Make local ffmpeg visible without touching system PATH (harmless if folder doesn't exist)
ffmpeg_bin = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin")
if os.path.isdir(ffmpeg_bin) and ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

app = FastAPI(title="Lylo — Local Transcription + Research Form")

# === Config (override via environment variables) ===
# Whisper
MODEL_SIZE   = os.environ.get("WHISPER_MODEL", "tiny")     # tiny/base/small
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")   # int8 (CPU), float16 (GPU), etc.

# Email (SMTP)
EMAIL_HOST = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "587"))      # 587 = STARTTLS
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_TO   = os.environ.get("EMAIL_TO", EMAIL_USER or "")

# CORS (if you later host the frontend elsewhere, add that origin here)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5500",
    # "https://your-frontend.example.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Whisper once at startup (cold start will pay this cost; /healthz can prewarm)
model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE)

# --- Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
def home():
    return FileResponse("static/index.html")

@app.get("/research", response_class=FileResponse)
def research():
    return FileResponse("static/research.html")

@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL_SIZE, "compute": COMPUTE_TYPE}

# --- Email helper
def send_email(subject: str, body: str):
    if not (EMAIL_HOST and EMAIL_PORT and EMAIL_USER and EMAIL_PASS and EMAIL_TO):
        raise RuntimeError("Email is not configured (missing EMAIL_* env vars).")
    msg = EmailMessage()
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=30) as server:
        server.starttls(context=context)
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

# --- API: questionnaire submit -> email
@app.post("/api/submit")
async def submit_questionnaire(
    name: str = Form(""),
    email: str = Form(""),
    answers: str = Form(""),      # JSON string from the frontend
    transcript: str = Form(""),   # optional combined transcript
):
    body = f"""New questionnaire submission

Name: {name}
Email: {email}

Answers:
{answers}

Transcript:
{transcript}
"""
    try:
        send_email(subject="New Questionnaire Submission", body=body)
        return {"ok": True, "message": "Submitted and emailed"}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# --- API: transcribe audio/video
@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), question_id: str = Form(None)):
    # Basic content-type guard (optional)
    if file.content_type and not any(file.content_type.startswith(p) for p in ("audio/", "video/")):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    tmp_path = None
    try:
        # Save upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Transcribe (English; VAD trims silence). beam_size=1 is faster on small CPUs.
        segments, info = model.transcribe(
            tmp_path,
            vad_filter=True,
            language="en",
            beam_size=1,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return JSONResponse({"ok": True, "question_id": question_id, "transcript": text})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    # Use $PORT if provided by the platform (Render sets this); default to 8000 locally.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
