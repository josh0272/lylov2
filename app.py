# app.py

import os, sys, tempfile, shutil, smtplib, ssl
from email.message import EmailMessage
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from lylo_chat_api import router as lylo_chat_router
import uvicorn

ffmpeg_bin = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin")
if os.path.isdir(ffmpeg_bin) and ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

app = FastAPI(title="Lylo — Local Transcription + Research Form")

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")

EMAIL_HOST = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "587"))
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER or "")

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lylo_chat_router)
model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE)
app.mount("/static", StaticFiles(directory="static"), name="static")


HTML_REVALIDATE_HEADERS = {
    "Cache-Control": "no-cache, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


def html_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), headers=HTML_REVALIDATE_HEADERS)


@app.middleware("http")
async def cache_policy(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path.startswith("/live-assets/"):
        if request.query_params.get("v"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "public, max-age=3600"
    return response


@app.get("/", response_class=HTMLResponse)
def home():
    return html_file("static/live/index.html")


@app.get("/about", response_class=HTMLResponse)
def about():
    return html_file("static/preview-about.html")


@app.get("/preview", response_class=HTMLResponse)
def preview():
    return html_file("static/live/preview.html")


@app.get("/live-assets/{name}")
def live_asset(name: str):
    allowed = {
        'mobile-preview.js', 'video-audio-state.js', 'phone-demo.js',
        'et1-intake.js', 'lylo-voice.js'
    }
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Not found")
    path = os.path.join("static", "live", name)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().replace('/static/', '/static/live/')
    if name == 'phone-demo.js':
        content = content.replace(
            "      const naturalTypingDuration = Math.max(0.35, turn.text.length / TYPE_CHARS_PER_SECOND);\n      const typingDuration = Math.min(availableDuration, naturalTypingDuration);",
            "      const typingDuration = availableDuration;"
        )
    return Response(
        content=content,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/call", response_class=HTMLResponse)
def call_page():
    return html_file("static/live/founding-pilot.html")


@app.get("/founding-pilot")
def founding_pilot_redirect():
    return RedirectResponse(url="/call", status_code=308)


@app.get("/preview/call")
def preview_call_redirect():
    return RedirectResponse(url="/call", status_code=308)


@app.get("/preview/founding-pilot")
def preview_founding_pilot_redirect():
    return RedirectResponse(url="/call", status_code=308)


@app.get("/call-lylo-out-of-hours-demo-record-for-jess", response_class=FileResponse)
def jess_out_of_hours_demo():
    response = FileResponse("static/jess-out-of-hours.html")
    response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/research", response_class=HTMLResponse)
def research():
    return html_file("static/research.html")


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return html_file("static/privacy.html")


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL_SIZE, "compute": COMPUTE_TYPE}


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


@app.post("/api/submit")
async def submit_questionnaire(
    name: str = Form(""),
    email: str = Form(""),
    answers: str = Form(""),
    transcript: str = Form(""),
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


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), question_id: str = Form(None)):
    if file.content_type and not any(file.content_type.startswith(p) for p in ("audio/", "video/")):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
