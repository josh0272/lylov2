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


def render_landing(template_path: str, live: bool = False):
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace(
        '<video controls playsinline preload="metadata" aria-label="Schedule of Loss demo"><source src="/static/schedule-of-loss.mp4" type="video/mp4"></video>',
        '<video muted loop playsinline controls preload="auto" aria-label="Schedule of Loss demo"><source src="/static/schedule-of-loss.mp4" type="video/mp4"></video>'
    )

    et1_original = '<section class="demo-section alt"><div class="demo-grid reveal"><div class="demo-copy"><h3>Turn case files into a draft form.</h3><p>Lylo is being designed to pull names, dates and case details from uploaded documents, fill the ET1 and prepare the information for review. You stay in control before it is used.</p><a class="demo-cta" href="/call#book">Book a 20 minute call</a></div><div class="demo-media"><div class="blank-video" aria-label="ET1 demo video space"></div></div></div></section>'
    et1_expanded = '''<section class="demo-section alt et1-section"><div class="demo-grid reveal"><div class="demo-copy"><h3>Turn case files into a draft form.</h3><p>Lylo is being designed to pull names, dates and case details from uploaded documents, fill the ET1 and prepare the information for review. You stay in control before it is used.</p><button class="demo-cta et1-suggest-jump" id="et1-suggest-jump" type="button">Suggest a form</button></div><div class="demo-media"><div class="blank-video" aria-label="ET1 demo video space"></div></div></div><div class="et1-extension reveal"><div class="et1-extension-head"><h4>One form is just the start.</h4><p>The same approach could be extended to other forms your firm uses.</p><p class="et1-potential-modules-label">Potential form modules include:</p></div><div class="et1-form-strip" aria-label="Examples of forms Lylo could be adapted to fill"><div class="et1-form-track"><div class="et1-form-set"><span class="et1-form-chip">ET1</span><span class="et1-form-chip">ET3</span><span class="et1-form-chip">N1 Claim Form</span><span class="et1-form-chip">N244 Application</span><span class="et1-form-chip">C100 Family Application</span><span class="et1-form-chip">Form E</span><span class="et1-form-chip">Simple Procedure</span><span class="et1-form-chip is-own">Your firm’s own forms</span></div><div class="et1-form-set" aria-hidden="true"><span class="et1-form-chip">ET1</span><span class="et1-form-chip">ET3</span><span class="et1-form-chip">N1 Claim Form</span><span class="et1-form-chip">N244 Application</span><span class="et1-form-chip">C100 Family Application</span><span class="et1-form-chip">Form E</span><span class="et1-form-chip">Simple Procedure</span><span class="et1-form-chip is-own">Your firm’s own forms</span></div></div></div><form class="et1-suggest" id="et1-form-suggest"><label for="et1-form-input">What form takes your firm too much time?</label><div class="et1-suggest-row"><input id="et1-form-input" name="form_suggestion" type="text" autocomplete="off" maxlength="140" placeholder="e.g. ET3, Form E, our client intake form…" aria-describedby="et1-form-status"><button type="submit">Suggest a form</button></div><div class="et1-suggest-status" id="et1-form-status" aria-live="polite"></div></form></div></section>'''
    html = html.replace(et1_original, et1_expanded)

    html = html.replace(
        '<div class="demo-media"><div class="blank-video" aria-label="ET1 demo video space"></div></div>',
        '<div class="demo-media"><video muted loop playsinline controls preload="auto" aria-label="ET1 demo"><source src="/static/et1.mp4" type="video/mp4"></video></div>'
    )

    html = html.replace(
        '<div class="call-number">07700 900 642</div>',
        '<button class="call-number lylo-voice-call lylo-desktop-call" id="lylo-voice-call" type="button">Call Lylo</button><a class="call-number call-number-link lylo-mobile-call" href="tel:+441416732902" aria-label="Call Lylo on 0141 673 2902">0141 673 2902</a><div class="lylo-voice-status" id="lylo-voice-status">Browser voice call · no phone number needed</div>'
    )

    html = html.replace("touch-action:pan-y", "touch-action:pan-x pan-y")
    html = html.replace(
        "if(e.pointerType==='mouse'&&e.button!==0)return;isDragging=true;",
        "if(e.pointerType!=='mouse'||e.button!==0)return;isDragging=true;"
    )
    html = html.replace(
        "if(!isDragging&&now>interactionUntil){regCarousel.scrollLeft+=dt*.032;wrapPosition()}",
        "if(window.innerWidth>979&&!isDragging&&now>interactionUntil){regCarousel.scrollLeft+=dt*.032;wrapPosition()}"
    )

    html = html.replace(
        "</head>",
        '<link rel="stylesheet" href="/static/et1-intake.css?v=2"></head>'
    )

    html = html.replace(
        "</body>",
        '<script src="/static/mobile-preview.js?v=11" defer></script><script src="/static/video-audio-state.js?v=7" defer></script><script src="/static/phone-demo.js?v=4" defer></script><script src="/static/et1-intake.js?v=3" defer></script><script id="preview-cta-script" src="/static/preview-cta.js?v=7" defer></script><script src="/static/lylo-voice.js?v=8" defer></script></body>'
    )

    if live:
        html = html.replace('/static/', '/static/live/')
        for name in ('mobile-preview.js','video-audio-state.js','phone-demo.js','et1-intake.js','preview-cta.js','lylo-voice.js'):
            html = html.replace(f'/static/live/{name}', f'/live-assets/{name}')
        html = html.replace('/live-assets/phone-demo.js?v=4', '/live-assets/phone-demo.js?v=5')

    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
def home():
    return render_landing("static/live/index.html", live=True)


@app.get("/about", response_class=FileResponse)
def about():
    return FileResponse("static/preview-about.html")


@app.get("/preview")
def preview_redirect():
    return RedirectResponse(url="/", status_code=308)


@app.get("/live-assets/{name}")
def live_asset(name: str):
    allowed = {
        'mobile-preview.js','video-audio-state.js','phone-demo.js',
        'et1-intake.js','preview-cta.js','lylo-voice.js'
    }
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Not found")
    path = os.path.join("static", "live", name)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().replace('/static/', '/static/live/')
    if name == 'preview-cta.js':
        content = content.replace("const aboutPath = 'static/preview-about.html';", "const aboutPath = '/about';")
    if name == 'phone-demo.js':
        content = content.replace(
            "      const naturalTypingDuration = Math.max(0.35, turn.text.length / TYPE_CHARS_PER_SECOND);\n      const typingDuration = Math.min(availableDuration, naturalTypingDuration);",
            "      const typingDuration = availableDuration;"
        )
    return Response(content=content, media_type="application/javascript")


@app.get("/call", response_class=HTMLResponse)
def call_page():
    with open("static/live/founding-pilot.html", "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace('href="/preview#', 'href="/#').replace('href="/preview"', 'href="/"')
    return HTMLResponse(content=html)


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


@app.get("/research", response_class=FileResponse)
def research():
    return FileResponse("static/research.html")


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
