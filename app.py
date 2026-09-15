# app.py

import os, sys, tempfile, shutil, smtplib, ssl
from email.message import EmailMessage
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
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


@app.get("/", response_class=FileResponse)
def home():
    return FileResponse("static/index.html")


@app.get("/preview", response_class=HTMLResponse)
def preview():
    with open("static/preview/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Keep these videos fully preloadable. Mobile playback itself is controlled by mobile-preview.js.
    html = html.replace(
        '<video controls playsinline preload="metadata" aria-label="Schedule of Loss demo"><source src="/static/schedule-of-loss.mp4" type="video/mp4"></video>',
        '<video muted loop playsinline controls preload="auto" aria-label="Schedule of Loss demo"><source src="/static/schedule-of-loss.mp4" type="video/mp4"></video>'
    )

    html = html.replace(
        '<div class="demo-media"><div class="blank-video" aria-label="ET1 demo video space"></div></div>',
        '<div class="demo-media"><video muted loop playsinline controls preload="auto" aria-label="ET1 demo"><source src="/static/et1.mp4" type="video/mp4"></video></div>'
    )

    # Keep the ET1 demo as the main feature, then lightly show that the same workflow can be adapted to other forms.
    et1_original = '<section class="demo-section alt"><div class="demo-grid reveal"><div class="demo-copy"><h3>Turn case files into a completed form.</h3><p>Lylo pulls names, dates and case details from uploaded documents, fills the ET1 and prepares the information for review. You stay in control before it is used.</p><a class="demo-cta" href="/research">Automate a form</a></div><div class="demo-media"><div class="blank-video" aria-label="ET1 demo video space"></div></div></div></section>'
    et1_expanded = '''<section class="demo-section alt et1-section"><div class="demo-grid reveal"><div class="demo-copy"><h3>Turn case files into a completed form.</h3><p>Lylo pulls names, dates and case details from uploaded documents, fills the ET1 and prepares the information for review. You stay in control before it is used.</p><button class="demo-cta et1-suggest-jump" id="et1-suggest-jump" type="button">Suggest a form</button></div><div class="demo-media"><div class="blank-video" aria-label="ET1 demo video space"></div></div></div><div class="et1-extension reveal"><div class="et1-extension-head"><h4>One form is just the start.</h4><p>Lylo can be built to fill the forms your firm uses every day.</p></div><div class="et1-form-strip" aria-label="Examples of forms Lylo could be adapted to fill"><div class="et1-form-track"><div class="et1-form-set"><span class="et1-form-chip">ET1</span><span class="et1-form-chip">ET3</span><span class="et1-form-chip">N1 Claim Form</span><span class="et1-form-chip">N244 Application</span><span class="et1-form-chip">C100 Family Application</span><span class="et1-form-chip">Form E</span><span class="et1-form-chip">Simple Procedure</span><span class="et1-form-chip is-own">Your firm’s own forms</span></div><div class="et1-form-set" aria-hidden="true"><span class="et1-form-chip">ET1</span><span class="et1-form-chip">ET3</span><span class="et1-form-chip">N1 Claim Form</span><span class="et1-form-chip">N244 Application</span><span class="et1-form-chip">C100 Family Application</span><span class="et1-form-chip">Form E</span><span class="et1-form-chip">Simple Procedure</span><span class="et1-form-chip is-own">Your firm’s own forms</span></div></div></div><form class="et1-suggest" id="et1-form-suggest"><label for="et1-form-input">What form takes your firm too much time?</label><div class="et1-suggest-row"><input id="et1-form-input" name="form_suggestion" type="text" autocomplete="off" maxlength="140" placeholder="e.g. ET3, Form E, our client intake form…" aria-describedby="et1-form-status"><button type="submit">Suggest a form</button></div><div class="et1-suggest-status" id="et1-form-status" aria-live="polite"></div></form></div></section>'''
    html = html.replace(et1_original, et1_expanded)

    # Desktop uses the browser voice demo. Mobile shows the real Twilio number and opens the native dialler.
    html = html.replace(
        '<div class="call-number">07700 900 642</div>',
        '<button class="call-number lylo-voice-call lylo-desktop-call" id="lylo-voice-call" type="button">Call Lylo</button><a class="call-number call-number-link lylo-mobile-call" href="tel:+441416732902" aria-label="Call Lylo on 0141 673 2902">0141 673 2902</a><div class="lylo-voice-status" id="lylo-voice-status">Browser voice call · no phone number needed</div>'
    )

    # Native touch scrolling on mobile; keep desktop mouse drag logic from taking touch pointers.
    html = html.replace("touch-action:pan-y", "touch-action:pan-x pan-y")
    html = html.replace(
        "if(e.pointerType==='mouse'&&e.button!==0)return;isDragging=true;",
        "if(e.pointerType!=='mouse'||e.button!==0)return;isDragging=true;"
    )

    # Desktop keeps the page's original carousel animation. Mobile uses mobile-preview.js instead.
    html = html.replace(
        "if(!isDragging&&now>interactionUntil){regCarousel.scrollLeft+=dt*.032;wrapPosition()}",
        "if(window.innerWidth>979&&!isDragging&&now>interactionUntil){regCarousel.scrollLeft+=dt*.032;wrapPosition()}"
    )

    mobile_overrides = """
    <link rel="stylesheet" href="/static/et1-intake.css?v=2">
    <style id="lylo-mobile-overrides">
      .call-number-link {
        color: inherit;
        text-decoration: none;
        font: inherit;
        letter-spacing: inherit;
      }

      .lylo-mobile-call {
        display: none;
      }

      .lylo-voice-call {
        appearance: none;
        -webkit-appearance: none;
        font-family: inherit;
        cursor: pointer;
        transition: transform .2s ease, border-color .2s ease, background .2s ease, color .2s ease;
      }

      .lylo-voice-call:hover {
        transform: translateY(-2px);
        border-color: rgba(255,255,255,.24);
        background: rgba(12,24,41,.72);
      }

      .lylo-voice-call:disabled {
        cursor: wait;
        opacity: .72;
        transform: none;
      }

      .lylo-voice-call.is-active {
        border-color: rgba(255,170,170,.22);
        color: #f3dede;
      }

      .lylo-voice-status {
        min-height: 18px;
        margin-top: 10px;
        color: #748398;
        font-size: 11px;
        line-height: 1.45;
        text-align: center;
      }

      @media (max-width: 979px) {
        .demo-grid,
        .demo-section.alt .demo-grid,
        .demo-section.alt.phone-section .demo-grid {
          grid-template-columns: 1fr !important;
          gap: 30px !important;
        }

        .demo-section .demo-media,
        .demo-section.alt .demo-media {
          order: 1 !important;
        }

        .demo-section .demo-copy,
        .demo-section.alt .demo-copy {
          order: 2 !important;
        }

        .lylo-desktop-call,
        .lylo-voice-status {
          display: none !important;
        }

        .lylo-mobile-call,
        .lylo-mobile-call:link,
        .lylo-mobile-call:visited,
        .lylo-mobile-call:hover,
        .lylo-mobile-call:active {
          display: inline-flex !important;
          align-items: center !important;
          justify-content: center !important;
          color: #dce5ef !important;
          text-decoration: none !important;
          -webkit-text-fill-color: #dce5ef !important;
        }

        .phone-section .audio-stage {
          order: 1 !important;
          width: 100% !important;
          max-width: 100% !important;
          height: 210px !important;
          min-height: 0 !important;
          padding: 8px 0 0 !important;
          align-items: flex-start !important;
          overflow: visible !important;
          transition: height .48s cubic-bezier(.22,.78,.24,1) !important;
        }

        .phone-section .audio-stage:has(.audio-demo.started),
        .phone-section .audio-stage.phone-demo-active {
          height: 445px !important;
        }

        .phone-section .audio-demo {
          width: 100% !important;
          max-width: 100% !important;
          padding: 0 8px !important;
          transition: transform .48s cubic-bezier(.22,.78,.24,1) !important;
        }

        .phone-section .audio-demo.started {
          transform: translateY(-8px) !important;
        }

        .phone-section .audio-wave {
          height: 66px !important;
          margin-bottom: 18px !important;
        }

        .phone-section .audio-title {
          margin-bottom: 10px !important;
        }

        .phone-section .transcript {
          top: calc(100% + 18px) !important;
          height: 190px !important;
          width: min(94%, 430px) !important;
        }

        .phone-section .demo-copy {
          order: 2 !important;
          width: 100% !important;
          max-width: 100% !important;
        }

        .reg-cards {
          width: calc(100vw - 24px) !important;
          max-width: none !important;
          gap: 12px !important;
          padding: 6px 8px 10px !important;
          overflow-x: auto !important;
          overflow-y: hidden !important;
          -webkit-overflow-scrolling: touch !important;
          overscroll-behavior-x: contain !important;
          touch-action: pan-x pan-y !important;
          scroll-behavior: auto !important;
          cursor: default !important;
        }

        .reg-cards.dragging {
          cursor: default !important;
        }

        .reg-card {
          flex: 0 0 min(72vw, 238px) !important;
          min-height: 138px !important;
          padding: 17px 15px !important;
          border-radius: 15px !important;
        }

        .reg-card strong {
          font-size: 14px !important;
          margin-bottom: 6px !important;
        }

        .reg-card span {
          font-size: 12px !important;
          line-height: 1.4 !important;
          max-width: 202px !important;
        }

        .reg-card em {
          font-size: 11px !important;
          line-height: 1.4 !important;
          margin-top: 8px !important;
          padding-top: 8px !important;
          max-width: 205px !important;
        }

        .privacy-visual {
          display: flex !important;
          flex-direction: column !important;
          gap: 8px !important;
          width: min(100%, 430px) !important;
          max-width: 430px !important;
          margin: 30px auto 0 !important;
        }

        .privacy-node {
          width: 100% !important;
          min-height: 0 !important;
          padding: 14px 16px !important;
          border-radius: 14px !important;
          display: flex !important;
          flex-direction: column !important;
          align-items: center !important;
          justify-content: center !important;
        }

        .privacy-node strong {
          font-size: 14px !important;
          margin-bottom: 4px !important;
        }

        .privacy-node span {
          font-size: 12px !important;
          line-height: 1.4 !important;
          max-width: 320px !important;
        }
      }
    </style>
    """
    html = html.replace("</head>", mobile_overrides + "</head>")

    html = html.replace(
        "</body>",
        '<script src="/static/mobile-preview.js?v=10" defer></script><script src="/static/video-audio-state.js?v=7" defer></script><script src="/static/phone-demo.js?v=4" defer></script><script src="/static/et1-intake.js?v=2" defer></script><script id="preview-cta-script" src="/static/preview-cta.js?v=2" defer></script><script src="/static/lylo-voice.js?v=5" defer></script></body>'
    )

    return HTMLResponse(content=html)


@app.get("/founding-pilot", response_class=FileResponse)
def founding_pilot():
    return FileResponse("static/founding-pilot.html")


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
