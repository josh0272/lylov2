import json
import os
import urllib.error
import urllib.request

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

VAPI_API_KEY = os.environ.get("VAPI_API_KEY", "").strip()
VAPI_PUBLIC_KEY = os.environ.get("VAPI_PUBLIC_KEY", "").strip()
VAPI_ASSISTANT_ID = os.environ.get(
    "VAPI_ASSISTANT_ID",
    "f3b4952e-ce7f-4c19-b20e-661e27b42f0f",
).strip()


class LyloChatRequest(BaseModel):
    message: str
    previousChatId: str | None = None


def _extract_vapi_error(raw_body: str, status_code: int) -> str:
    fallback = f"Vapi returned HTTP {status_code}."
    if not raw_body:
        return fallback

    try:
        detail = json.loads(raw_body)
    except Exception:
        text = raw_body.strip()
        return f"Vapi returned HTTP {status_code}: {text[:500]}" if text else fallback

    if isinstance(detail, str):
        return f"Vapi returned HTTP {status_code}: {detail}"

    if not isinstance(detail, dict):
        return f"Vapi returned HTTP {status_code}: {str(detail)[:500]}"

    candidates = [
        detail.get("message"),
        detail.get("error"),
        detail.get("detail"),
        detail.get("statusCode"),
    ]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return f"Vapi returned HTTP {status_code}: {candidate.strip()}"
        if isinstance(candidate, dict):
            nested = candidate.get("message") or candidate.get("detail") or candidate.get("error")
            if isinstance(nested, str) and nested.strip():
                return f"Vapi returned HTTP {status_code}: {nested.strip()}"
            return f"Vapi returned HTTP {status_code}: {json.dumps(candidate)[:500]}"
        if isinstance(candidate, list) and candidate:
            return f"Vapi returned HTTP {status_code}: {json.dumps(candidate)[:500]}"

    return f"Vapi returned HTTP {status_code}: {json.dumps(detail)[:500]}"


@router.get("/api/lylo-voice-config")
def lylo_voice_config():
    if not VAPI_PUBLIC_KEY:
        return JSONResponse(
            {
                "ok": False,
                "error": "Lylo voice calling is not connected yet. Add VAPI_PUBLIC_KEY to the server environment."
            },
            status_code=503,
        )

    if not VAPI_ASSISTANT_ID:
        return JSONResponse(
            {"ok": False, "error": "VAPI_ASSISTANT_ID is not configured."},
            status_code=503,
        )

    return {
        "ok": True,
        "publicKey": VAPI_PUBLIC_KEY,
        "assistantId": VAPI_ASSISTANT_ID,
    }


@router.post("/api/lylo-chat")
def lylo_chat(payload: LyloChatRequest):
    message = payload.message.strip()
    if not message:
        return JSONResponse({"ok": False, "error": "Please enter a message."}, status_code=400)

    if not VAPI_API_KEY:
        return JSONResponse(
            {
                "ok": False,
                "error": "Lylo chat is not connected yet. Add VAPI_API_KEY to the server environment."
            },
            status_code=503,
        )

    if not VAPI_ASSISTANT_ID:
        return JSONResponse(
            {"ok": False, "error": "VAPI_ASSISTANT_ID is not configured."},
            status_code=503,
        )

    body = {
        "assistantId": VAPI_ASSISTANT_ID,
        "input": message,
        "stream": False,
    }

    if payload.previousChatId:
        body["previousChatId"] = payload.previousChatId

    request = urllib.request.Request(
        "https://api.vapi.ai/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {VAPI_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
            "Accept-Language": "en-GB,en;q=0.9",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            raw_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw_body = ""
        return JSONResponse(
            {"ok": False, "error": _extract_vapi_error(raw_body, exc.code)},
            status_code=502,
        )
    except Exception:
        return JSONResponse(
            {"ok": False, "error": "Could not connect to Lylo right now. Please try again."},
            status_code=502,
        )

    output = data.get("output") or []
    reply = ""
    if output and isinstance(output[0], dict):
        reply = output[0].get("content") or ""

    if isinstance(reply, list):
        parts = []
        for part in reply:
            if isinstance(part, dict) and part.get("text"):
                parts.append(part["text"])
        reply = "\n".join(parts)

    return {
        "ok": True,
        "chatId": data.get("id"),
        "reply": str(reply).strip(),
    }
