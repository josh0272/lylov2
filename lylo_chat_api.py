import json
import os
import urllib.error
import urllib.request

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

VAPI_API_KEY = os.environ.get("VAPI_API_KEY", "").strip()
VAPI_ASSISTANT_ID = os.environ.get(
    "VAPI_ASSISTANT_ID",
    "f3b4952e-ce7f-4c19-b20e-661e27b42f0f",
).strip()


class LyloChatRequest(BaseModel):
    message: str
    previousChatId: str | None = None


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
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            message = detail.get("message") or detail.get("error") or f"Vapi returned HTTP {exc.code}."
            if isinstance(message, dict):
                message = message.get("message") or str(message)
        except Exception:
            message = f"Vapi returned HTTP {exc.code}."
        return JSONResponse({"ok": False, "error": str(message)}, status_code=502)
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
