import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

VAPI_PUBLIC_KEY = os.environ.get("VAPI_PUBLIC_KEY", "").strip()
VAPI_ASSISTANT_ID = os.environ.get(
    "VAPI_ASSISTANT_ID",
    "f3b4952e-ce7f-4c19-b20e-661e27b42f0f",
).strip()
JESS_VAPI_ASSISTANT_ID = os.environ.get(
    "JESS_VAPI_ASSISTANT_ID",
    "ecb1ce5b-8fd1-4d66-8bb8-e243561f7977",
).strip()


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


@router.get("/api/jess-voice-config")
def jess_voice_config():
    if not VAPI_PUBLIC_KEY:
        return JSONResponse(
            {
                "ok": False,
                "error": "Lylo voice calling is not connected yet. Add VAPI_PUBLIC_KEY to the server environment."
            },
            status_code=503,
        )

    if not JESS_VAPI_ASSISTANT_ID:
        return JSONResponse(
            {"ok": False, "error": "JESS_VAPI_ASSISTANT_ID is not configured."},
            status_code=503,
        )

    return {
        "ok": True,
        "publicKey": VAPI_PUBLIC_KEY,
        "assistantId": JESS_VAPI_ASSISTANT_ID,
    }

