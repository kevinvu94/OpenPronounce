"""Deploy OpenPronounce as a Web API on Modal.

    python -m modal secret create asep-pronounce-secret API_KEY=<your-key>
    python -m modal deploy modal_app.py

Every request except GET /health and CORS preflights must carry ``X-API-Key``.
Endpoints mirror server.py: POST /pronunciation, /free, /speech2text,
/phonemes, /tts and GET /languages, /health, /docs.
"""

import modal

APP_NAME = "asep-pronounce"
MODELS_DIR = "/models"

# ---------------------------------------------------------------- image ----


def _bake_models() -> None:
    """Runs at image build time: downloads both Wav2Vec2 checkpoints into the image."""
    from openpronounce.phones import _load_model as load_phone_model
    from openpronounce.speech import _load_models as load_speech_models

    load_speech_models()
    load_phone_model()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "espeak-ng", "libsndfile1")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .env({
        "HF_HOME": MODELS_DIR,
        "OPENPRONOUNCE_DEVICE": "cpu",
        "PYTHONUNBUFFERED": "1",
        # Distilled English phone model: hears closer to what learners say than the large
        # default, with its own word/score models picked automatically (docs: CHANGELOG 1.1.0).
        "OPENPRONOUNCE_PHONEME_MODEL": "OpenPronounce/wav2vec2-base-phones-en",
    })
    .add_local_file("pyproject.toml", "/build/pyproject.toml", copy=True)
    .add_local_file("README.md", "/build/README.md", copy=True)
    .add_local_dir("openpronounce", "/build/openpronounce", copy=True, ignore=["__pycache__", "*.pyc"])
    .run_commands("cd /build && pip install '.[app]'")
    .run_function(_bake_models)  # models live inside the image: no Volume needed
)

app = modal.App(APP_NAME, image=image)

# ------------------------------------------------------------------ API ----


def _build_api():
    import hmac
    import logging
    import os
    import subprocess
    import tempfile

    from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask

    from openpronounce import __version__, audio, free, speech
    from openpronounce.languages import DEFAULT_LANGUAGE, LANGUAGES, get_language

    log = logging.getLogger("openpronounce.modal")
    MAX_SECONDS = 600  # hard cap on decoded audio, far above the 1-2 minute use case

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = os.environ.get("API_KEY", "")
        if not expected or not x_api_key or not hmac.compare_digest(
            x_api_key.encode(), expected.encode()
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    api = FastAPI(
        title="OpenPronounce (ASEP)",
        version=__version__,
        description="Phoneme-level pronunciation assessment. Send X-API-Key on every call.",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # auth is by header, not cookies, so a wildcard is safe
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=86400,
    )

    def to_wav16k(upload: UploadFile):
        """Any phone/browser format (webm, m4a, mp3, ogg, wav...) -> mono 16 kHz WAV, then a waveform."""
        data = upload.file.read()
        if not data:
            raise HTTPException(status_code=422, detail="Empty audio file")
        src = tempfile.NamedTemporaryFile(prefix="up-", suffix=os.path.splitext(upload.filename or "")[1] or ".bin", delete=False)
        dst = src.name + ".16k.wav"
        try:
            src.write(data)
            src.close()
            proc = subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", src.name,
                 "-vn", "-t", str(MAX_SECONDS), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst],
                capture_output=True, timeout=180,
            )
            if proc.returncode != 0:
                log.error("ffmpeg failed: %s", proc.stderr.decode(errors="replace")[-500:])
                raise HTTPException(status_code=415, detail="Unsupported or corrupt audio file")
            return audio.load(dst)
        finally:
            for p in (src.name, dst):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def lang_code(lang: str) -> str:
        try:
            return get_language(lang).code
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    def fail(what: str):
        log.exception("%s failed", what)
        raise HTTPException(status_code=500, detail="Something went wrong")

    auth = [Depends(require_api_key)]

    # Plain `def` routes run in a threadpool, so long recordings never block the event loop.
    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.get("/languages", dependencies=auth)
    def languages():
        return {
            "default": DEFAULT_LANGUAGE,
            "languages": [{"code": l.code, "name": l.name} for l in LANGUAGES.values()],
        }

    @api.post("/pronunciation", dependencies=auth)
    def pronunciation(file: UploadFile = File(...), expected_text: str = Form(...), lang: str = Form(DEFAULT_LANGUAGE)):
        lang = lang_code(lang)
        sound = to_wav16k(file)
        try:
            return speech.compare_audio_with_text(sound, expected_text, lang=lang)
        except HTTPException:
            raise
        except Exception:
            fail("pronunciation")

    @api.post("/free", dependencies=auth)
    def free_speech(file: UploadFile = File(...), lang: str = Form(DEFAULT_LANGUAGE)):
        lang = lang_code(lang)
        sound = to_wav16k(file)
        try:
            return free.assess(sound, lang=lang)
        except Exception:
            fail("free speech")

    @api.post("/speech2text", dependencies=auth)
    def speech2text(file: UploadFile = File(...), lang: str = Form(DEFAULT_LANGUAGE)):
        lang = lang_code(lang)
        sound = to_wav16k(file)
        try:
            return {"transcript": speech.transcribe(sound, lang)}
        except Exception:
            fail("transcription")

    @api.post("/phonemes", dependencies=auth)
    def phonemes(text: str = Form(...), lang: str = Form(DEFAULT_LANGUAGE)):
        lang = lang_code(lang)
        try:
            phones, words = speech.get_phonemes_with_word_mapping(text, lang)
            return {"phonemes": phones, "words": list(words.values())}
        except Exception:
            fail("phonemes")

    @api.post("/tts", dependencies=auth)
    def tts(text: str = Form(...), lang: str = Form(DEFAULT_LANGUAGE)):
        lang = lang_code(lang)
        try:
            return FileResponse(audio.text2speech(text, lang=lang), media_type="audio/wav")
        except Exception:
            fail("tts")

    return api


@app.cls(
    cpu=4.0,
    memory=8192,
    scaledown_window=120,  # back to zero containers after 2 idle minutes
    timeout=900,           # room for 1-2 minute recordings on CPU
    secrets=[modal.Secret.from_name("asep-pronounce-secret")],
)
class Api:
    @modal.enter()
    def warm_up(self):
        # Load both models into RAM once per container, before the first request.
        from openpronounce.phones import _load_model as load_phone_model
        from openpronounce.speech import _load_models as load_speech_models

        load_speech_models()
        load_phone_model()

    @modal.asgi_app(label="asep-pronounce")
    def web(self):
        return _build_api()
