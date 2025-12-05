import io
import os
import hashlib
import time
import shutil
import base64
import torch
import torchaudio
import httpx
import asyncio
import logging
import subprocess
import tempfile
import soundfile as sf
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager
from functools import partial
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- IMPORTS ---
from inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
    load_audio, 
    get_speaker_latent_and_mask,
    get_text_input_ids_and_mask,
    sample_euler_cfg_independent_guidances,
    ae_decode,
    crop_audio_to_flattening_point,
    # NEW IMPORTS FOR SPEED
    compile_model,
    compile_fish_ae
)

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("echo_api")

# --- Global State ---
models = {}
inference_lock = asyncio.Lock()

# --- Request Model ---
class GenerateAudioRequest(BaseModel):
    gen_text: str
    ref_audio_url: str
    steps: int = 40
    cfg_text: float = 3.0
    cfg_speaker: float = 8.0
    truncation: float = 0.8 

# --- Helper Functions ---

def sanitize_audio(input_path: Path, output_path: Path):
    command = [
        "ffmpeg", "-y", "-i", str(input_path), "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", "-vn", str(output_path)
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info(f"SANITIZER: Cleaned {input_path.name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"SANITIZER: FFmpeg failed: {e.stderr}")
        raise RuntimeError(f"FFmpeg sanitization failed: {e.stderr}")

# --- Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SYSTEM: Loading Echo TTS models...")
    try:
        # 1. Load Eager Models
        models["model"] = load_model_from_hf(delete_blockwise_modules=True)
        models["fish_ae"] = load_fish_ae_from_hf() 
        models["pca_state"] = load_pca_state_from_hf()
        
        # 2. COMPILE (This adds ~1-2 mins to startup, but speeds up requests)
        # Note: If this crashes or uses too much VRAM, remove these lines.
        logger.info("SYSTEM: Compiling models for speed (this may take a minute)...")
        models["model"] = compile_model(models["model"])
        models["fish_ae"] = compile_fish_ae(models["fish_ae"])
        logger.info("SYSTEM: Compilation complete.")

        models["is_ready"] = True
        logger.info("SYSTEM: Echo TTS Ready.")
    except Exception as e:
        logger.error(f"SYSTEM: Model load/compile failed: {e}")
        models["is_ready"] = False
    yield
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/health")
async def health_check():
    if models.get("is_ready"):
        return "OK"
    return JSONResponse(content="Not Ready", status_code=503)

@app.post("/generate")
async def generate_audio(request: GenerateAudioRequest):
    logger.info("------------------------------------------------")
    logger.info(f"REQUEST: Text='{request.gen_text}'")
    
    if not models.get("is_ready"):
        raise HTTPException(status_code=503, detail="Model not initialized")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        raw_path = temp_path / "raw_download.tmp"
        clean_path = temp_path / "clean_input.wav"
        
        # 1. Download
        logger.info(f"DOWNLOAD: Fetching {request.ref_audio_url}...")
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(request.ref_audio_url, timeout=30.0, follow_redirects=True)
                resp.raise_for_status()
                with open(raw_path, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

        # 2. Sanitize
        try:
            await asyncio.to_thread(sanitize_audio, raw_path, clean_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # 3. Inference
        async with inference_lock:
            try:
                speaker_audio = load_audio(str(clean_path)).cuda()
                
                # --- FIXED PADDING FOR COMPILED MODEL ---
                # To use the compiled model efficiently, we MUST pad inputs to fixed sizes.
                # If we don't, torch.compile will re-compile for every new length (very slow).
                
                # Fixed Bucket: 6400 latent steps (approx 5 mins of reference max)
                speaker_latent, speaker_mask = get_speaker_latent_and_mask(
                    models["fish_ae"], 
                    models["pca_state"], 
                    speaker_audio,
                    max_speaker_latent_length=6400, # Explicit max bucket
                    pad_to_max=True                 # Force padding
                )

                # Fixed Bucket: 768 tokens (plenty for 30s of text)
                text_input_ids, text_mask = get_text_input_ids_and_mask(
                    [request.gen_text], 
                    max_length=768,                 # Explicit max bucket
                    device="cuda",
                    pad_to_max=True                 # Force padding
                )

                latent_out = sample_euler_cfg_independent_guidances(
                    model=models["model"],
                    speaker_latent=speaker_latent,
                    speaker_mask=speaker_mask,
                    text_input_ids=text_input_ids,
                    text_mask=text_mask,
                    rng_seed=0,
                    num_steps=request.steps,
                    cfg_scale_text=request.cfg_text,
                    cfg_scale_speaker=request.cfg_speaker,
                    cfg_min_t=0.5,
                    cfg_max_t=1.0,
                    truncation_factor=request.truncation,
                    rescale_k=None,
                    rescale_sigma=None,
                    speaker_kv_scale=None,
                    speaker_kv_max_layers=None,
                    speaker_kv_min_t=None,
                    sequence_length=640, # This is already fixed, so it's compile-friendly
                )

                audio_out = ae_decode(models["fish_ae"], models["pca_state"], latent_out)
                
                # Crop the padding silence
                audio_out = crop_audio_to_flattening_point(audio_out, latent_out[0])
                
                audio_tensor = audio_out[0].cpu()
                del speaker_audio
                torch.cuda.empty_cache()
                
                audio_np = audio_tensor.squeeze().numpy()
                
                buffer = io.BytesIO()
                sf.write(buffer, audio_np, 44100, format='wav')
                buffer.seek(0)
                audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                logger.info(f"RESPONSE: Success. Duration: {len(audio_np)/44100:.2f}s")
                return audio_base64

            except Exception as e:
                logger.error(f"Inference failed: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)