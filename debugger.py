import torch
import torchaudio
from inference import load_model_from_hf, load_fish_ae_from_hf, load_pca_state_from_hf, ae_reconstruct, load_audio
from pathlib import Path

# 1. Load just the Autoencoder (We don't need the big generation model)
print("Loading Autoencoder...")
fish_ae = load_fish_ae_from_hf()
pca_state = load_pca_state_from_hf()
fish_ae = fish_ae.cuda()

# 2. Define the URL or Path you are testing
# (Use the path to the 'clean_input.wav' if you can, or a known local file)
TEST_AUDIO_PATH = "reference_audio_cache/6100651d61d0d666a783db539e426791_clean.wav" 

# If you don't have a file yet, create a dummy one or download one manually
if not Path(TEST_AUDIO_PATH).exists():
    print(f"Please put a .wav file named '{TEST_AUDIO_PATH}' in this folder to test.")
    exit()

# 3. Load Audio using the Echo TTS loader (The potential suspect)
print(f"Loading {TEST_AUDIO_PATH} using inference.load_audio...")
try:
    # This uses torchcodec/AudioDecoder internally
    speaker_audio = load_audio(TEST_AUDIO_PATH).cuda()
    print(f"Loaded successfully. Shape: {speaker_audio.shape}")
except Exception as e:
    print(f"CRITICAL ERROR: load_audio failed! {e}")
    exit()

# 4. Reconstruct (Encode -> Decode)
print("Running Autoencoder Reconstruction...")
with torch.no_grad():
    # Pad it slightly to fit the AE's requirements (logic copied from Gradio app)
    padded_audio = torch.nn.functional.pad(
        speaker_audio[..., :2048 * 640],
        (0, max(0, 2048 * 640 - speaker_audio.shape[-1])),
    )[None]
    
    recon_audio = ae_reconstruct(
        fish_ae=fish_ae,
        pca_state=pca_state,
        audio=padded_audio
    )

# 5. Save the result
output_filename = "debug_reconstruction.wav"
torchaudio.save(output_filename, recon_audio[0].cpu(), 44100)
print(f"Saved reconstruction to {output_filename}")
print("LISTEN TO THIS FILE. If it sounds like static, your torchcodec/ffmpeg is broken.")