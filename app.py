import os
import time
import torch
import random
import numpy as np
import gradio as gr
from core.tts import ChatterboxTTS

torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"🚀 Running on device: {DEVICE}")

MODEL = None

PRIMARY_AVATARS = os.path.join("core", "avatars")
FALLBACK_AVATARS = os.path.join("/", "content", "drive", "MyDrive", "chatterbox", "core", "avatars")
AVATAR_PATH = PRIMARY_AVATARS if os.path.isdir(PRIMARY_AVATARS) else FALLBACK_AVATARS
AVATAR = "Alexa"

def load_model():
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        MODEL = ChatterboxTTS.from_pretrained(DEVICE)
        if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
            MODEL.to(DEVICE)
        print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
    return MODEL

def list_avatars(avatar_dir=AVATAR_PATH, preferred=AVATAR):
    files = [
        f for f in os.listdir(avatar_dir)
        if os.path.isfile(os.path.join(avatar_dir, f)) and f.lower().endswith(('.wav', '.flac', '.mp3'))
    ]
    avatar_map = {os.path.splitext(f)[0].capitalize(): os.path.join(avatar_dir, f) for f in files}
    options = sorted(avatar_map.keys(), key=lambda x: x.lower())
    default = preferred if preferred in avatar_map else (options[0] if options else None)
    return avatar_map, options, default

def set_seed(seed: int):
    seed = int(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_audio(
    text_input: str,
    avatar_name: str,
    exaggeration_input: float,
    cfgw_input: float,
    temperature_input: float,
    seed_num_input: float
) -> tuple[int, np.ndarray]:

    current_model = load_model()
    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if int(seed_num_input) != 0:
        set_seed(int(seed_num_input))

    avatar_map, _, _ = list_avatars()
    avatar_path_input = avatar_map.get(avatar_name) if avatar_name else None

    print(f"Generating audio for text: '{text_input[:30]}...'")

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        wav = current_model.generate(
            text_input[:1000],
            audio_prompt_path=avatar_path_input,
            exaggeration=exaggeration_input,
            temperature=temperature_input,
            cfg_weight=cfgw_input,
        )

    print("Audio generation complete.")
    wav_np = wav.squeeze(0).cpu().numpy().astype(np.float32)
    return current_model.sr, wav_np

avatar_map, avatar_list, default_avatar = list_avatars()

with gr.Blocks(
    title="Chatterbox TTS",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="teal", neutral_hue="slate"),
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
    body, .gradio-container {font-family: 'Poppins', sans-serif;}
    .gradio-container {padding: 2rem;}
    """
) as demo:

    gr.Markdown("""
    ## Chatterbox TTS  
    **A streamlined, high-quality text-to-speech tool built for clarity, expressiveness, and control.**
    """)

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="📝 Text to Synthesize",
                placeholder="Enter text to synthesize (optimal: 400–950 chars)...",
                lines=8,
                max_lines=8,
            )

            ref_avatar = gr.Dropdown(
                choices=avatar_list,
                label="🎭 Voice Avatar",
                value=default_avatar,
                info="Select the voice character"
            )

            with gr.Accordion("⚙️ Advanced Options", open=False):
                exaggeration = gr.Slider(0.25, 2, step=0.05, label="Exaggeration", value=0.5)
                cfg_weight = gr.Slider(0.2, 1, step=0.05, label="CFG / Pace", value=0.5)
                temp = gr.Slider(0.05, 5, step=0.05, label="Temperature", value=0.8)
                seed_num = gr.Number(value=0, label="Seed (0 = random)")

            error_display = gr.Markdown(visible=False)
            run_btn = gr.Button("Generate Speech", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(
                label="Output Audio",
                type="numpy",
                interactive=False,
                autoplay=False,
                show_download_button=True
            )

            gr.Markdown("""
            ### 📋 Model Summary
            **Current Constraints:**
            - **ROPE Scaling**: 8,192 positional tokens  
            - **Embeddings**: Text (2,048), Speech (4,096)  
            - **Vocab**: 704 compact tokens  
            - **Recommended Input**: 400–950 characters  
            
            _Note: Input limits may expand in future updates._
            """)

    def check_limit(txt):
        char_count = len(txt)
        if char_count > 1000:
            return (
                gr.update(visible=True, value=f"⚠️ Text exceeds 1000 characters ({char_count}/1000). Please shorten it."),
                gr.update(interactive=False)
            )
        return gr.update(visible=False), gr.update(interactive=True)

    text.change(fn=check_limit, inputs=text, outputs=[error_display, run_btn])

    run_btn.click(
        fn=generate_audio,
        inputs=[text, ref_avatar, exaggeration, cfg_weight, temp, seed_num],
        outputs=[audio_output],
    )

demo.queue(max_size=5)
demo.launch(share=True, inbrowser=True)