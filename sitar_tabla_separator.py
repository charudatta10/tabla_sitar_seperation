import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import tempfile
import subprocess
import os
from pathlib import Path

st.set_page_config(page_title="Sitar–Tabla Separator", layout="centered")

st.title("🎵 Sitar–Tabla Separation Tool")
st.caption("HPSS (Analytical) + Demucs (AI, optional) — single-file Streamlit app")

# -----------------------------
# Helpers
# -----------------------------

def run_hpss(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    harmonic, percussive = librosa.effects.hpss(y)

    return harmonic, percussive, sr


def run_demucs(audio_path, out_dir):
    cmd = [
        "demucs",
        "-n", "htdemucs_ft",
        "-o", out_dir,
        audio_path
    ]
    subprocess.run(cmd, check=True)


# -----------------------------
# UI
# -----------------------------

uploaded = st.file_uploader(
    "Upload MP3 or WAV (Sitar + Tabla mix)",
    type=["mp3", "wav"]
)

method = st.radio(
    "Separation method",
    ["HPSS (Analytical, Fast)", "Demucs (AI, High Quality)"]
)

if uploaded:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / uploaded.name
        with open(input_path, "wb") as f:
            f.write(uploaded.read())

        st.audio(str(input_path))

        if st.button("🚀 Separate"):
            st.info("Processing… please wait")

            if method.startswith("HPSS"):
                harmonic, percussive, sr = run_hpss(str(input_path))

                sitar_path = Path(tmpdir) / "sitar_like.wav"
                tabla_path = Path(tmpdir) / "tabla_like.wav"

                sf.write(sitar_path, harmonic, sr)
                sf.write(tabla_path, percussive, sr)

                st.success("Separation complete (HPSS)")

                st.subheader("🎻 Sitar (Harmonic)")
                st.audio(str(sitar_path))
                st.download_button(
                    "Download sitar_like.wav",
                    data=open(sitar_path, "rb"),
                    file_name="sitar_like.wav"
                )

                st.subheader("🥁 Tabla (Percussive)")
                st.audio(str(tabla_path))
                st.download_button(
                    "Download tabla_like.wav",
                    data=open(tabla_path, "rb"),
                    file_name="tabla_like.wav"
                )

                st.markdown("""
                **Method note (for papers):**  
                Harmonic–Percussive Source Separation exploits
                time–frequency continuity differences between
                string instruments (sitar) and percussive strokes (tabla).
                """)

            else:
                demucs_out = Path(tmpdir) / "demucs_out"
                demucs_out.mkdir(exist_ok=True)

                try:
                    run_demucs(str(input_path), str(demucs_out))
                except Exception as e:
                    st.error("Demucs not found. Install with: `pip install demucs`")
                    st.stop()

                stem_dir = next(demucs_out.glob("**/*"), None)

                if stem_dir:
                    st.success("Separation complete (Demucs)")

                    for stem in stem_dir.glob("*.wav"):
                        st.subheader(stem.stem)
                        st.audio(str(stem))
                        st.download_button(
                            f"Download {stem.name}",
                            data=open(stem, "rb"),
                            file_name=stem.name
                        )

                st.markdown("""
                **Method note:**  
                Demucs uses deep convolutional and transformer layers
                trained on large music corpora.
                Tabla is mostly captured in `drums.wav`;
                sitar energy appears mainly in `other.wav`.
                """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Designed for educators, researchers, and archivists • "
    "HPSS = explainable • Demucs = perceptual quality"
)
