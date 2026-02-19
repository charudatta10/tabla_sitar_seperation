import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import tempfile
import subprocess
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, lfilter

st.set_page_config(page_title="Sitar-Tabla Separator", layout="centered")

st.title("🎵 Sitar-Tabla Separation Tool")
st.caption("HPSS (Analytical) + EQ Filtering + Demucs (AI, optional) — single-file Streamlit app")

# -----------------------------
# Helpers
# -----------------------------

def run_hpss(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    harmonic, percussive = librosa.effects.hpss(y)
    return harmonic, percussive, sr


def butter_bandstop(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a


def apply_eq(data, lowcut, highcut, fs):
    b, a = butter_bandstop(lowcut, highcut, fs)
    return lfilter(b, a, data)


def apply_hpss_with_eq(audio_path, notch_low, notch_high):
    """Run HPSS then apply bandstop EQ to reduce tabla bleed in harmonic stem."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    harmonic, percussive = librosa.effects.hpss(y)

    # Apply EQ notch filter to harmonic stem
    harmonic_eq = apply_eq(harmonic, notch_low, notch_high, sr)
    harmonic_eq = np.nan_to_num(harmonic_eq, nan=0.0, posinf=0.0, neginf=0.0)
    harmonic_eq = np.asarray(harmonic_eq).flatten()

    return harmonic, percussive, harmonic_eq, sr


def plot_waveform(audio, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr, alpha=0.8, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_fft(audio, sr, xlim=5000):
    fft_spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), d=1 / sr)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs, np.abs(fft_spectrum), color="purple")
    ax.set_title("Frequency Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, xlim)
    fig.tight_layout()
    return fig


def plot_spectrogram(audio, sr):
    S = np.abs(librosa.stft(audio))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz", cmap="magma", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Spectrogram")
    fig.tight_layout()
    return fig


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
    ["HPSS (Analytical, Fast)", "HPSS + EQ Filtering (Enhanced)", "Demucs (AI, High Quality)"]
)

# EQ options (shown only for HPSS + EQ)
if method == "HPSS + EQ Filtering (Enhanced)":
    st.markdown("#### EQ Bandstop Filter Settings")
    st.caption(
        "A bandstop (notch) filter is applied to the harmonic stem to reduce tabla bleed. "
        "Tabla energy tends to concentrate in the bass (~80–250 Hz) and around transient peaks (~2–4 kHz)."
    )
    col1, col2 = st.columns(2)
    with col1:
        notch_low = st.number_input("Notch Low Cutoff (Hz)", min_value=10, max_value=20000, value=10, step=10)
    with col2:
        notch_high = st.number_input("Notch High Cutoff (Hz)", min_value=10, max_value=20000, value=4000, step=100)

    show_viz = st.checkbox("Show Visualizations (Waveform, FFT, Spectrogram)", value=True)

if uploaded:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / uploaded.name
        with open(input_path, "wb") as f:
            f.write(uploaded.read())

        st.audio(str(input_path))

        if st.button("🚀 Separate"):
            st.info("Processing… please wait")

            # ---- HPSS only ----
            if method == "HPSS (Analytical, Fast)":
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
                Harmonic-Percussive Source Separation exploits
                time-frequency continuity differences between
                string instruments (sitar) and percussive strokes (tabla).
                """)

            # ---- HPSS + EQ ----
            elif method == "HPSS + EQ Filtering (Enhanced)":
                harmonic, percussive, harmonic_eq, sr = apply_hpss_with_eq(
                    str(input_path), notch_low, notch_high
                )

                sitar_raw_path = Path(tmpdir) / "sitar_harmonic.wav"
                sitar_eq_path = Path(tmpdir) / "sitar_harmonic_clean.wav"
                tabla_path = Path(tmpdir) / "tabla_percussive.wav"

                sf.write(sitar_raw_path, harmonic, sr)
                sf.write(sitar_eq_path, harmonic_eq, sr)
                sf.write(tabla_path, percussive, sr)

                st.success("Separation complete (HPSS + EQ)")

                st.subheader("🎻 Sitar — Harmonic Stem (raw HPSS)")
                st.audio(str(sitar_raw_path))
                st.download_button(
                    "Download sitar_harmonic.wav",
                    data=open(sitar_raw_path, "rb"),
                    file_name="sitar_harmonic.wav"
                )

                st.subheader("🎻✨ Sitar — Harmonic Stem (EQ cleaned)")
                st.audio(str(sitar_eq_path))
                st.download_button(
                    "Download sitar_harmonic_clean.wav",
                    data=open(sitar_eq_path, "rb"),
                    file_name="sitar_harmonic_clean.wav"
                )

                st.subheader("🥁 Tabla (Percussive)")
                st.audio(str(tabla_path))
                st.download_button(
                    "Download tabla_percussive.wav",
                    data=open(tabla_path, "rb"),
                    file_name="tabla_percussive.wav"
                )

                # ---- Visualizations ----
                if show_viz:
                    st.subheader("📊 Visualizations — Cleaned Harmonic Stem")

                    st.markdown("**Waveform**")
                    fig_wave = plot_waveform(harmonic_eq, sr)
                    st.pyplot(fig_wave)
                    plt.close(fig_wave)

                    st.markdown("**Frequency Spectrum (FFT, 0–5 kHz)**")
                    fig_fft = plot_fft(harmonic_eq, sr, xlim=5000)
                    st.pyplot(fig_fft)
                    plt.close(fig_fft)

                    st.markdown("**Spectrogram**")
                    fig_spec = plot_spectrogram(harmonic_eq, sr)
                    st.pyplot(fig_spec)
                    plt.close(fig_spec)

                st.markdown(f"""
                **Method note (for papers):**  
                A bandstop (notch) Butterworth filter ({notch_low}–{notch_high} Hz, order 4)
                is applied post-HPSS to attenuate tabla bleed remaining in the harmonic stem.
                Tabla energy concentrates in the low-frequency range (~80–250 Hz) and around
                sharp transients (~2–4 kHz); the notch targets this overlap region.
                """)

            # ---- Demucs ----
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
    "HPSS = explainable • HPSS+EQ = reduced bleed • Demucs = perceptual quality"
)