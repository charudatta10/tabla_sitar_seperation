import io
import subprocess
import tempfile

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

matplotlib.use("Agg")
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ReportLab imports for PDF export
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import Image as RLImage
from scipy.signal import butter, lfilter

st.set_page_config(page_title="Sitar-Tabla Separator", layout="centered")

st.title("🎵 Sitar-Tabla Separation Tool")
st.caption(
    "HPSS (Analytical) + EQ Filtering + Demucs (AI, optional) — single-file Streamlit app"
)

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
    b, a = butter(order, [low, high], btype="bandstop")
    return b, a


def apply_eq(data, lowcut, highcut, fs):
    b, a = butter_bandstop(lowcut, highcut, fs)
    return lfilter(b, a, data)


def apply_hpss_with_eq(audio_path, notch_low, notch_high):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_eq = apply_eq(harmonic, notch_low, notch_high, sr)
    harmonic_eq = np.nan_to_num(harmonic_eq, nan=0.0, posinf=0.0, neginf=0.0)
    harmonic_eq = np.asarray(harmonic_eq).flatten()
    return harmonic, percussive, harmonic_eq, sr


# ---- Plot generators ----


def plot_waveform(audio, sr, title="Waveform"):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr, alpha=0.8, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_fft(audio, sr, xlim=5000, title="Frequency Spectrum"):
    fft_spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), d=1 / sr)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs, np.abs(fft_spectrum), color="purple")
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, xlim)
    fig.tight_layout()
    return fig


def plot_spectrogram(audio, sr, title="Spectrogram"):
    S = np.abs(librosa.stft(audio))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_db, sr=sr, x_axis="time", y_axis="hz", cmap="magma", ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_to_bytes(fig):
    """Convert a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def show_signal_visualizations(audio, sr, label="Signal"):
    """Display waveform, FFT, and spectrogram inside a Streamlit expander."""
    with st.expander(f"📊 Visualizations — {label}", expanded=False):
        st.markdown("**Waveform**")
        fig_wave = plot_waveform(audio, sr, title=f"Waveform — {label}")
        st.pyplot(fig_wave)
        plt.close(fig_wave)

        st.markdown("**Frequency Spectrum (FFT, 0–5 kHz)**")
        fig_fft = plot_fft(audio, sr, xlim=5000, title=f"Frequency Spectrum — {label}")
        st.pyplot(fig_fft)
        plt.close(fig_fft)

        st.markdown("**Spectrogram**")
        fig_spec = plot_spectrogram(audio, sr, title=f"Spectrogram — {label}")
        st.pyplot(fig_spec)
        plt.close(fig_spec)


def run_demucs(audio_path, out_dir):
    cmd = ["demucs", "-n", "htdemucs_ft", "-o", out_dir, audio_path]
    subprocess.run(cmd, check=True)


# ---- PDF Export ----


def signal_stats(audio, sr):
    duration = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    return duration, rms, peak


def build_pdf(filename, method_label, signals, method_note, notch_params=None):
    """
    Build a PDF analysis report.

    signals: list of dicts with keys:
        - label (str)
        - audio (np.ndarray)
        - sr (int)
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Sitar-Tabla Separation Report",
        author="Sitar-Tabla Separator Tool",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=16,
    )
    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=18,
        spaceAfter=6,
        textColor=colors.HexColor("#2d3561"),
    )
    signal_style = ParagraphStyle(
        "SignalHead",
        parent=styles["Heading3"],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=4,
        textColor=colors.HexColor("#c84b31"),
    )
    note_style = ParagraphStyle(
        "MethodNote",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#444444"),
        backColor=colors.HexColor("#f5f5f5"),
        borderPad=6,
        spaceAfter=10,
    )
    footer_style = ParagraphStyle(
        "footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER,
    )

    story = []

    # ── Title block ──
    story.append(Paragraph("Sitar-Tabla Separation Report", title_style))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
            f"File: <b>{filename}</b> &nbsp;|&nbsp; Method: <b>{method_label}</b>",
            subtitle_style,
        )
    )
    story.append(
        HRFlowable(
            width="100%", thickness=1.5, color=colors.HexColor("#2d3561"), spaceAfter=10
        )
    )

    # ── Method note ──
    story.append(Paragraph("Method Summary", section_style))
    story.append(Paragraph(method_note.replace("\n", "<br/>"), note_style))

    # ── EQ params table if applicable ──
    if notch_params:
        story.append(Paragraph("EQ Filter Parameters", section_style))
        tdata = [
            ["Parameter", "Value"],
            ["Filter Type", "Butterworth Bandstop (order 4)"],
            ["Notch Low Cutoff", f"{notch_params['low']} Hz"],
            ["Notch High Cutoff", f"{notch_params['high']} Hz"],
        ]
        t = Table(tdata, colWidths=[7 * cm, 9 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3561")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#f0f4ff"), colors.white],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 10))

    # ── Per-signal sections ──
    story.append(Paragraph("Signal Analysis", section_style))

    for sig in signals:
        label = sig["label"]
        audio = sig["audio"]
        sr = sig["sr"]

        story.append(Paragraph(label, signal_style))

        # Stats table
        duration, rms, peak = signal_stats(audio, sr)
        tdata = [
            ["Metric", "Value"],
            ["Duration", f"{duration:.2f} s"],
            ["Sample Rate", f"{sr} Hz"],
            ["RMS Amplitude", f"{rms:.4f}"],
            ["Peak Amplitude", f"{peak:.4f}"],
        ]
        t = Table(tdata, colWidths=[6 * cm, 10 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c84b31")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#fff5f0"), colors.white],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 8))

        # Waveform
        fig_w = plot_waveform(audio, sr, title=f"Waveform — {label}")
        story.append(
            RLImage(io.BytesIO(fig_to_bytes(fig_w)), width=16 * cm, height=4.8 * cm)
        )
        plt.close(fig_w)
        story.append(Spacer(1, 4))

        # FFT
        fig_f = plot_fft(audio, sr, xlim=5000, title=f"Frequency Spectrum — {label}")
        story.append(
            RLImage(io.BytesIO(fig_to_bytes(fig_f)), width=16 * cm, height=4.8 * cm)
        )
        plt.close(fig_f)
        story.append(Spacer(1, 4))

        # Spectrogram
        fig_s = plot_spectrogram(audio, sr, title=f"Spectrogram — {label}")
        story.append(
            RLImage(io.BytesIO(fig_to_bytes(fig_s)), width=16 * cm, height=5.5 * cm)
        )
        plt.close(fig_s)

        story.append(Spacer(1, 6))
        story.append(
            HRFlowable(
                width="100%",
                thickness=0.5,
                color=colors.HexColor("#dddddd"),
                spaceAfter=4,
            )
        )

    # ── Footer ──
    story.append(Spacer(1, 10))
    story.append(
        HRFlowable(
            width="100%", thickness=1, color=colors.HexColor("#2d3561"), spaceAfter=6
        )
    )
    story.append(
        Paragraph(
            "Generated by Sitar-Tabla Separator Tool • "
            "HPSS = explainable • HPSS+EQ = reduced bleed • Demucs = perceptual quality",
            footer_style,
        )
    )

    doc.build(story)
    buf.seek(0)
    return buf.read()


# -----------------------------
# UI
# -----------------------------

uploaded = st.file_uploader(
    "Upload MP3 or WAV (Sitar + Tabla mix)", type=["mp3", "wav"]
)

method = st.radio(
    "Separation method",
    [
        "HPSS (Analytical, Fast)",
        "HPSS + EQ Filtering (Enhanced)",
        "Demucs (AI, High Quality)",
    ],
)

if method == "HPSS + EQ Filtering (Enhanced)":
    st.markdown("#### EQ Bandstop Filter Settings")
    st.caption(
        "A bandstop (notch) filter is applied to the harmonic stem to reduce tabla bleed. "
        "Tabla energy tends to concentrate in the bass (~80–250 Hz) and around transient peaks (~2–4 kHz)."
    )
    col1, col2 = st.columns(2)
    with col1:
        notch_low = st.number_input(
            "Notch Low Cutoff (Hz)", min_value=10, max_value=20000, value=10, step=10
        )
    with col2:
        notch_high = st.number_input(
            "Notch High Cutoff (Hz)",
            min_value=10,
            max_value=20000,
            value=4000,
            step=100,
        )

if uploaded:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / uploaded.name
        with open(input_path, "wb") as f:
            f.write(uploaded.read())

        st.audio(str(input_path))

        # Original signal visualizations (always shown)
        y_orig, sr_orig = librosa.load(str(input_path), sr=None, mono=True)
        show_signal_visualizations(y_orig, sr_orig, label="Original Mix")

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
                    file_name="sitar_like.wav",
                )
                show_signal_visualizations(harmonic, sr, label="Sitar — Harmonic")

                st.subheader("🥁 Tabla (Percussive)")
                st.audio(str(tabla_path))
                st.download_button(
                    "Download tabla_like.wav",
                    data=open(tabla_path, "rb"),
                    file_name="tabla_like.wav",
                )
                show_signal_visualizations(percussive, sr, label="Tabla — Percussive")

                method_note = (
                    "Harmonic-Percussive Source Separation (HPSS) exploits time-frequency continuity "
                    "differences between string instruments (sitar) and percussive strokes (tabla). "
                    "The harmonic mask retains sustained sinusoidal components while the percussive "
                    "mask captures transient energy."
                )
                st.markdown(f"**Method note (for papers):** {method_note}")

                signals_for_pdf = [
                    {"label": "Original Mix", "audio": y_orig, "sr": sr_orig},
                    {"label": "Sitar — Harmonic", "audio": harmonic, "sr": sr},
                    {"label": "Tabla — Percussive", "audio": percussive, "sr": sr},
                ]

                st.subheader("📄 Export Report")
                with st.spinner("Building PDF report…"):
                    pdf_bytes = build_pdf(
                        filename=uploaded.name,
                        method_label="HPSS (Analytical, Fast)",
                        signals=signals_for_pdf,
                        method_note=method_note,
                    )
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"separation_report_{Path(uploaded.name).stem}.pdf",
                    mime="application/pdf",
                )

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
                    file_name="sitar_harmonic.wav",
                )
                show_signal_visualizations(
                    harmonic, sr, label="Sitar — Harmonic (raw HPSS)"
                )

                st.subheader("🎻✨ Sitar — Harmonic Stem (EQ cleaned)")
                st.audio(str(sitar_eq_path))
                st.download_button(
                    "Download sitar_harmonic_clean.wav",
                    data=open(sitar_eq_path, "rb"),
                    file_name="sitar_harmonic_clean.wav",
                )
                show_signal_visualizations(
                    harmonic_eq, sr, label="Sitar — Harmonic (EQ cleaned)"
                )

                st.subheader("🥁 Tabla (Percussive)")
                st.audio(str(tabla_path))
                st.download_button(
                    "Download tabla_percussive.wav",
                    data=open(tabla_path, "rb"),
                    file_name="tabla_percussive.wav",
                )
                show_signal_visualizations(percussive, sr, label="Tabla — Percussive")

                method_note = (
                    f"A bandstop (notch) Butterworth filter ({notch_low}–{notch_high} Hz, order 4) "
                    "is applied post-HPSS to attenuate tabla bleed remaining in the harmonic stem. "
                    "Tabla energy concentrates in the low-frequency range (~80–250 Hz) and around "
                    "sharp transients (~2–4 kHz); the notch targets this overlap region."
                )
                st.markdown(f"**Method note (for papers):** {method_note}")

                signals_for_pdf = [
                    {"label": "Original Mix", "audio": y_orig, "sr": sr_orig},
                    {
                        "label": "Sitar — Harmonic (raw HPSS)",
                        "audio": harmonic,
                        "sr": sr,
                    },
                    {
                        "label": "Sitar — Harmonic (EQ cleaned)",
                        "audio": harmonic_eq,
                        "sr": sr,
                    },
                    {"label": "Tabla — Percussive", "audio": percussive, "sr": sr},
                ]

                st.subheader("📄 Export Report")
                with st.spinner("Building PDF report…"):
                    pdf_bytes = build_pdf(
                        filename=uploaded.name,
                        method_label="HPSS + EQ Filtering (Enhanced)",
                        signals=signals_for_pdf,
                        method_note=method_note,
                        notch_params={"low": notch_low, "high": notch_high},
                    )
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"separation_report_{Path(uploaded.name).stem}.pdf",
                    mime="application/pdf",
                )

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

                signals_for_pdf = [
                    {"label": "Original Mix", "audio": y_orig, "sr": sr_orig}
                ]

                if stem_dir:
                    st.success("Separation complete (Demucs)")
                    for stem in stem_dir.glob("*.wav"):
                        st.subheader(stem.stem.capitalize())
                        st.audio(str(stem))
                        st.download_button(
                            f"Download {stem.name}",
                            data=open(stem, "rb"),
                            file_name=stem.name,
                        )
                        stem_audio, stem_sr = librosa.load(
                            str(stem), sr=None, mono=True
                        )
                        show_signal_visualizations(
                            stem_audio, stem_sr, label=stem.stem.capitalize()
                        )
                        signals_for_pdf.append(
                            {
                                "label": stem.stem.capitalize(),
                                "audio": stem_audio,
                                "sr": stem_sr,
                            }
                        )

                method_note = (
                    "Demucs uses deep convolutional and transformer layers trained on large music corpora. "
                    "Tabla is mostly captured in 'drums.wav'; sitar energy appears mainly in 'other.wav'."
                )
                st.markdown(f"**Method note:** {method_note}")

                st.subheader("📄 Export Report")
                with st.spinner("Building PDF report…"):
                    pdf_bytes = build_pdf(
                        filename=uploaded.name,
                        method_label="Demucs (AI, High Quality)",
                        signals=signals_for_pdf,
                        method_note=method_note,
                    )
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"separation_report_{Path(uploaded.name).stem}.pdf",
                    mime="application/pdf",
                )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Designed for educators, researchers, and archivists • "
    "HPSS = explainable • HPSS+EQ = reduced bleed • Demucs = perceptual quality"
)
