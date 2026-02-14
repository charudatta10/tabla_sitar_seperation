Perfect — now this is clean.

You’re using:

* `pyproject.toml` (PEP 621)
* `uv.lock`
* Streamlit app: `sitar-tabla.py`
* Entry stub: `main.py`

Below is a **professional README.md** for your project.

You can copy-paste this directly.

---

# 📄 `README.md`

````markdown
# 🎵 Sitar–Tabla Separation Tool

Analytical + AI-based source separation for Hindustani classical music.

This Streamlit application separates a mixed Sitar + Tabla recording into:

- 🎻 Harmonic component (Sitar-like)
- 🥁 Percussive component (Tabla-like)

Two methods are provided:

1. **HPSS (Analytical, Fast, Explainable)**
2. **Demucs (AI-based, High Quality)**

Designed for educators, researchers, archivists, and music technologists.

---

## ✨ Features

- Single-file Streamlit interface
- HPSS (Harmonic–Percussive Source Separation)
- Optional Demucs AI model
- Downloadable WAV stems
- Academic-friendly method notes

---

# 🚀 Installation

This project uses **pyproject.toml + uv** (recommended).

---

## 🔹 Option 1 — Automatic Install (Windows PowerShell)

Run:

```powershell
irm https://raw.githubusercontent.com/charudatta10/tabla_sitar_seperation/main/install.ps1 | iex
````

This will:

* Install Python (if missing)
* Install uv
* Install dependencies
* Prepare the project

---

## 🔹 Option 2 — Manual Install (Recommended for Developers)

### 1️⃣ Install Python 3.10+

Download from:
[https://www.python.org/downloads/](https://www.python.org/downloads/)

---

### 2️⃣ Install uv

```bash
pip install uv
```

---

### 3️⃣ Clone the repository

```bash
git clone https://github.com/charudatta10/tabla_sitar_seperation.git
cd tabla_sitar_seperation
```

---

### 4️⃣ Install dependencies

```bash
uv sync
```

This creates a virtual environment and installs exact locked versions.

---

# ▶️ Running the App

From project root:

```bash
uv run streamlit run sitar-tabla.py
```

Your browser will open automatically.

---

# 🧠 Methods

## 🎻 HPSS (Analytical)

Uses:

* Short-Time Fourier Transform
* Median filtering
* Time–frequency continuity differences

Best for:

* Explainable research
* Classroom demonstrations
* Low-resource machines

---

## 🧠 Demucs (AI-Based)

Deep neural network trained on large music corpora.

* High perceptual quality
* Requires more compute
* Uses PyTorch backend

Install separately if needed:

```bash
pip install demucs
```

---

# 📦 Dependencies

Defined in `pyproject.toml`:

* demucs
* librosa
* soundfile
* streamlit
* torchcodec

---

# 📚 Academic Usage

If citing HPSS:

> Fitzgerald, D. (2010). Harmonic/Percussive Separation using Median Filtering.

If citing Demucs:

> Défossez et al. (2021). Hybrid Spectrogram and Waveform Source Separation.

---

# 🛠 Project Structure

```
tabla_sitar_seperation/
│
├── sitar-tabla.py      # Streamlit app
├── main.py             # Entry stub
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 🧪 Tested With

* Python 3.10+
* Windows 10/11
* uv package manager

---

# ⚠ Notes

* Demucs requires additional compute resources.
* For GPU acceleration, install appropriate PyTorch CUDA build.
* HPSS works entirely on CPU.

---

# 👤 Author

Charudatta Korde
Music Technology • Research • Computational Audio

---

# 📜 License

MIT License 





