import librosa
import librosa.display
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

# 1. Load audio
y, sr = librosa.load("separated/mdx_extra_q/input/other.wav", sr=None)

# 2. Apply HPSS (Harmonic–Percussive Separation)
harmonic, percussive = librosa.effects.hpss(y)

# Save stems
sf.write("harmonic.wav", harmonic, sr)
sf.write("percussive.wav", percussive, sr)

# 3. EQ Filtering Helper
def butter_bandstop(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

def apply_eq(data, lowcut, highcut, fs):
    b, a = butter_bandstop(lowcut, highcut, fs)
    return lfilter(b, a, data)

# Example: Reduce tabla bleed in harmonic stem
# Tabla often dominates 80–250 Hz and sharp transients ~2–4 kHz
harmonic_eq = apply_eq(harmonic, 10, 4000, sr)   # notch out bass bleed
#harmonic_eq = apply_eq(harmonic_eq, 2000, 4000, sr)  # notch out transient bleed

sf.write("harmonic_clean.wav", harmonic_eq, sr)

import matplotlib.pyplot as plt

# --- 4. Visualization ---
harmonic_eq = np.nan_to_num(harmonic_eq, nan=0.0, posinf=0.0, neginf=0.0)

# Ensure it's a 1D array if plotting waveform
harmonic_eq = np.asarray(harmonic_eq).flatten()

# Plot amplitude waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(harmonic_eq, sr=sr, alpha=0.8)
plt.title("Cleaned Harmonic Stem - Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Plot frequency spectrum (FFT)
fft_spectrum = np.fft.rfft(harmonic_eq)
freqs = np.fft.rfftfreq(len(harmonic_eq), d=1/sr)

plt.figure(figsize=(12, 4))
plt.plot(freqs, np.abs(fft_spectrum), color="purple")
plt.title("Frequency Spectrum of Cleaned Harmonic Stem")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 5000)  # focus on 0–5 kHz range
plt.tight_layout()
plt.show()

# Plot spectrogram
S = np.abs(librosa.stft(harmonic_eq))
S_db = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(12, 6))
librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of Cleaned Harmonic Stem")
plt.tight_layout()
plt.show()