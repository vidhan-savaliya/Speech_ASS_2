"""
part1_denoise.py
────────────────
Task 1.3 — Denoising & Normalization

Primary:  DeepFilterNet  (neural noise reduction + dereverberation)
Fallback: Spectral Subtraction (no extra C++ dependency)

Outputs:
  • data/denoised_segment.wav   — cleaned audio at 16 kHz
  • plots/spectrogram_compare.png — before/after spectrogram
"""

import os
import numpy as np
import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Subtraction fallback
# ─────────────────────────────────────────────────────────────────────────────

def spectral_subtraction(waveform: np.ndarray, sr: int,
                         alpha: float = 2.0,
                         noise_frames: int = 20) -> np.ndarray:
    """
    Classic power-spectrum subtraction:
        |S̃(f)|² = max(|X(f)|² - α·|N̂(f)|², 0)

    α (over-subtraction factor) controls aggressiveness.
    The noise PSD is estimated from the first `noise_frames` frames.
    """
    n_fft    = 512
    hop      = n_fft // 4
    win_len  = n_fft

    # STFT
    stft = np.array([
        np.fft.rfft(waveform[i:i+win_len] * np.hanning(win_len))
        for i in range(0, len(waveform) - win_len + 1, hop)
    ])  # (T, F)

    # Estimate noise from the first ~noise_frames frames (assumed silent)
    noise_mag = np.mean(np.abs(stft[:noise_frames]) ** 2, axis=0)

    # Subtract
    clean_pow = np.maximum(np.abs(stft) ** 2 - alpha * noise_mag, 0)
    clean_mag = np.sqrt(clean_pow)
    phase     = np.angle(stft)
    clean_stft = clean_mag * np.exp(1j * phase)

    # ISTFT (overlap-add)
    out_len = (len(clean_stft) - 1) * hop + win_len
    output  = np.zeros(out_len)
    window  = np.hanning(win_len)
    norm    = np.zeros(out_len)
    for i, frame in enumerate(clean_stft):
        seg   = np.fft.irfft(frame, n=win_len)
        start = i * hop
        output[start:start+win_len] += seg * window
        norm  [start:start+win_len] += window ** 2

    norm = np.maximum(norm, 1e-8)
    output /= norm
    return output.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Loudness normalisation (peak normalise to –1 dBFS)
# ─────────────────────────────────────────────────────────────────────────────

def peak_normalize(waveform: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(waveform))
    if peak < 1e-8:
        return waveform
    target_linear = 10 ** (target_dbfs / 20.0)
    return waveform * (target_linear / peak)


# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def save_spectrogram_plot(before: np.ndarray, after: np.ndarray,
                          sr: int, output_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    for ax, sig, title in zip(axes,
                               [before, after],
                               ["Original (noisy)", "Denoised"]):
        n_fft = 1024
        hop   = 256
        stft  = np.abs(np.array([
            np.fft.rfft(sig[i:i+n_fft] * np.hanning(n_fft))
            for i in range(0, len(sig) - n_fft + 1, hop)
        ])).T
        log_mag = 20 * np.log10(np.maximum(stft, 1e-8))
        im = ax.imshow(log_mag, aspect="auto", origin="lower",
                       extent=[0, len(sig)/sr, 0, sr//2],
                       vmin=-80, vmax=0, cmap="magma")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        plt.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[denoise] Spectrogram saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main denoising entry point
# ─────────────────────────────────────────────────────────────────────────────

def denoise_audio(input_path: str, output_path: str) -> None:
    print("[denoise] Starting denoising pipeline…")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Source audio '{input_path}' not found.  "
            "Run `python download_segment.py` first, or place your WAV there."
        )

    # Load raw audio
    waveform, sr = torchaudio.load(input_path)
    waveform_np  = waveform.mean(0).numpy()   # collapse to mono if stereo

    # Resample to 16 kHz if needed
    if sr != 16000:
        resamp = torchaudio.transforms.Resample(sr, 16000)
        waveform = resamp(waveform.mean(0, keepdim=True))
        sr = 16000
        waveform_np = waveform.squeeze().numpy()

    original_np = waveform_np.copy()

    # ── Try DeepFilterNet first ───────────────────────────────────────────────
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        print("[denoise] Using DeepFilterNet…")
        model, df_state, _ = init_df()
        wav_df, _ = load_audio(input_path, sr=df_state.sr())
        enhanced  = enhance(model, df_state, wav_df)
        # Convert to numpy for normalisation step
        enhanced_np = enhanced.squeeze().numpy() if isinstance(enhanced, torch.Tensor) \
                      else np.array(enhanced).squeeze()
        method = "DeepFilterNet"
    except Exception as e:
        print(f"[denoise] DeepFilterNet unavailable ({e}); falling back to Spectral Subtraction…")
        enhanced_np = spectral_subtraction(waveform_np, sr, alpha=2.0)
        method = "SpectralSubtraction"

    # ── Loudness normalisation ────────────────────────────────────────────────
    cleaned = peak_normalize(enhanced_np)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_tensor = torch.from_numpy(cleaned).unsqueeze(0)
    torchaudio.save(output_path, out_tensor, 16000)
    print(f"[denoise] Method: {method}  |  Saved → {output_path}")

    # ── Spectrogram comparison ────────────────────────────────────────────────
    Config.setup()
    plot_path = os.path.join(Config.PLOTS_DIR, "spectrogram_compare.png")
    try:
        save_spectrogram_plot(original_np, cleaned, 16000, plot_path)
    except Exception as pe:
        print(f"[denoise] Spectrogram plot skipped: {pe}")


if __name__ == "__main__":
    Config.setup()
    denoise_audio(Config.INPUT_AUDIO, Config.DENOISED_AUDIO)
