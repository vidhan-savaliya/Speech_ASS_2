import os
import torch

class Config:
    # ── System ────────────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Directory structure ───────────────────────────────────────────────────
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR  = os.path.join(BASE_DIR, "data")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")

    # ── Input / Output audio files ────────────────────────────────────────────
    INPUT_AUDIO    = os.path.join(DATA_DIR, "original_segment.wav")
    MY_VOICE_SRC   = os.path.join(BASE_DIR, "Myvoice.m4a")          # My actual recorded voice
    REF_VOICE      = os.path.join(DATA_DIR, "student_voice_ref.wav") # Converted 16kHz version for model
    DENOISED_AUDIO = os.path.join(DATA_DIR, "denoised_segment.wav")
    TMP_TTS        = os.path.join(DATA_DIR, "tmp_tts.wav")
    FINAL_AUDIO    = os.path.join(DATA_DIR, "output_LRL_cloned.wav")

    # ── Text files ────────────────────────────────────────────────────────────
    SYLLABUS_TXT        = os.path.join(DATA_DIR, "syllabus.txt")
    TECHNICAL_CORPUS    = os.path.join(DATA_DIR, "technical_corpus.json")
    OUTPUT_LID          = os.path.join(DATA_DIR, "lid_timestamps.csv")
    OUTPUT_TRANSCRIPT   = os.path.join(DATA_DIR, "transcript.txt")
    OUTPUT_IPA          = os.path.join(DATA_DIR, "ipa_transcript.txt")
    OUTPUT_TRANSLATION  = os.path.join(DATA_DIR, "translated.txt")
    METRICS_REPORT      = os.path.join(DATA_DIR, "metrics_report.json")

    # ── Model weights ─────────────────────────────────────────────────────────
    LID_WEIGHTS = os.path.join(BASE_DIR, "custom_lid_weights.pt")
    CM_WEIGHTS  = os.path.join(BASE_DIR, "cm_weights.pt")

    # ── STT / Whisper ─────────────────────────────────────────────────────────
    # whisper-medium is a good balance; change to whisper-large-v3 if GPU ≥ 10 GB
    WHISPER_MODEL   = "openai/whisper-large-v3-turbo"
    LOGIT_BOOST     = 0.2           # logit boost for syllabus terms
    WHISPER_BEAMS   = 5

    # ── LID ───────────────────────────────────────────────────────────────────
    LID_BASE_MODEL  = "facebook/wav2vec2-base"
    LID_FRAME_MS    = 200           # 200 ms per LID frame
    LID_HIDDEN_DIM  = 256
    LID_EPOCHS      = 3
    LID_LR          = 1e-4
    LID_BATCH_SIZE  = 8

    # ── Translation ───────────────────────────────────────────────────────────
    NLLB_MODEL  = "facebook/nllb-200-distilled-600M"
    LRL_LANG    = "mar_Deva"        # Marathi — target Low Resource Language
    SRC_LANG    = "hin_Deva"        # Hindi (source side of Hinglish)

    # ── TTS ───────────────────────────────────────────────────────────────────
    TTS_MODEL   = "facebook/mms-tts-mar"   # Marathi MMS VITS
    TARGET_SR   = 22050                    # Output sample rate (>= 22.05 kHz required)

    # ── Prosody / DTW ─────────────────────────────────────────────────────────
    MFCC_BINS   = 13
    PITCH_FLOOR = 75    # Hz
    PITCH_CEIL  = 600   # Hz

    # ── Anti-Spoofing ─────────────────────────────────────────────────────────
    N_LFCC      = 20
    CM_LR       = 1e-3
    CM_EPOCHS   = 10

    # ── FGSM ─────────────────────────────────────────────────────────────────
    FGSM_SEG_S  = 5        # segment length for adversarial attack (seconds)
    FGSM_SNR_MIN = 40.0    # minimum SNR (dB) to remain "inaudible"

    @classmethod
    def setup(cls):
        os.makedirs(cls.DATA_DIR,  exist_ok=True)
        os.makedirs(cls.PLOTS_DIR, exist_ok=True)

        # Write extended syllabus if absent
        if not os.path.exists(cls.SYLLABUS_TXT):
            terms = [
                "stochastic", "cepstrum", "mel-frequency cepstral coefficients", "MFCC",
                "spectrogram", "short-time Fourier transform", "STFT", "filterbank",
                "hidden Markov model", "HMM", "Viterbi", "Baum-Welch",
                "connectionist temporal classification", "CTC", "beam search",
                "attention mechanism", "transformer", "self-attention", "encoder", "decoder",
                "phoneme", "grapheme", "allophones", "prosody", "intonation",
                "fundamental frequency", "pitch", "formant", "resonance", "bandwidth",
                "acoustic model", "language model", "n-gram", "perplexity",
                "word error rate", "WER", "phoneme error rate",
                "convolution", "pooling", "recurrent neural network", "LSTM", "GRU",
                "wav2vec", "speaker embedding", "x-vector", "d-vector",
                "voice activity detection", "VAD", "endpoint detection",
                "code switching", "language identification", "multilingual",
                "distillation", "fine-tuning", "transfer learning",
                "signal-to-noise ratio", "SNR", "dynamic range compression",
                "linear prediction", "LPC", "cepstral mean normalization",
                "mel filterbank", "log-mel", "pitch contour", "energy contour",
                "duration modeling", "text normalization", "grapheme-to-phoneme", "G2P",
                "forced alignment", "Montreal Forced Aligner", "MFA",
                "dynamic time warping", "DTW", "PSOLA", "pitch synchronous",
                "vocoder", "neural vocoder", "HiFi-GAN", "WaveNet", "WaveGlow",
                "mel-cepstral distortion", "MCD", "naturalness", "intelligibility"
            ]
            with open(cls.SYLLABUS_TXT, "w", encoding="utf-8") as f:
                f.write("\n".join(terms))

        print(f"[Config] Device: {cls.DEVICE}")
        print(f"[Config] Data dir: {cls.DATA_DIR}")
