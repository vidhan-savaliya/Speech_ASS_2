"""
pipeline.py
───────────
End-to-end orchestration for Speech Understanding PA2.

Stages (each can be skipped with flags):
  1. Denoise + Normalize
  2. Frame-level LID
  3. Constrained STT (Whisper + N-gram logit bias)
  4. Hinglish → IPA → Marathi translation
  5. MMS TTS + DTW Prosody Warping
  6. CM training + EER evaluation
  7. FGSM Adversarial Attack

Usage:
    python pipeline.py                         # full run
    python pipeline.py --skip-train-lid        # skip LID fine-tuning
    python pipeline.py --skip-denoise          # assume denoised audio exists
    python pipeline.py --only-evaluate         # run evaluation only
    python pipeline.py --help
"""

import os
import sys
import json
import argparse
import time

from config import Config


def banner(msg: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def check_inputs() -> bool:
    ok = True
    if not os.path.exists(Config.INPUT_AUDIO):
        print(f"[!] Missing: {Config.INPUT_AUDIO}")
        print(f"    → Run: python download_segment.py")
        ok = False
    
    # Ensure our base voice is available or prepared
    if not os.path.exists(Config.REF_VOICE):
        print(f"[!] Missing: {Config.REF_VOICE}")
        if os.path.exists(Config.MY_VOICE_SRC):
            print(f"    → Found {Config.MY_VOICE_SRC}, run 'python prepare_student_voice.py' first.")
        else:
            print(f"    → Missing both {Config.REF_VOICE} and {Config.MY_VOICE_SRC}.")
            print(f"      Please place your voice recording as Myvoice.m4a in the root folder.")
        ok = False
    return ok


def summarize(results: dict) -> None:
    banner("Pipeline Summary")
    rows = [
        ("Metric",                  "Value",  "Target"),
        ("-" * 25,                  "-" * 15, "-" * 15),
        ("LID Macro F1",            f"{results.get('lid_f1', 'N/A')}",
         "≥ 0.85"),
        ("WER (overall)",           f"{results.get('wer', 'N/A')}",
         "< 15% EN / < 25% HI"),
        ("MCD (dB)",                f"{results.get('mcd', 'N/A'):.2f}" 
                                     if isinstance(results.get("mcd"), float)
                                     else "N/A",
         "< 8.0 dB"),
        ("Anti-Spoof EER",          f"{results.get('eer', 'N/A'):.3f}"
                                     if isinstance(results.get("eer"), float)
                                     else "N/A",
         "< 10%"),
        ("FGSM Best ε",             f"{results.get('best_epsilon', 'N/A'):.2e}"
                                     if isinstance(results.get("best_epsilon"), float)
                                     else "N/A",
         "SNR > 40 dB"),
    ]
    for r in rows:
        print(f"  {r[0]:<28}  {r[1]:<18}  {r[2]}")

    # Save JSON
    with open(Config.METRICS_REPORT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Metrics saved → {Config.METRICS_REPORT}")


def main(args) -> None:
    banner("Speech Understanding — Programming Assignment 2\n  Made by: Vidhatha (Student Pipeline)")
    Config.setup()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not args.only_evaluate and not check_inputs():
        sys.exit(1)

    results = {}
    t0 = time.time()

    # ── Stage 1: Denoising ────────────────────────────────────────────────────
    if not args.skip_denoise:
        banner("Stage 1 — Denoising & Normalization")
        from part1_denoise import denoise_audio
        denoise_audio(Config.INPUT_AUDIO, Config.DENOISED_AUDIO)
    else:
        print("[skip] Denoising (using existing denoised audio)")
        if not os.path.exists(Config.DENOISED_AUDIO):
            print(f"[!] Denoised audio not found at {Config.DENOISED_AUDIO}; "
                  "cannot skip.")
            sys.exit(1)

    # ── Stage 2: LID fine-tuning ──────────────────────────────────────────────
    if not args.skip_train_lid and not args.only_evaluate:
        banner("Stage 2a — LID Training")
        from part1_lid import train_lid
        train_lid()

    banner("Stage 2b — LID Inference")
    from part1_lid import infer_lid, evaluate_lid
    infer_lid(Config.DENOISED_AUDIO, Config.OUTPUT_LID)

    if os.path.exists(Config.LID_WEIGHTS):
        f1 = evaluate_lid(Config.LID_WEIGHTS)
        results["lid_f1"] = round(f1, 4)

    # ── Stage 3: STT ──────────────────────────────────────────────────────────
    if not args.skip_stt:
        banner("Stage 3 — Constrained Whisper Transcription")
        from part1_stt import transcribe_with_bias, report_wer
        transcribe_with_bias(
            Config.DENOISED_AUDIO,
            Config.OUTPUT_TRANSCRIPT,
            Config.SYLLABUS_TXT,
        )
        wer_info = report_wer(Config.OUTPUT_TRANSCRIPT)
        results.update(wer_info)
    else:
        print("[skip] STT transcription")

    # ── Stage 4: Translation ──────────────────────────────────────────────────
    if not args.skip_translate:
        banner("Stage 4 — IPA Mapping & LRL Translation")
        from part2_translation import run_translation_pipeline
        run_translation_pipeline(
            Config.OUTPUT_TRANSCRIPT,
            Config.OUTPUT_TRANSLATION,
        )
    else:
        print("[skip] Translation")

    # ── Stage 5: TTS + Prosody ────────────────────────────────────────────────
    if not args.skip_tts:
        banner("Stage 5 — TTS Synthesis & DTW Prosody Warping")
        from part3_tts_prosody import run_tts_pipeline
        tts_results = run_tts_pipeline(
            Config.DENOISED_AUDIO,
            Config.OUTPUT_TRANSLATION,
            Config.REF_VOICE,
            Config.FINAL_AUDIO,
        )
        results.update(tts_results)
    else:
        print("[skip] TTS synthesis")

    # ── Stage 6: Anti-Spoofing ────────────────────────────────────────────────
    if not args.skip_cm:
        banner("Stage 6a — CM Training")
        if not args.skip_train_cm:
            from part4_antispoof import train_cm
            train_cm()

        banner("Stage 6b — EER Evaluation")
        from part4_antispoof import evaluate_cm_eer
        eer = evaluate_cm_eer()
        results["eer"] = round(eer, 4)
    else:
        print("[skip] Anti-spoofing CM")

    # ── Stage 7: FGSM ─────────────────────────────────────────────────────────
    if not args.skip_fgsm:
        banner("Stage 7 — FGSM Adversarial Attack")
        from part4_antispoof import fgsm_adversarial_attack
        fgsm_res = fgsm_adversarial_attack(Config.DENOISED_AUDIO)
        results["fgsm_success"]    = fgsm_res.get("success")
        results["best_epsilon"]    = fgsm_res.get("best_epsilon")
        results["fgsm_snr"]        = fgsm_res.get("snr")
    else:
        print("[skip] FGSM adversarial attack")

    # ── Summary ───────────────────────────────────────────────────────────────
    results["elapsed_s"] = round(time.time() - t0, 1)
    summarize(results)

    banner(f"OK Pipeline complete in {results['elapsed_s']}s")
    print(f"  Final audio -> {Config.FINAL_AUDIO}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speech PA2 — end-to-end pipeline"
    )
    parser.add_argument("--skip-denoise",    action="store_true")
    parser.add_argument("--skip-train-lid",  action="store_true")
    parser.add_argument("--skip-stt",        action="store_true")
    parser.add_argument("--skip-translate",  action="store_true")
    parser.add_argument("--skip-tts",        action="store_true")
    parser.add_argument("--skip-cm",         action="store_true")
    parser.add_argument("--skip-train-cm",   action="store_true")
    parser.add_argument("--skip-fgsm",       action="store_true")
    parser.add_argument("--only-evaluate",   action="store_true",
                        help="Run CM EER + FGSM only (skip all training)")
    args = parser.parse_args()

    if args.only_evaluate:
        args.skip_denoise   = True
        args.skip_train_lid = True
        args.skip_stt       = True
        args.skip_translate = True
        args.skip_tts       = True
        args.skip_train_cm  = True

    main(args)
