"""
evaluate.py
───────────
Standalone evaluation script — computes all 5 grading metrics and
writes data/metrics_report.json.

Metrics:
  1. LID F1 score       (target ≥ 0.85)
  2. WER                (target < 15% EN, < 25% HI)
  3. MCD                (target < 8.0 dB)
  4. Anti-Spoof EER     (target < 10%)
  5. FGSM epsilon       (SNR > 40 dB)

Usage:
    python evaluate.py
    python evaluate.py --wer-reference data/reference.txt
"""

import os
import json
import argparse
import traceback

from config import Config


def _ok(val, threshold, lower_is_better=True) -> str:
    if val is None:
        return "N/A"
    passed = val < threshold if lower_is_better else val > threshold
    return f"{'✓ PASS' if passed else '✗ FAIL'}  (target {'<' if lower_is_better else '>'} {threshold})"


def run_evaluation(wer_ref_path: str = None) -> dict:
    Config.setup()
    results = {}

    print("\n" + "=" * 60)
    print("  Speech PA2 — Evaluation")
    print("=" * 60)

    # ── 1. LID F1 ────────────────────────────────────────────────────────────
    print("\n[Eval] 1. Language Identification F1…")
    try:
        from part1_lid import evaluate_lid
        f1 = evaluate_lid(Config.LID_WEIGHTS)
        results["lid_f1"] = round(float(f1), 4)
        print(f"       LID F1 = {results['lid_f1']}  "
              f"{_ok(f1, 0.85, lower_is_better=False)}")
    except Exception as e:
        print(f"       [!] LID eval failed: {e}")
        results["lid_f1"] = None

    # ── 2. WER ───────────────────────────────────────────────────────────────
    print("\n[Eval] 2. Word Error Rate…")
    try:
        from part1_stt import report_wer
        wer_info = report_wer(Config.OUTPUT_TRANSCRIPT, wer_ref_path)
        results.update(wer_info)
        if "overall_wer" in wer_info:
            val = wer_info["overall_wer"]
            print(f"       WER = {val*100:.2f}%  {_ok(val, 0.25)}")
        else:
            print(f"       Token counts: {wer_info}")
    except Exception as e:
        print(f"       [!] WER eval failed: {e}")

    # ── 3. MCD ───────────────────────────────────────────────────────────────
    print("\n[Eval] 3. Mel-Cepstral Distortion (MCD)…")
    try:
        if (os.path.exists(Config.FINAL_AUDIO) and
                os.path.exists(Config.REF_VOICE)):
            from part3_tts_prosody import compute_mcd
            mcd = compute_mcd(Config.FINAL_AUDIO, Config.REF_VOICE)
            results["mcd"] = round(float(mcd), 3)
            print(f"       MCD = {mcd:.2f} dB  {_ok(mcd, 8.0)}")
        else:
            print("       [!] Audio files missing; skipping MCD.")
            results["mcd"] = None
    except Exception as e:
        print(f"       [!] MCD eval failed: {e}")
        traceback.print_exc()

    # ── 4. Anti-Spoof EER ────────────────────────────────────────────────────
    print("\n[Eval] 4. Anti-Spoofing EER…")
    try:
        from part4_antispoof import evaluate_cm_eer
        eer = evaluate_cm_eer(Config.REF_VOICE, Config.FINAL_AUDIO,
                              Config.CM_WEIGHTS)
        results["eer"] = round(float(eer), 4)
        print(f"       EER = {eer*100:.2f}%  {_ok(eer, 0.10)}")
    except Exception as e:
        print(f"       [!] EER eval failed: {e}")
        results["eer"] = None

    # ── 5. FGSM Adversarial Epsilon ───────────────────────────────────────────
    print("\n[Eval] 5. FGSM Adversarial Robustness…")
    try:
        from part4_antispoof import fgsm_adversarial_attack
        fgsm = fgsm_adversarial_attack(Config.DENOISED_AUDIO,
                                       Config.LID_WEIGHTS)
        results["fgsm_success"]   = fgsm.get("success")
        results["best_epsilon"]   = fgsm.get("best_epsilon")
        results["fgsm_snr_db"]    = fgsm.get("snr")
        eps = fgsm.get("best_epsilon")
        print(f"       Min ε = {eps:.2e}"
              if eps else "       Could not find valid ε.")
        if fgsm.get("snr"):
            print(f"       SNR  = {fgsm['snr']:.1f} dB  "
                  f"{_ok(fgsm['snr'], 40.0, lower_is_better=False)}")
    except Exception as e:
        print(f"       [!] FGSM eval failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    rows = [
        ("LID F1",        results.get("lid_f1"),       0.85, False),
        ("MCD (dB)",      results.get("mcd"),           8.0,  True),
        ("EER",           results.get("eer"),           0.10, True),
    ]
    for name, val, thr, lib in rows:
        if val is not None:
            flag = _ok(val, thr, lib)
            print(f"  {name:<22}  {val:<12}  {flag}")
        else:
            print(f"  {name:<22}  {'N/A':<12}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(Config.METRICS_REPORT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Metrics report saved → {Config.METRICS_REPORT}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all PA2 metrics")
    parser.add_argument("--wer-reference", default=None,
                        help="Path to ground-truth reference transcript for WER")
    args = parser.parse_args()
    run_evaluation(wer_ref_path=args.wer_reference)
