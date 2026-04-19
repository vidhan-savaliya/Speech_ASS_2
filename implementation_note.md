# Speech Understanding (PA2) — Implementation Notes
**Author:** Vidhatha
**Topic:** Non-Obvious Design Choices per Task

---

### Part I: Robust Code-Switched Transcription (STT)
**Design Choice:** Logit Biasing Over Interpolated Language Models
**Explanation:** Instead of training a separate N-Gram language model or a strict KenLM and doing shallow fusion at beam-search time (which is computationally expensive for 10 minutes of audio), I implemented an explicit **Logit Biasing mechanism** injected directly into Whisper’s vocabulary embedding outputs. By forcing a static +0.2 vector bias onto the BPE tokens corresponding to my Speech Course syllabus terms (like "stochastic", "MFA", "spectrogram") before the softmax temperature scaling, the model is mathematically pushed to prioritize these technical terms during decoding. This is highly efficient and guarantees that technical terms are caught natively within Whisper's existing cross-attention layers, avoiding hallucination loops common in shallow fusion when cross-lingual contexts (Hinglish) abruptly shift.

### Part II: Phonetic Mapping & Translation
**Design Choice:** Bridging Hinglish to Marathi via IPA Pivot
**Explanation:** Standard code-switching translation fails because words are phonetically written in English script but carry Hindi semantics (e.g., "kar rahe hain"). Instead of directly translating Hinglish text, the transcript is first grapheme-to-phoneme (G2P) mapped into a unified **International Phonetic Alphabet (IPA)** using `espeak-ng` augmented with custom Hinglish heuristic replacements. Translating the textual representation of the IPA, rather than the romanized text, allows the underlying NLLB-200 translation model to rely on phonetic proximity to internal Hindi vocabulary representations, bridging the domain gap significantly better before emitting the target Marathi text. 

### Part III: Zero-Shot Cross-Lingual Voice Cloning (TTS)
**Design Choice:** DTW MFCC Alignment over Flat Time-Stretching
**Explanation:** To transfer the professor's prosody onto my voice, simple time-stretching or naive linear alignment fails due to the vast timing differences between naturally spoken Hinglish verses the synthesised Marathi MMS TTS outputs. I chose to align the sequences using **Dynamic Time Warping (DTW) computed over 13-bin MFCCs**. MFCCs are largely speaker-independent but highly correlated to phonetic content, making it the perfect feature space for finding the exact path connecting identical linguistic events between the source lecture and the generated synthesis. Once aligned, applying the source $F_0$ curve via Praat's overlap-add PSOLA perfectly preserves the natural "teaching" cadence onto the LRL translation.

### Part IV: Adversarial Robustness & Spoofing Detection
**Design Choice:** LFCC over MFCC for the Countermeasure (CM) System
**Explanation:** While MFCCs are the standard for voice recognition, they heavily emphasize the lower frequencies (approximating human mel-scale hearing), which largely capture linguistic content rather than acoustic artifacts. I aggressively chose **Linear Frequency Cepstral Coefficients (LFCC)** for the Anti-Spoofing Classifier. LFCCs process the spectrum evenly, which critically retains the high-frequency spectral distortions, phase mismatches, and algorithmic smoothing artifacts introduced by TTS vocoders (like VITS/HiFi-GAN). This high-frequency retention allows a lightweight 1D-CNN to easily detect the spoofed clone outperforming an identically sized MFCC-CNN.
