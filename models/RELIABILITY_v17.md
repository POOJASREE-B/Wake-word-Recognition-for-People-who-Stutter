# Wake-word model v1.7: what changed and how it was tested

## What changed
- **Frame models** (CRNN `frame_model.pt`, embedding head `frame_emb.pt`) were fine-tuned on the full 32,706-clip training split (before: 1,600 clips).
- **Training audio was multi-condition**: noise (MS-SNSD noise_train, MIT licence; MUSAN noise, train half), babble and music at 0-25 dB SNR; reverberation; band limiting; level changes; 0.2-2.5 s of background before and after each clip.
- **Extra training material**: Piper near-miss words and "...Google..." sentences (voices never used in validation/test), the 23 development speakers, and the 117 fluent recordings.
- **Audio front-end**: none. High-pass + noise suppression + loudness normalisation was tested on validation data and lowered separation (AUC 0.905 -> 0.819 in quiet), so it is not used.
- **Verifier** (`verifier.json`): gradient-boosted trees on all 25 decoder features, stored as JSON and scored with numpy (`wakeword/gbt_verifier.py`; `detector.py` loads it when `"type": "gbt"`). It was trained with windows mined from the live loop on training-only negative streams (dataset negatives, MUSAN/MS-SNSD noise, ESC-50 folds 1-3) and live-style positive windows.
- **Threshold 0.91** was fixed by rule (on one grid used for every compared system): the lowest threshold with at most 1 false accept per hour on 4.3 h of calibration audio (MUSAN speech, MUSAN noise cal half, SEP-28k/control/dysarthric speech from the val split, ESC-50 fold 4). The confirming check may dip 0.08 below it (app v1.6 rule, unchanged).

## Held-out results at 0.91 (offline replay of the app's v1.6 listening rule)
| Test | Result |
|---|---|
| 10 unseen speakers who stutter, quiet room | 10/10 |
| same, MS-SNSD test noise 10 dB SNR | 8/10 |
| same, 5 dB SNR | 8/10 |
| 23 development speakers (seen in training), quiet / 10 dB / 5 dB | 23 / 18 / 18 of 23 |
| 13 new fluent recordings, quiet / 10 dB / 5 dB | 10 / 7 / 7 of 13 |
| False accepts, natural held-out audio, 4.9 h (LibriVox speech, spoken digits, SEP-28k test speech, MS-SNSD test noise, ESC-50 fold 5) | 2 events = 0.41 per hour |
| Stress test: back-to-back constructed hard units ("Hey" clip, pause, "Google" clip ...), 1.66 h | 18.1 per hour |
| Piper near-miss words (held-out voices) / "...Google..." sentences / Qualcomm clips accepted | 28/434, 11/307, 2/2161 |
| Hey Siri / Alexa from the 10 new speakers accepted | 0/20 |
| Pooled AUC, 10 new speakers vs 4,622 held-out negatives / vs near-miss words | 0.994 / 0.962 |

## Limits
- Ten unseen speakers give wide intervals: 8/10 has a 95% Wilson interval of 49-94%.
- Fluent recall in noise is weak (7/13 at 10 dB).
- The CRNN had about one pass over the full split; a second pass was cut short by a memory limit.
- The stress test shows that a "Hey" followed by a separate "Google" is accepted, by design (it looks like a blocked wake word).

The previous model files are kept in `models/previous_v16/`.
