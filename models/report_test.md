# Evaluation on the **test** split (threshold 0.91)

- wake-word clips: 2009   negative clips: 3175 (2.413 h)
- **FRR** (missed wake words): overall 0.3221  | fluent 0.334  | disfluent 0.3109
- **FAR** (false accepts on negatives): 0.0101  (32 clips, 13.26 per hour of negative audio)   AUC: 0.9803

## FRR by disfluency type

| type | n | FRR |
|---|---|---|
| block_interruption | 115 | 0.4 |
| final_prolongation | 74 | 0.2297 |
| fluent | 970 | 0.334 |
| interjection | 76 | 0.2368 |
| mixed | 137 | 0.4745 |
| revision | 57 | 0.2105 |
| sound_prolongation | 126 | 0.3254 |
| sound_repetition | 178 | 0.309 |
| syllable_repetition | 166 | 0.2771 |
| word_repetition | 110 | 0.2091 |

## FAR by negative category

| category | n | FAR |
|---|---|---|
| hard:goo_incomplete | 36 | 0.0 |
| hard:google_google | 40 | 0.1 |
| hard:google_only | 48 | 0.0208 |
| hard:google_stutter_only | 51 | 0.0588 |
| hard:google_then_hey | 32 | 0.0 |
| hard:hey_goo_no_gle | 47 | 0.0213 |
| hard:hey_only | 42 | 0.0 |
| hard:hey_other_word | 39 | 0.0256 |
| hard:hey_prolonged | 34 | 0.0 |
| hard:hey_repeated | 37 | 0.0 |
| hard:hey_trailing_block | 44 | 0.0 |
| hard:other_word_google | 43 | 0.1163 |
| hard:scrambled_units | 46 | 0.087 |
| noise:beeps | 3 | 0.0 |
| noise:breaths | 3 | 0.0 |
| noise:clicks | 3 | 0.0 |
| noise:cough_burst | 3 | 0.0 |
| noise:hum | 3 | 0.0 |
| noise:keyboard | 3 | 0.0 |
| noise:pink | 3 | 0.0 |
| noise:room_tone | 3 | 0.0 |
| noise:white | 3 | 0.0 |
| speech:control | 40 | 0.0 |
| speech:dysarthric | 40 | 0.025 |
| speech:real_stutter_clips | 2295 | 0.0 |
| tts:hard_google | 41 | 0.0244 |
| tts:near_miss | 83 | 0.1325 |
| tts:noise | 27 | 0.0 |
| tts:other_wake | 29 | 0.0 |
| tts:partial | 25 | 0.0 |
| tts:speech | 29 | 0.0 |

## FRR by speaker kind

| kind | n | FRR |
|---|---|---|
| real_recording | 816 | 0.3934 |
| tts_voice | 1193 | 0.2733 |

## FRR by stuttering profile

| stuttering profile | n | FRR |
|---|---|---|
| developmental | 498 | 0.3353 |
| neurogenic | 290 | 0.2241 |
| none | 970 | 0.334 |
| project_dataset | 69 | 0.4783 |
| psychogenic | 182 | 0.3187 |

## FRR by emotion

| emotion | n | FRR |
|---|---|---|
| angry | 168 | 0.4345 |
| calm | 80 | 0.1875 |
| excited | 88 | 0.2841 |
| friendly | 112 | 0.3393 |
| grumpy | 112 | 0.2589 |
| happy | 72 | 0.2778 |
| neutral | 104 | 0.1731 |
| sad | 80 | 0.225 |
| scared | 112 | 0.5625 |
| tired | 128 | 0.1953 |
| unspecified | 953 | 0.3389 |

## FRR by age group

| age group | n | FRR |
|---|---|---|
| adult | 336 | 0.2292 |
| child | 256 | 0.4883 |
| elderly | 224 | 0.2545 |
| teen | 240 | 0.2708 |
| unspecified | 953 | 0.3389 |

## FRR by secondary behaviour

| secondary behaviour | n | FRR |
|---|---|---|
| inhale | 112 | 0.2946 |
| inhale,tap | 41 | 0.4878 |
| inhale,tap,tense_voicing | 36 | 0.5 |
| inhale,tense_voicing | 148 | 0.3176 |
| none | 1500 | 0.3093 |
| tense_voicing | 172 | 0.3779 |

## FRR by source (label / how it was made / voice)

| source (label / how it was made / voice) | n | FRR |
|---|---|---|
| disfluent / recorded_splicing / real_recording | 36 | 0.9167 |
| disfluent / recorded_splicing / tts_voice | 33 | 0.0 |
| disfluent / wsola_subword_splicing / real_recording | 390 | 0.3385 |
| disfluent / wsola_subword_splicing / tts_voice | 580 | 0.2724 |
| fluent / unit_resynthesis / real_recording | 390 | 0.4 |
| fluent / unit_resynthesis / tts_voice | 580 | 0.2897 |

## Threshold sweep

| threshold | FRR | FAR |
|---|---|---|
| 0.3 | 0.0996 | 0.0324 |
| 0.5 | 0.1254 | 0.0249 |
| 0.7 | 0.1976 | 0.0173 |
| 0.9 | 0.3051 | 0.0101 |
| 0.95 | 0.435 | 0.0057 |
| 0.99 | 0.7476 | 0.0013 |
