# DeepFense Experiment Batch 1 Plan

This table tracks the planned experiments for "Batch 1". 
**Goal**: Baseline various Frontends and Backends with and without Augmentations.

*   **Dataset**: ASVSpoof19 (Train/Val)
*   **Epochs**: 100 (with Early Stopping patience=7)
*   **Base Transform**: Pad/Repeat to 64000 samples
*   **Seeds**: 2, 42, 240
*   **Augmentations**: RIR + Rawboost 3

| ID | Experiment Name | Frontend | Backend | Augmentation | Seed | Assignee | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | wav2vec2_AASIST_NoAug_seed2 | Wav2Vec2 | AASIST | NoAug | 2 | | Pending |
| 2 | wav2vec2_AASIST_NoAug_seed42 | Wav2Vec2 | AASIST | NoAug | 42 | | Pending |
| 3 | wav2vec2_AASIST_NoAug_seed240 | Wav2Vec2 | AASIST | NoAug | 240 | | Pending |
| 4 | wav2vec2_AASIST_Concat_RawBoost_RIR_seed2 | Wav2Vec2 | AASIST | Concat (Raw/RIR) | 2 | | Pending |
| 5 | wav2vec2_AASIST_Concat_RawBoost_RIR_seed42 | Wav2Vec2 | AASIST | Concat (Raw/RIR) | 42 | | Pending |
| 6 | wav2vec2_AASIST_Concat_RawBoost_RIR_seed240 | Wav2Vec2 | AASIST | Concat (Raw/RIR) | 240 | | Pending |
| 7 | wav2vec2_MLP_NoAug_seed2 | Wav2Vec2 | MLP | NoAug | 2 | | Pending |
| 8 | wav2vec2_MLP_NoAug_seed42 | Wav2Vec2 | MLP | NoAug | 42 | | Pending |
| 9 | wav2vec2_MLP_NoAug_seed240 | Wav2Vec2 | MLP | NoAug | 240 | | Pending |
| 10 | wav2vec2_MLP_Concat_RawBoost_RIR_seed2 | Wav2Vec2 | MLP | Concat (Raw/RIR) | 2 | | Pending |
| 11 | wav2vec2_MLP_Concat_RawBoost_RIR_seed42 | Wav2Vec2 | MLP | Concat (Raw/RIR) | 42 | | Pending |
| 12 | wav2vec2_MLP_Concat_RawBoost_RIR_seed240 | Wav2Vec2 | MLP | Concat (Raw/RIR) | 240 | | Pending |
| 13 | wav2vec2_Nes2Net_NoAug_seed2 | Wav2Vec2 | Nes2Net | NoAug | 2 | | Pending |
| 14 | wav2vec2_Nes2Net_NoAug_seed42 | Wav2Vec2 | Nes2Net | NoAug | 42 | | Pending |
| 15 | wav2vec2_Nes2Net_NoAug_seed240 | Wav2Vec2 | Nes2Net | NoAug | 240 | | Pending |
| 16 | wav2vec2_Nes2Net_Concat_RawBoost_RIR_seed2 | Wav2Vec2 | Nes2Net | Concat (Raw/RIR) | 2 | | Pending |
| 17 | wav2vec2_Nes2Net_Concat_RawBoost_RIR_seed42 | Wav2Vec2 | Nes2Net | Concat (Raw/RIR) | 42 | | Pending |
| 18 | wav2vec2_Nes2Net_Concat_RawBoost_RIR_seed240 | Wav2Vec2 | Nes2Net | Concat (Raw/RIR) | 240 | | Pending |
| 19 | wav2vec2_TCM_NoAug_seed2 | Wav2Vec2 | TCM | NoAug | 2 | | Pending |
| 20 | wav2vec2_TCM_NoAug_seed42 | Wav2Vec2 | TCM | NoAug | 42 | | Pending |
| 21 | wav2vec2_TCM_NoAug_seed240 | Wav2Vec2 | TCM | NoAug | 240 | | Pending |
| 22 | wav2vec2_TCM_Concat_RawBoost_RIR_seed2 | Wav2Vec2 | TCM | Concat (Raw/RIR) | 2 | | Pending |
| 23 | wav2vec2_TCM_Concat_RawBoost_RIR_seed42 | Wav2Vec2 | TCM | Concat (Raw/RIR) | 42 | | Pending |
| 24 | wav2vec2_TCM_Concat_RawBoost_RIR_seed240 | Wav2Vec2 | TCM | Concat (Raw/RIR) | 240 | | Pending |
| 25 | hubert_AASIST_NoAug_seed2 | HuBERT | AASIST | NoAug | 2 | | Pending |
| 26 | hubert_AASIST_NoAug_seed42 | HuBERT | AASIST | NoAug | 42 | | Pending |
| 27 | hubert_AASIST_NoAug_seed240 | HuBERT | AASIST | NoAug | 240 | | Pending |
| 28 | hubert_AASIST_Concat_RawBoost_RIR_seed2 | HuBERT | AASIST | Concat (Raw/RIR) | 2 | | Pending |
| 29 | hubert_AASIST_Concat_RawBoost_RIR_seed42 | HuBERT | AASIST | Concat (Raw/RIR) | 42 | | Pending |
| 30 | hubert_AASIST_Concat_RawBoost_RIR_seed240 | HuBERT | AASIST | Concat (Raw/RIR) | 240 | | Pending |
| 31 | hubert_MLP_NoAug_seed2 | HuBERT | MLP | NoAug | 2 | | Pending |
| 32 | hubert_MLP_NoAug_seed42 | HuBERT | MLP | NoAug | 42 | | Pending |
| 33 | hubert_MLP_NoAug_seed240 | HuBERT | MLP | NoAug | 240 | | Pending |
| 34 | hubert_MLP_Concat_RawBoost_RIR_seed2 | HuBERT | MLP | Concat (Raw/RIR) | 2 | | Pending |
| 35 | hubert_MLP_Concat_RawBoost_RIR_seed42 | HuBERT | MLP | Concat (Raw/RIR) | 42 | | Pending |
| 36 | hubert_MLP_Concat_RawBoost_RIR_seed240 | HuBERT | MLP | Concat (Raw/RIR) | 240 | | Pending |
| 37 | hubert_Nes2Net_NoAug_seed2 | HuBERT | Nes2Net | NoAug | 2 | | Pending |
| 38 | hubert_Nes2Net_NoAug_seed42 | HuBERT | Nes2Net | NoAug | 42 | | Pending |
| 39 | hubert_Nes2Net_NoAug_seed240 | HuBERT | Nes2Net | NoAug | 240 | | Pending |
| 40 | hubert_Nes2Net_Concat_RawBoost_RIR_seed2 | HuBERT | Nes2Net | Concat (Raw/RIR) | 2 | | Pending |
| 41 | hubert_Nes2Net_Concat_RawBoost_RIR_seed42 | HuBERT | Nes2Net | Concat (Raw/RIR) | 42 | | Pending |
| 42 | hubert_Nes2Net_Concat_RawBoost_RIR_seed240 | HuBERT | Nes2Net | Concat (Raw/RIR) | 240 | | Pending |
| 43 | hubert_TCM_NoAug_seed2 | HuBERT | TCM | NoAug | 2 | | Pending |
| 44 | hubert_TCM_NoAug_seed42 | HuBERT | TCM | NoAug | 42 | | Pending |
| 45 | hubert_TCM_NoAug_seed240 | HuBERT | TCM | NoAug | 240 | | Pending |
| 46 | hubert_TCM_Concat_RawBoost_RIR_seed2 | HuBERT | TCM | Concat (Raw/RIR) | 2 | | Pending |
| 47 | hubert_TCM_Concat_RawBoost_RIR_seed42 | HuBERT | TCM | Concat (Raw/RIR) | 42 | | Pending |
| 48 | hubert_TCM_Concat_RawBoost_RIR_seed240 | HuBERT | TCM | Concat (Raw/RIR) | 240 | | Pending |
| 49 | wavlm_AASIST_NoAug_seed2 | WavLM | AASIST | NoAug | 2 | | Pending |
| 50 | wavlm_AASIST_NoAug_seed42 | WavLM | AASIST | NoAug | 42 | | Pending |
| 51 | wavlm_AASIST_NoAug_seed240 | WavLM | AASIST | NoAug | 240 | | Pending |
| 52 | wavlm_AASIST_Concat_RawBoost_RIR_seed2 | WavLM | AASIST | Concat (Raw/RIR) | 2 | | Pending |
| 53 | wavlm_AASIST_Concat_RawBoost_RIR_seed42 | WavLM | AASIST | Concat (Raw/RIR) | 42 | | Pending |
| 54 | wavlm_AASIST_Concat_RawBoost_RIR_seed240 | WavLM | AASIST | Concat (Raw/RIR) | 240 | | Pending |
| 55 | wavlm_MLP_NoAug_seed2 | WavLM | MLP | NoAug | 2 | | Pending |
| 56 | wavlm_MLP_NoAug_seed42 | WavLM | MLP | NoAug | 42 | | Pending |
| 57 | wavlm_MLP_NoAug_seed240 | WavLM | MLP | NoAug | 240 | | Pending |
| 58 | wavlm_MLP_Concat_RawBoost_RIR_seed2 | WavLM | MLP | Concat (Raw/RIR) | 2 | | Pending |
| 59 | wavlm_MLP_Concat_RawBoost_RIR_seed42 | WavLM | MLP | Concat (Raw/RIR) | 42 | | Pending |
| 60 | wavlm_MLP_Concat_RawBoost_RIR_seed240 | WavLM | MLP | Concat (Raw/RIR) | 240 | | Pending |
| 61 | wavlm_Nes2Net_NoAug_seed2 | WavLM | Nes2Net | NoAug | 2 | | Pending |
| 62 | wavlm_Nes2Net_NoAug_seed42 | WavLM | Nes2Net | NoAug | 42 | | Pending |
| 63 | wavlm_Nes2Net_NoAug_seed240 | WavLM | Nes2Net | NoAug | 240 | | Pending |
| 64 | wavlm_Nes2Net_Concat_RawBoost_RIR_seed2 | WavLM | Nes2Net | Concat (Raw/RIR) | 2 | | Pending |
| 65 | wavlm_Nes2Net_Concat_RawBoost_RIR_seed42 | WavLM | Nes2Net | Concat (Raw/RIR) | 42 | | Pending |
| 66 | wavlm_Nes2Net_Concat_RawBoost_RIR_seed240 | WavLM | Nes2Net | Concat (Raw/RIR) | 240 | | Pending |
| 67 | wavlm_TCM_NoAug_seed2 | WavLM | TCM | NoAug | 2 | | Pending |
| 68 | wavlm_TCM_NoAug_seed42 | WavLM | TCM | NoAug | 42 | | Pending |
| 69 | wavlm_TCM_NoAug_seed240 | WavLM | TCM | NoAug | 240 | | Pending |
| 70 | wavlm_TCM_Concat_RawBoost_RIR_seed2 | WavLM | TCM | Concat (Raw/RIR) | 2 | | Pending |
| 71 | wavlm_TCM_Concat_RawBoost_RIR_seed42 | WavLM | TCM | Concat (Raw/RIR) | 42 | | Pending |
| 72 | wavlm_TCM_Concat_RawBoost_RIR_seed240 | WavLM | TCM | Concat (Raw/RIR) | 240 | | Pending |
