# Experiment Metrics Tables

Model name abbreviations:

- `EMTTS-Pretrain (Pred)` = `EMTTS_Pretraining_Qwen_WithCrossLingual_3_5_Delay..._Phoneme_predicted_argmax...`
- `PO-NoViZhHi-N8 (Pred)` = `PO_CodeRefactor_NoViZhHi_NGEN8..._Phoneme_predicted_argmax...`
- `EMTTS-Pretrain (GT)` = `EMTTS_Pretraining_Qwen_WithCrossLingual_3_5_Delay..._Phoneme_gt_argmax...`
- `PO-NoViZhHi-N8 (GT)` = `PO_CodeRefactor_NoViZhHi_NGEN8..._Phoneme_gt_argmax...`
- `PO-NoZh-N12 (Pred)` = `PO_CodeRefactor_NoZh_NGEN12..._Phoneme_predicted_argmax...`

Datasets:

- `French` = `french`
- `Digits` = `riva_hard_digits`
- `Letters` = `riva_hard_letters`
- `Money` = `riva_hard_money`
- `Riva Challenging` = average of (`Digits`, `Letters`, `Money`)
- `Short` = `riva_hard_short`
- `VCTK` = `vctk`
- `LibriSeen` = `libritts_seen`
- `LibriClean` = `libritts_test_clean`
- `RivaMultiBPE` = `riva_multibpe`

## CER Cumulative


| Model                 | French | Digits | Letters | Money  | Riva Challenging | Short  | VCTK   | LibriSeen | LibriClean | RivaMultiBPE |
| --------------------- | ------ | ------ | ------- | ------ | ---------------- | ------ | ------ | --------- | ---------- | ------------ |
| EMTTS-Pretrain (Pred) | 0.0655 | 0.0258 | 0.0567  | 0.0110 | 0.0312           | 0.0322 | 0.0123 | 0.0107    | 0.0044     | 0.0331       |
| PO-NoViZhHi-N8 (Pred) | 0.0556 | 0.0220 | 0.0456  | 0.0142 | 0.0273           | 0.0239 | 0.0046 | 0.0050    | 0.0045     | 0.0499       |
| EMTTS-Pretrain (GT)   | 0.0490 | 0.0226 | 0.0579  | 0.0122 | 0.0309           | 0.0198 | 0.0039 | 0.0043    | 0.0047     | 0.0162       |
| PO-NoViZhHi-N8 (GT)   | 0.0438 | 0.0235 | 0.0474  | 0.0159 | 0.0289           | 0.0235 | 0.0055 | 0.0042    | 0.0047     | 0.0197       |
| PO-NoZh-N12 (Pred)    | 0.9820 | 0.0201 | 0.0387  | 0.0118 | 0.0235           | 0.0136 | 0.0030 | 0.0046    | 0.0043     | 0.0369       |


## SSIM Pred Context Avg


| Model                 | French | Digits | Letters | Money  | Riva Challenging | Short  | VCTK   | LibriSeen | LibriClean | RivaMultiBPE |
| --------------------- | ------ | ------ | ------- | ------ | ---------------- | ------ | ------ | --------- | ---------- | ------------ |
| EMTTS-Pretrain (Pred) | 0.6642 | 0.7056 | 0.6639  | 0.7151 | 0.6949           | 0.3665 | 0.6581 | 0.7996    | 0.8032     | 0.7264       |
| PO-NoViZhHi-N8 (Pred) | 0.6686 | 0.7079 | 0.6653  | 0.7167 | 0.6966           | 0.3640 | 0.6560 | 0.8028    | 0.8040     | 0.7247       |
| EMTTS-Pretrain (GT)   | 0.6651 | 0.7060 | 0.6643  | 0.7128 | 0.6944           | 0.3672 | 0.6659 | 0.7985    | 0.8023     | 0.7204       |
| PO-NoViZhHi-N8 (GT)   | 0.6590 | 0.7079 | 0.6666  | 0.7184 | 0.6976           | 0.3653 | 0.6640 | 0.8026    | 0.8040     | 0.7259       |
| PO-NoZh-N12 (Pred)    | 0.6805 | 0.7066 | 0.6683  | 0.7155 | 0.6968           | 0.3786 | 0.6584 | 0.7975    | 0.8026     | 0.7233       |


## UTMOS


| Model                 | French | Digits | Letters | Money  | Riva Challenging | Short  | VCTK   | LibriSeen | LibriClean | RivaMultiBPE |
| --------------------- | ------ | ------ | ------- | ------ | ---------------- | ------ | ------ | --------- | ---------- | ------------ |
| EMTTS-Pretrain (Pred) | 2.7335 | 3.3208 | 3.2727  | 3.2975 | 3.2970           | 2.7342 | 3.4448 | 3.5053    | 3.4621     | 3.3327       |
| PO-NoViZhHi-N8 (Pred) | 2.8172 | 3.3278 | 3.2953  | 3.3439 | 3.3223           | 2.6765 | 3.4075 | 3.5085    | 3.4721     | 3.3519       |
| EMTTS-Pretrain (GT)   | 2.8002 | 3.3298 | 3.2641  | 3.3019 | 3.2986           | 2.7189 | 3.3916 | 3.5155    | 3.4747     | 3.3364       |
| PO-NoViZhHi-N8 (GT)   | 2.7908 | 3.3347 | 3.2948  | 3.3259 | 3.3185           | 2.6659 | 3.4212 | 3.5200    | 3.4784     | 3.3788       |
| PO-NoZh-N12 (Pred)    | 2.9275 | 3.4328 | 3.3684  | 3.4159 | 3.4057           | 2.7637 | 3.5527 | 3.6091    | 3.5604     | 3.4586       |


## Focused Comparison (Selected Models/Datasets)

| Model                 | Riva Challenging CER | Riva Challenging SSIM | Riva Challenging UTMOS | LibriClean CER | LibriClean SSIM | LibriClean UTMOS |
| --------------------- | -------------------- | --------------------- | ---------------------- | -------------- | --------------- | ---------------- |
| EMTTS-Pretrain (Pred) | 0.0312               | 0.6949                | 3.2970                 | 0.0044         | 0.8032          | 3.4621           |
| PO-NoZh-N12 (Pred)    | 0.0235               | 0.6968                | 3.4057                 | 0.0043         | 0.8026          | 3.5604           |
| MagpieTTS (Post-GRPO) | 0.0223               | 0.7190                | 3.3100                 | 0.0040         | 0.8640          | 3.4400           |

Note: CER values for `MagpieTTS (Post-GRPO)` were read from chart labels in percent and converted to fractions to match this table format.

