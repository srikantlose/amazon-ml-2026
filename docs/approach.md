# Amazon ML Challenge 2026: Approach

**Team:** _name_ · **Members:** _names_ · **Final score:** _public LB_ / _CV_

> Keep this to 1–2 pages. Fill each section as the work happens, not at hour 70.

## 1. Problem & metric
- Task:
- Input data / modalities:
- Target and output format:
- Metric (official definition) and what it rewards:

## 2. Data insights (EDA)
- Sizes, missing values, image availability (% downloaded):
- Target distribution (skew, outliers, zeros):
- Findings that shaped the model:

## 3. Preprocessing
- Text cleaning / field parsing:
- Image handling (resize, missing images → placeholder + `has_image`):
- Target transform:

## 4. Features
- Handcrafted (quantities/units, pack counts, text stats, brand token):
- TF-IDF → SVD:
- Text embeddings (model, license):
- Image embeddings (model, license):

## 5. Models
| Model | Inputs | Key settings | CV |
|---|---|---|---|
| | | | |

## 6. Validation strategy
- Folds (scheme, k, seed), and why:
- How well CV tracked the public LB:

## 7. Results (CV vs LB)
| Run id | Description | CV | Public LB |
|---|---|---|---|
| | | | |

## 8. Ensembling
- Members, weights, blend space (linear / log):
- Gain over the best single model:

## 9. What didn't work
-

## 10. Compute used
- Hardware (instance types, hours):
- Full test-set inference time for the final pipeline:
- Model sizes and licenses (all ≤ 8B params, MIT/Apache 2.0):
