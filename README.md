# ca1d-sleep-apnea

> **Coordinate Attention adapted to 1D — a 14,001-parameter Stage-2 classifier that detects sleep apnea events from a 200×3 cascade-output feature matrix at 87% accuracy.**

This is **Paper C** in a 3-paper series on smartphone-deployable sleep monitoring. The CA-1D mechanism characterized in this paper is the **Stage-2 classifier of a cascaded two-stage pipeline** introduced in companion work. Companion repositories:
- [`audio-sleep-cnn-baselines`](https://github.com/somnisense/audio-sleep-cnn-baselines) (Paper A — the cascaded two-stage pipeline, of which the model here is the Stage-2 instance)
- [`apnea-compression-pipeline`](https://github.com/somnisense/apnea-compression-pipeline) (Paper E — model compression of the architecture characterized here)

A 1D adaptation of the Coordinate Attention mechanism (originally proposed for 2D mobile vision networks) applied to audio-based sleep apnea detection. The architecture preserves **temporal-position information** in the attention map — for typical obstructive apnea events the discriminative information tends to cluster around event onset and recovery transitions rather than being uniformly distributed across the analysis window, and most off-the-shelf attention designs collapse exactly the dimension you'd want to keep.

Evaluated against a vanilla 1D CNN and a Squeeze-and-Excitation 1D CNN of comparable size, under an identical 5-seed bootstrap protocol on **40 participants / 80 person-nights** of audio paired with PSG annotations.

---

## Headline result

**14,001 parameters → 87.14% accuracy, 86.94% F1.**

A 93.2% parameter reduction relative to a 204,801-parameter 1D-CNN baseline, while *improving* six of seven evaluation metrics (Accuracy, Precision, Specificity, F1, AUC-ROC, AUC-PR) statistically significantly under paired bootstrap.

| Model | Params | Accuracy (95% CI) | F1 (95% CI) | Precision |
|---|---|---|---|---|
| Original 1D CNN baseline | 204,801 | 83.82% (82.61, 85.14) | 83.99% (82.87, 85.05) | 82.13% |
| SE-Attention 1D CNN | 13,425 | 86.78% | 86.27% | 84.13% |
| **Coord-Attn 1D CNN** *(ours)* | **14,001** | **87.14% (85.14, 89.68)** | **86.94% (84.28, 89.77)** | **86.75%** |

vs SE-Attn at comparable parameter count: Coord-Attn provides a statistically significant **Precision (+2.62 pp)** and **Specificity (+3.40 pp)** advantage, but F1 / AUC differences are statistically indistinguishable from zero under paired bootstrap — a finding the paper reports precisely rather than overstates.

---

## Preprint

The full paper (English; results tables, per-seed metrics, figures, discussion) is available at:

- **Zenodo (canonical, citable DOI)**: [10.5281/zenodo.20663376](https://doi.org/10.5281/zenodo.20663376)
- **ORCID**: [0009-0002-4798-5161](https://orcid.org/0009-0002-4798-5161)
- **arXiv (cs.LG)**: *planned*

This repository is the **code companion** to the paper, not a mirror of it.

---

## Architecture in one diagram

```
Input (200, 3)               # SPL, ΔSPL, snore-presence-indicator-from-Stage-1 per second
  → Conv1D(8,  k=3) → BN → ReLU → MaxPool(2) → Coord-Attn block #1
  → Conv1D(16, k=3) → BN → ReLU → MaxPool(2) → Coord-Attn block #2
  → Conv1D(32, k=3) → BN → ReLU → MaxPool(2) → Coord-Attn block #3
  → GlobalAveragePool1D
  → Dense(64, ReLU) → Dropout(0.2)
  → Dense(1, Sigmoid)
```

The third input channel — binary snore-presence indicator at 1 Hz — is the **per-second output of the Stage-1 short-window snore-detection CNN** of the cascade pipeline ([Paper A](https://github.com/somnisense/audio-sleep-cnn-baselines)). This Stage-1-output-as-Stage-2-input-channel structure is what makes the cascade pipeline cascade.

A Coord-Attn block factorizes attention along the time axis (the input has no spatial dimension), preserves the relative position of attention activations, and uses a global-local fusion design described in §2.2 of the paper. Full TensorFlow/Keras implementation in [`code/models/coord_attn_1d.py`](code/models/coord_attn_1d.py).

---

## Reproduce

```bash
cd code
pip install -r requirements.txt

python run_experiments.py        # 5 seeds × 3 architectures
python analyze_results.py        # bootstrap 95% CI + paired bootstrap
python generate_figures.py       # all paper figures
```

Random seeds fixed per-experiment. Each seed reproduces bit-exactly on the same hardware (consumer M2 CPU, no specialized accelerator).

---

## Data

This is an **algorithm-framework release** — no audio recordings and no feature matrices are distributed with this repository, by design. The 2,953-sample / 40-participant / 80-person-night dataset was collected under participant consent that does not cover public release of either the audio waveforms or the derived feature matrices.

The architecture and training code are task-agnostic and run against any dataset that conforms to the published I/O contract:

- **Sleep apnea detection (Stage-2 task)** — binary classification of **Abnormal** (apnea or hypopnea event) vs **Normal** windows, on 200×3 acoustic-feature matrices over 200-second windows. The third feature channel is expected to be a binary snore-presence indicator at 1 Hz (which, in the cascade pipeline, is produced by the Stage-1 model of [Paper A](https://github.com/somnisense/audio-sleep-cnn-baselines), but this repository runs against any 200×3 input regardless of how the third channel is produced).

Point the loaders at your own dataset by placing it at `<paper_C_ca1d_apnea>/data/` (sibling to `code/`); the directory layout expected by the loaders is documented in the header of `code/run_experiments.py`.

---

## Patent notice

> The Coord-Attn 1D block disclosed in this paper, the cascaded two-stage architecture in which it serves as Stage-2 classifier, and the compression pipeline and system-level procedures used in production deployment are the subject of **three co-filed U.S. provisional patent applications** by SomniAI LLC (filed 2026-06; application numbers pending). The paper and this repository disclose the CA-1D mathematical formulation, training protocol, and evaluation methodology for reproducibility; certain implementation specifics — particularly the multi-stage gating, event-driven triggering, and privacy-preserving on-device system architecture — are covered by the co-filed patent applications and are not described here.
>
> Code in this repository is licensed under MIT for research, evaluation, and reproducibility purposes (see [LICENSE](LICENSE)).

---

## Citation

```bibtex
@misc{yang2026ca1d,
  author       = {Yang, L.},
  title        = {Coordinate Attention for {1D} Audio-Based Sleep Apnea Detection:
                  A Multi-Seed Empirical Study on Smartphone-Deployable Architectures},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20663376},
  howpublished = {Zenodo preprint, \url{https://doi.org/10.5281/zenodo.20663376}},
  note         = {Code: \url{https://github.com/somnisense/ca1d-sleep-apnea}},
}
```

---

## License

Code: **MIT**. Patent rights are not granted by this license — see *Patent notice* above.

---

## About

Built and maintained by [**SomniAI LLC**](https://github.com/somnisense). The production app (SomniSense, Wellness category) that uses this line of work runs on-device on **iOS and Android**: → **[somnisense.top](https://www.somnisense.top)**.

Questions: `service@somnisense.top`.
