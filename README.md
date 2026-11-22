# Calibration-Aware Vision-Based Road Surface Analysis for Uncertainty Propagation in Braking Systems

**Mouhamadou Makhtar Fall, Samia Ainouz, Pierre-Jean Lapray, Jean-Philippe Tarel** — IEEE ITSC

Vision-based friction estimation usually classifies the road surface and maps the
predicted class probabilities to friction values. This code studies how the
**calibration** of those probabilities propagates to the friction estimate, its
uncertainty, and the braking distance, for models of similar accuracy.

**Pipeline**

1. A six-class surface classifier (dry, wet, water, fresh snow, melted snow, ice), trained
   with cross-entropy (CE) or margin-based label smoothing (MbLS), optionally followed by
   temperature scaling (TS).
2. Friction as a mixture of per-class uniform laws weighted by p(c | x):

   $$\mu \mid x \sim \sum_c p(c\mid x)\, \mathcal U(\mu_c^{\min}, \mu_c^{\max}),\qquad
   \hat\mu(x) = \sum_c p(c\mid x)\,\bar\mu_c$$

   $$\mathrm{Var}(\mu\mid x) = \sum_c p(c\mid x)\frac{(\mu_c^{\max}-\mu_c^{\min})^2}{12}
   + \sum_c p(c\mid x)\,(\bar\mu_c-\hat\mu(x))^2$$

3. Stopping distance $d(\mu) = v^2 / (2\mu g)$, and its error with respect to the mean friction
   of the true class: MABE (m) and RBE (%) at 90 km/h.

## Install

```bash
conda create -n roadcal python=3.9 -y && conda activate roadcal
# install torch / torchvision for your CUDA version first: https://pytorch.org
pip install -e .
```

## Data

The RSCD dataset (Zhao et al., IEEE T-ITS 2023) is reorganised into six classes and stored as an
`ImageFolder` under `data/road_dataset` (override with `data.data_root=/abs/path`):

```
road_dataset/{train,val,test}/{dry,wet,water,fresh_snow,melted_snow,ice}/*.jpg
```

`dry`, `wet` and `water` gather all materials (asphalt, concrete, mud, gravel); `ice` includes
compacted snow.

## Usage

```bash
bash scripts/paper.sh effnet      # EfficientNet-B0, CE and MbLS
bash scripts/paper.sh resnet50    # ResNet-50 (ImageNet-pretrained), CE and MbLS
```

or a single run:

```bash
python tools/train_net.py model=efficientnet_b0 loss=ce \
    optim=sgd optim.lr=0.1 scheduler=one_cycle train.max_epoch=60
```

At the end of training, `best.pth` is evaluated on the test set **before and after temperature
scaling** (T fitted on the validation set by grid search over 0.1, 0.2, …, 10 minimising ECE).
The run folder `outputs/road/<model>-<loss>-sgd/<date>/` then contains:

| File | Content |
|---|---|
| `results.json` | T, Acc, NLL, ECE, AECE, CECE (%, 15 bins), mean μ̂, mean Var(μ) and its two terms, MABE and RBE |
| `*_reliability.png` | reliability diagram (Fig. 2) |
| `*_confusion.png` | confusion matrix (Figs. 3-4) |
| `test_logits.npz` | test logits, labels and T, to recompute anything without the GPU |

To evaluate an existing checkpoint:

```bash
python tools/test_net.py model=efficientnet_b0 loss=ce test.checkpoint=/abs/path/best.pth
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `loss` | `ce` | `ce` or `logit_margin` (MbLS, margin 10, α 0.1) |
| `friction` | `experiments` | friction intervals, see [`configs/friction/`](configs/friction) |
| `braking.speeds` | `[25.0]` | speeds (m/s) for MABE / RBE, e.g. `"braking.speeds=[13.9,25.0]"` |

The friction model is in [`calibrate/friction.py`](calibrate/friction.py) and can be used on its own:

```python
from calibrate.friction import FrictionModel, braking_errors
fm = FrictionModel({"dry": (0.65, 0.80), ...}, class_names)
m = fm.moments(probs)                       # mu_hat, var_physical, var_classification, var_total
mabe, rbe = braking_errors(m["mu_hat"], fm.reference(labels), speed=25.0)
```

Unit tests: `pytest tests`.

## Results

| Backbone | Model | Acc | ECE | T* | Mean μ̂ | Var(μ) | MABE (m) | RBE (%) |
|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | CE | 93.78 | 0.72 | 1.00 | 0.4205 | 0.0050 | 2.326 | 2.57 |
| | CE + TS | 93.78 | 0.34 | 0.90 | 0.4204 | 0.0047 | 2.079 | 2.38 |
| | MbLS | 93.47 | 3.40 | 1.00 | 0.4209 | 0.0082 | 5.081 | 4.61 |
| | MbLS + TS | 93.47 | 1.25 | 0.80 | 0.4206 | 0.0056 | 2.815 | 2.94 |
| ResNet-50 | CE | 97.08 | 0.50 | 1.00 | 0.4207 | 0.0037 | 0.726 | 0.88 |
| | CE + TS | 97.08 | 0.41 | 1.20 | 0.4207 | 0.0040 | 0.969 | 1.07 |
| | MbLS | 97.06 | 1.27 | 1.00 | 0.4208 | 0.0052 | 2.028 | 1.84 |
| | MbLS + TS | 97.06 | 0.73 | 0.90 | 0.4208 | 0.0045 | 1.356 | 1.33 |

Braking errors at 90 km/h. Full tables in the paper.

## Acknowledgements

Built on [MbLS](https://github.com/by-liu/MbLS) (Liu et al., CVPR 2022): training loop,
MbLS loss, calibration metrics and temperature scaling. Road images from RSCD (Zhao et al.,
IEEE T-ITS 2023). Supported by the French ANR project INARI (ANR-23-CE22-0002).

## Citation

```bibtex
@inproceedings{fall2026roadcalibration,
  title     = {Calibration-Aware Vision-Based Road Surface Analysis for Uncertainty Propagation in Braking Systems},
  author    = {Fall, Mouhamadou Makhtar and Ainouz, Samia and Lapray, Pierre-Jean and Tarel, Jean-Philippe},
  booktitle = {IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year      = {2026}
}
```
