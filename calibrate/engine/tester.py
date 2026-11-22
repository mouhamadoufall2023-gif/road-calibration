"""Test a trained road surface classifier: classification, calibration, friction
uncertainty and braking distance error, before and after temperature scaling
(Tables II-IV and Figs. 2-4 of the paper)."""
import json
import logging
import os.path as osp
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from terminaltables.ascii_table import AsciiTable  # noqa: E402

from calibrate.evaluation import CalibrateEvaluator  # noqa: E402
from calibrate.evaluation.metrics import ECELoss  # noqa: E402
from calibrate.friction import FrictionModel, braking_errors  # noqa: E402
from calibrate.utils import load_checkpoint  # noqa: E402

logger = logging.getLogger(__name__)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> float:
    """Grid search T in {0.1, 0.2, ..., 10.0} minimising the validation ECE."""
    ece = ECELoss(num_bins)
    grid = np.round(np.arange(1, 101) * 0.1, 1)
    scores = [ece(logits / t, labels).item() for t in grid]
    return float(grid[int(np.argmin(scores))])


_STYLE = {"font.size": 9}


def plot_reliability(probs: np.ndarray, labels: np.ndarray, path: str,
                     num_bins: int = 15, title: str = "") -> None:
    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == labels
    edges = np.linspace(0, 1, num_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, num_bins - 1)
    centers, width = (edges[:-1] + edges[1:]) / 2, 1.0 / num_bins
    acc = np.array([correct[which == b].mean() if (which == b).any() else 0 for b in range(num_bins)])
    avg = np.array([conf[which == b].mean() if (which == b).any() else 0 for b in range(num_bins)])
    with plt.rc_context(_STYLE):
        _draw_reliability(path, centers, width, acc, avg, title)


def _draw_reliability(path, centers, width, acc, avg, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(centers, acc, width=width, color="tab:blue", edgecolor="black", label="Accuracy $A_m$")
    ax.bar(centers, avg - acc, bottom=acc, width=width, color="tab:red", alpha=0.6,
           edgecolor="tab:red", label="Gap to confidence $C_m$")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy", title=title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(probs: np.ndarray, labels: np.ndarray, class_names, path: str,
                   title: str = "") -> None:
    cm = confusion_matrix(labels, probs.argmax(1), labels=range(len(class_names)))
    with plt.rc_context(_STYLE):
        _draw_confusion(cm, class_names, path, title)


def _draw_confusion(cm, class_names, path, title):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


class Tester:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.work_dir = cfg.work_dir
        self.device = torch.device(cfg.device)
        self.num_bins = cfg.calibrate.num_bins
        self.model = instantiate(cfg.model.object).to(self.device)
        checkpoint = cfg.test.checkpoint or osp.join(self.work_dir, "best.pth")
        load_checkpoint(checkpoint, self.model, self.device)
        self.test_loader = instantiate(cfg.data.object.test)
        self.class_names = list(self.test_loader.dataset.classes)
        self.friction = FrictionModel(OmegaConf.to_container(cfg.friction), self.class_names)

    @torch.no_grad()
    def collect(self, loader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        logits, labels = [], []
        for inputs, targets in loader:
            logits.append(self.model(inputs.to(self.device, non_blocking=True)).float().cpu())
            labels.append(targets)
        return torch.cat(logits), torch.cat(labels)

    def evaluate(self, logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> Dict:
        scaled = logits / temperature
        evaluator = CalibrateEvaluator(len(self.class_names), self.num_bins, device="cpu")
        evaluator.update(scaled, labels)
        calib, _ = evaluator.mean_score()
        probs, y = F.softmax(scaled, 1).numpy(), labels.numpy()
        moments = self.friction.moments(probs)
        result = {
            "T": temperature,
            "acc": 100.0 * float((probs.argmax(1) == y).mean()),
            "nll": calib["nll"],
            "ece": 100.0 * calib["ece"],
            "aece": 100.0 * calib["aece"],
            "cece": 100.0 * calib["cece"],
            "mu_hat": float(moments["mu_hat"].mean()),
            "var_mu": float(moments["var_total"].mean()),
            "var_mu_physical": float(moments["var_physical"].mean()),
            "var_mu_classification": float(moments["var_classification"].mean()),
        }
        mu_ref = self.friction.reference(y)
        for v in self.cfg.braking.speeds:
            mabe, rbe = braking_errors(moments["mu_hat"], mu_ref, v, self.cfg.braking.g)
            result["MABE@{:g}".format(v)], result["RBE@{:g}".format(v)] = mabe, rbe
        return result, probs, y

    def run(self) -> Dict:
        test_logits, test_labels = self.collect(self.test_loader)
        _, val_loader = instantiate(self.cfg.data.object.trainval)
        temperature = fit_temperature(*self.collect(val_loader), num_bins=self.num_bins)
        logger.info("Optimal temperature on the validation set: T* = {:.1f}".format(temperature))
        np.savez(osp.join(self.work_dir, "test_logits.npz"),
                 logits=test_logits.numpy(), labels=test_labels.numpy(), T=temperature)

        results = {}
        name = "{}-{}".format(self.cfg.model.name, self.cfg.loss.name)
        for tag, t in (("", 1.0), ("+TS", temperature)):
            res, probs, y = self.evaluate(test_logits, test_labels, t)
            results[name + tag] = res
            stem = osp.join(self.work_dir, (name + tag).replace("+", "_"))
            plot_reliability(probs, y, stem + "_reliability.png", self.num_bins, name + tag)
            plot_confusion(probs, y, self.class_names, stem + "_confusion.png", name + tag)

        keys = list(next(iter(results.values())).keys())
        table = [["model"] + keys] + [
            [k] + ["{:.4f}".format(v[c]) for c in keys] for k, v in results.items()
        ]
        logger.info("\n" + AsciiTable(table).table)
        with open(osp.join(self.work_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return results
