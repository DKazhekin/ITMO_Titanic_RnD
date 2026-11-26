"""Render figures from saved predictions and validation scores without fitting."""

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, verify_checksum, write_atomic
from src.experiment import load_run, read_json


def save_figure(figure, path: Path) -> None:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160, facecolor="white")
    write_atomic(path, buffer.getvalue())
    plt.close(figure)


def confusion_figure(metrics: dict):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
    largest = max(np.max(metrics[name]["confusion_matrix"]) for name in ("model", "control"))
    for axis, name, title in zip(axes, ("model", "control"), ("Логистическая регрессия", "Постоянный контроль")):
        matrix = np.array(metrics[name]["confusion_matrix"])
        picture = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=largest)
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center", fontsize=17,
                      color="white" if value > largest / 2 else "black")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Погиб", "Выжил"],
                 yticklabels=["Погиб", "Выжил"], xlabel="Предсказанный класс",
                 ylabel="Истинный класс", title=title)
    figure.colorbar(picture, ax=axes, shrink=0.8, label="Число пассажиров")
    figure.suptitle("Матрицы ошибок на итоговой выборке", fontsize=14)
    return figure


def probability_figure(predictions: pd.DataFrame):
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    bins = np.linspace(0, 1, 21)
    for cls, label, color in ((0, "Погибли", "#2874A6"), (1, "Выжили", "#C46A16")):
        values = predictions.loc[predictions["Survived"] == cls, "probability_model"]
        axis.hist(values, bins=bins, density=True, alpha=0.45, edgecolor=color,
                  color=color, label=f"{label} (n={len(values)})")
    axis.axvline(0.5, color="#333333", linestyle="--", linewidth=1.5, label="Порог 0.5")
    axis.set(xlim=(0, 1), xlabel="Оценка вероятности выживания",
             ylabel="Плотность внутри истинного класса",
             title="Оценки модели на итоговой выборке")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    return figure


def validation_figure(training: dict):
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for weight, label, color in ((None, "Без весов классов", "#2874A6"),
                                 ("balanced", "Обратные частотам веса", "#C46A16")):
        rows = sorted((row for row in training["cv_summary"]
                       if row["model"] == "logistic_regression" and row["class_weight"] == weight),
                      key=lambda row: row["C"])
        if not rows:
            continue
        c = np.array([row["C"] for row in rows])
        mean = np.array([row["mean_balanced_accuracy"] for row in rows])
        std = np.array([row["std_balanced_accuracy"] for row in rows])
        axis.plot(c, mean, "o-", color=color, label=label)
        axis.fill_between(c, mean - std, mean + std, color=color, alpha=0.16)
    control = next(row for row in training["cv_summary"] if row["model"] == "majority")
    axis.axhline(control["mean_balanced_accuracy"], linestyle=":", color="#555555", label="Контроль")
    axis.set(xscale="log", xlabel="C: меньшее значение — сильнее регуляризация",
             ylabel="Средняя balanced accuracy",
             title="Внутренняя проверка по пяти частям\nСреднее ± стандартное отклонение")
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    return figure


def make_plots(artifacts_dir: Path, reports_dir: Path) -> list[Path]:
    run = load_run(artifacts_dir)
    evaluation = read_json(artifacts_dir / "evaluation.json")
    if evaluation["run_id"] != run["run_id"]:
        raise ValueError("Evaluation belongs to a different training run; run evaluate first")
    verify_checksum((artifacts_dir / "predictions.csv").read_bytes(),
                    evaluation["predictions_sha256"], "predictions.csv")
    predictions = pd.read_csv(artifacts_dir / "predictions.csv")
    training = read_json(artifacts_dir / "training.json")
    figures = {
        "confusion_matrix.png": confusion_figure(evaluation["metrics"]),
        "probabilities.png": probability_figure(predictions),
        "validation_curve.png": validation_figure(training),
    }
    paths = []
    for name, figure in figures.items():
        path = reports_dir / "figures" / name
        save_figure(figure, path)
        paths.append(path)
    print(f"Saved {len(paths)} figures from existing results; no models were trained or scored.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts/baseline")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    make_plots(args.artifacts_dir, args.reports_dir)


if __name__ == "__main__":
    main()
