"""Render the controlled comparisons exclusively from saved experiment results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import write_atomic
from src.experiment import read_json
from src.plots import save_figure


LABELS = {"baseline": "Исходный", "interaction": "+ взаимодействие",
          "threshold": "+ подбор порога", "full": "Оба изменения"}
COMPARISONS = {"full_minus_baseline": "Полный − исходный",
               "full_minus_interaction": "Вклад подбора порога",
               "full_minus_threshold": "Вклад взаимодействия"}


def interpretation(row: dict) -> str:
    if row["lower"] > 0:
        return "Положительная разность; условный интервал выше нуля."
    if row["upper"] < 0:
        return "Отрицательная разность; условный интервал ниже нуля."
    return "Интервал включает ноль; убедительного различия на этом тесте не установлено."


def render_report(directory: Path, manifest: dict) -> list[Path]:
    training = read_json(directory / "training.json")
    evaluation = read_json(directory / "evaluation.json")
    metrics = evaluation["metrics"]
    comparisons = evaluation["comparisons"]
    paths = []

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained", sharey=True)
    for axis, name, title in zip(axes, ("original", "interaction"),
                                 ("Исходные признаки", "С взаимодействием возраста и пола")):
        table = pd.read_csv(directory / name / "threshold_candidates.csv")
        selected = training["thresholds"][name]
        axis.step(table.threshold, table.mean_balanced_accuracy, where="post", color="#2874A6")
        axis.axvline(0.5, color="#777777", linestyle=":", label="Исходный порог 0.5")
        axis.axvline(selected, color="#C46A16", linestyle="--", label=f"Выбранный: {selected:.4f}")
        axis.set(xlim=(0, 1), xlabel="Общий порог", title=title)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    axes[0].set_ylabel("Средняя сбалансированная точность")
    figure.suptitle("Подбор на пяти внутренних проверочных частях", fontsize=14)
    path = directory / "figures/thresholds.png"
    save_figure(figure, path)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for position, row in enumerate(comparisons):
        delta, lower, upper = [100 * row[k] for k in ("delta_balanced_accuracy", "lower", "upper")]
        axis.hlines(position, lower, upper, color="#2874A6", linewidth=3)
        axis.plot(delta, position, "o", color="#C46A16", markersize=8)
        axis.annotate(f"{delta:+.2f} [{lower:+.2f}; {upper:+.2f}]", (delta, position),
                      xytext=(0, 13), textcoords="offset points", ha="center", fontsize=10)
    axis.axvline(0, color="#777777", linestyle=":")
    axis.set(yticks=range(3), yticklabels=[COMPARISONS[r["comparison"]] for r in comparisons],
             ylim=(-0.55, 2.6), xlabel="Разность сбалансированной точности, процентные пункты",
             title="Три сравнения на итоговом тесте\nОтдельные 95%-е интервалы парного группового бутстрапа")
    axis.invert_yaxis()
    axis.margins(x=0.2)
    axis.grid(axis="x", alpha=0.2)
    path = directory / "figures/comparisons.png"
    save_figure(figure, path)
    paths.append(path)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    largest = max(np.max(m["confusion_matrix"]) for m in metrics.values())
    for axis, (key, title) in zip(axes.flat, LABELS.items()):
        matrix = np.array(metrics[key]["confusion_matrix"])
        picture = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=largest)
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center", fontsize=18,
                      color="white" if value > largest / 2 else "black")
        threshold = training["validation"][key]["threshold"]
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Погиб", "Выжил"],
                 yticklabels=["Погиб", "Выжил"], xlabel="Предсказанный класс",
                 ylabel="Истинный класс", title=f"{title}; порог {threshold:.4f}")
    figure.colorbar(picture, ax=axes.ravel().tolist(), shrink=0.8, label="Число пассажиров")
    figure.suptitle("Матрицы ошибок: одни и те же 179 пассажиров", fontsize=14)
    path = directory / "figures/confusion_matrices.png"
    save_figure(figure, path)
    paths.append(path)

    lines = [
        "# Titanic: взаимодействие возраста и пола и подбор порога", "",
        f"Серия: `{manifest['series_id']}`. Данные и версии: [series.json](series.json).", "",
        "## Гипотезы и контролируемые условия", "",
        "1. Фиксированный порог 0.5 не выбирался под сбалансированную точность. "
        "Подбор одного общего порога может улучшить баланс полноты двух классов.",
        "2. Дополнительный признак `Age * male` может учесть различную связь возраста с выживанием "
        "у мужчин и женщин. Исходный возраст сохраняется; это дополнительный коэффициент, "
        "а не замена возраста женщин нулями. До масштабирования возрастной вклад в линейной "
        "части равен `a * Age + c * Age * male`.", "",
        "Наблюдение исходной модели — только 1 распознанный выживший мужчина из 18 — "
        "мотивировало эти проверки, но само по себе не объясняет причины ошибок.", "",
        "Во всех вариантах используется логистическая регрессия с `C=1`, `class_weight=None`, "
        "L2-регуляризацией и прежними настройками решателя. Нового поиска этих параметров нет. "
        "Остальные семь признаков и их обработка одинаковы. Для взаимодействия пропущенный возраст "
        "заполняется медианой обучающей части до умножения; масштабирование также обучается только на ней.", "",
        "Сохранены точные разбиения исходного эксперимента: 712 пассажиров для разработки, "
        "179 для итогового теста. Все пассажиры с общим билетом остаются вместе. "
        "Пять внутренних проверочных частей: 144, 142, 143, 142 и 141 пассажир; параметр случайности — 42.", "",
        "## Подбор порога и защита от утечек", "",
        "Для каждого набора признаков пять моделей предсказывают вероятность только для своей "
        "отложенной части. Каждый из 712 пассажиров получает ровно одно такое предсказание. "
        "Подготовка признаков обучается внутри соответствующей обучающей части.", "",
        "Кандидаты — уникальные неокруглённые проверочные вероятности плюс 0, 0.5 и 1. "
        "Для каждого общего порога считается обычное среднее пяти значений сбалансированной точности, "
        "без взвешивания частей. Пять оптимальных порогов не усредняются. "
        "Метка 1 получается только при `probability > threshold`. Среди равных максимумов "
        "выбирается ближайший к 0.5 порог, затем меньший. Наборы признаков имеют собственные пороги.", "",
        "После выбора каждая итоговая модель заново обучается на всех 712 пассажирах. "
        "Всего выполнено 12 обучений: по пять внутренних и одному итоговому для двух наборов признаков. "
        "Варианты с одним набором признаков используют одну модель и одинаковые вероятности. "
        "Сравнение на тесте выполняется после фиксации порогов. Метрики подбора отделены от итоговых.", "",
        "## Результаты четырёх вариантов", "",
        "Основная метрика — сбалансированная точность: среднее полноты классов погибших и выживших. "
        "Значения в таблице округлены только для чтения; полная точность — в [experiments.csv](experiments.csv).", "",
        "| Вариант | Порог | Среднее пяти проверок | Тест | Полнота погибших | Полнота выживших |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, title in LABELS.items():
        val = training["validation"][key]
        row = metrics[key]
        lines.append(f"| {title} | {val['threshold']:.6f} | {val['summary']['balanced_accuracy_mean']:.6f} "
                     f"| {row['balanced_accuracy']:.6f} | {row['recall_0']:.6f} | {row['recall_1']:.6f} |")
    lines += ["", "Исходный вариант воспроизвёл файл предсказаний ДЗ1 побайтно, включая порядок "
              "пассажиров, вероятности и метки. Рост внутреннего максимума после подбора порога "
              "не является независимым доказательством улучшения.", "",
              "![Подбор порогов](figures/thresholds.png)", "", "## Общий эффект и две абляции", "",
              "В абляции порога сравниваются одна модель и одинаковые вероятности, меняется только порог. "
              "В абляции признака модель без взаимодействия обучается отдельно, "
              "а её собственный порог выбирается тем же способом. Полная версия означает наличие "
              "двух компонентов и не обязательно является лучшей.", "",
              "| Сравнение | Разность, п.п. | 95%-й интервал, п.п. |",
              "| --- | ---: | ---: |"]
    for row in comparisons:
        lines.append(f"| {COMPARISONS[row['comparison']]} | {100 * row['delta_balanced_accuracy']:+.3f} "
                     f"| [{100 * row['lower']:+.3f}; {100 * row['upper']:+.3f}] |")
    lines += ["", "П.п. — процентные пункты. Интервалы получены парным бутстрапом 136 билетных групп: "
              "2000 повторов, выбор с возвращением, параметр случайности 42. "
              "В каждом повторе все варианты используют одни и те же выбранные группы со всеми "
              "пассажирами и кратностью выбора. Модели и пороги фиксированы. "
              "Границы — квантили 0.025 и 0.975 разностей с линейной интерполяцией. "
              "Неопределённых повторов не было; такой повтор прерывает расчёт, а не удаляется.", "",
              "![Разности с интервалами](figures/comparisons.png)", "", "## Выводы", ""]
    for row in comparisons:
        lines.append(f"- **{COMPARISONS[row['comparison']]}:** "
                     f"{100 * row['delta_balanced_accuracy']:+.3f} процентного пункта. {interpretation(row)}")
    lines.append("")
    for key, label in (("interaction", "Добавление взаимодействия при пороге 0.5"),
                       ("threshold", "Подбор порога на исходных признаках")):
        current, original = metrics[key], metrics["baseline"]
        delta = 100 * (current["balanced_accuracy"] - original["balanced_accuracy"])
        lines.append(
            f"{label}: наблюдаемая разность {delta:+.3f} процентного пункта. "
            f"Полнота погибших изменилась с {original['recall_0']:.4f} до {current['recall_0']:.4f}, "
            f"выживших — с {original['recall_1']:.4f} до {current['recall_1']:.4f}. "
            "Это описательное сравнение; отдельный интервал для этой пары не рассчитывался."
        )
        lines.append("")
    for key, label in (("interaction", "Подбор порога при наличии взаимодействия"),
                       ("threshold", "Добавление взаимодействия при сохранении процедуры подбора")):
        before, after = metrics[key]["confusion_matrix"], metrics["full"]["confusion_matrix"]
        lines.append(
            f"{label}: число пропущенных выживших {before[1][0]} → {after[1][0]}, "
            f"ложных прогнозов выживания {before[0][1]} → {after[0][1]}. "
            "Эти числа показывают изменение каждого типа ошибок отдельно."
        )
        lines.append("")
    lines += ["", "Матрицы ниже показывают, какие ошибки изменились. "
              "Выигрыш в полноте одного класса может сопровождаться потерей в другом. "
              "Связь возраста с полом здесь предсказательная; причинных выводов из коэффициентов нет.", "",
              "![Матрицы ошибок](figures/confusion_matrices.png)", "", "## Ограничения", "",
              "- Итоговый тест уже изучался в ДЗ1 и участвовал в формировании гипотез. "
              "Это прозрачное повторное сравнение, но не новая полностью независимая проверка.",
              "- Лучшие настройки модели ранее выбраны на данных разработки; они фиксированы "
              "и не переоптимизируются для расширенного набора признаков.",
              "- Интервалы условны при фиксированных моделях и порогах. Они не включают разброс "
              "обучения и подбора и являются тремя отдельными интервалами, а не совместной "
              "95%-й гарантией. Наличие нуля в интервале не доказывает равенство.",
              "- Небольшой исторический набор и билетные группы ограничивают переносимость выводов.", "",
              "## История и воспроизведение", "",
              "Полный запуск: `bash scripts/run_hw2.sh`. Новая попытка получает отдельный каталог "
              "и четыре записи в локальном MLflow. База SQLite и файлы находятся в `artifacts/mlflow/`; "
              "история серий — в `artifacts/hw2/runs/`. Интерфейс: "
              "`uv run --locked python -m src.hw2 ui` (http://127.0.0.1:8080).", "",
              "В каждой записи сохранены гипотеза, конфигурация, версии, контрольные суммы, "
              "раздельные внутренние и тестовые метрики, таблицы порогов, предсказания и рисунки. "
              "Этапы дополняют те же четыре записи. Неудачные этапы отмечаются отдельно. "
              "Текущая выгрузка читается без запуска MLflow.", "",
              "Воспроизводимость и проверки: [verification.json](verification.json). "
              "Успех запуска и совпадение повторов следует проверять по этому файлу, "
              "а не предполагать по наличию отчёта.", "",
              "Источники: [условие ДЗ2](https://github.com/PikaChuChuMF/RnD-ML-Course-2026-2/blob/main/HW_2/HomeWork_2.md), "
              "[Titanic](https://www.kaggle.com/competitions/titanic), "
              "[MLflow](https://mlflow.org/docs/latest/ml/tracking/).", ""]
    write_atomic(directory / "report.md", "\n".join(lines).encode())
    return paths
