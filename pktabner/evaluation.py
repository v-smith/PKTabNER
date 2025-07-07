from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score,
    confusion_matrix
)


def print_ctc_model_scores(
        y_labels: List[int],
        y_preds: List[int],
        id2label: Dict[int, str],
        condition_name: str
) -> None:
    """
    Prints a classification report for the given test and predicted labels, formatted with percentages
    rounded to two decimal places.

    Parameters:
    - y_test (List[int]): True labels of the test set.
    - y_test_pred (List[int]): Predicted labels from the model.
    - id2label (Dict[int, str]): A dictionary mapping label indices to their string names.
    - condition_name (str): The name of the condition being evaluated.

    Returns:
    - None: Outputs the classification report directly to the console.
    """
    print(f"Classification report (%) to 2 d.p for {condition_name}")
    cr = classification_report(
        y_labels,
        y_preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True
    )
    df = pd.DataFrame(cr).transpose()
    metrics_columns = ['precision', 'recall', 'f1-score']
    df[metrics_columns] = df[metrics_columns] * 100
    df_rounded = df.round(2)
    print(df_rounded)


def evaluate(y_true, y_pred, id_to_label=None, print_results=True):
    """
    Compute micro precision, recall, F1, and accuracy for multiclass predictions.
    """
    if id_to_label:
        y_true = [id_to_label.get(y, y) for y in y_true]
        y_pred = [id_to_label.get(y, y) for y in y_pred]

    metrics = {
        "mic_P": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "mic_R": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "mic_F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "mac_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    metrics = {k: f"{v * 100:.2f}%" for k, v in metrics.items()}

    if print_results:
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
    return metrics


def print_classification_errors(results_dict, id_to_label=None, max_errors=100, y_true_label="y_true", y_pred_label="y_pred", texts_label="texts"):
    """
    Print and return misclassified examples.
    """
    y_true = results_dict[y_true_label]
    y_pred = results_dict[y_pred_label]
    texts = results_dict[texts_label]
    model_name = results_dict.get("model_name", "Model")

    print(f"\n🔍 Misclassifications from {model_name}:\n")
    errors_summary = []

    for i, (true, pred, text) in enumerate(zip(y_true, y_pred, texts)):
        if true != pred:
            true_label = id_to_label.get(true, true) if id_to_label else true
            pred_label = id_to_label.get(pred, pred) if id_to_label else pred

            print(f"🟥 TEXT: {text}")
            print(f"   ✅ True: {true_label}")
            print(f"   ❌ Pred: {pred_label}\n")

            errors_summary.append({"text": text, "true": true, "pred": pred})
            if len(errors_summary) >= max_errors:
                break

    if not errors_summary:
        print("✅ No misclassifications found!")

    return errors_summary


def plot_confusion_matrix(y_true, y_pred, label_mapping=None, normalize=True, figsize=(10, 8), save_path=None):
    """
    Plot a heatmap-style confusion matrix like your reference image.
    """
    label_mapping["Q100"] = "NIL"

    # Optionally remap class labels
    if label_mapping:
        y_true = [label_mapping.get(y, y) for y in y_true]
        y_pred = [label_mapping.get(y, y) for y in y_pred]

    # Determine label set
    labels = sorted(set(y_true) | set(y_pred), key=lambda x: (x == "NIL", x))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true' if normalize else None)

    # Optional: identify missing columns and mask them (make fully white)
    missing_preds = [i for i, col in enumerate((cm == 0).all(axis=0)) if col]
    missing_actuals = [i for i, row in enumerate((cm == 0).all(axis=1)) if row]
    mask = np.zeros_like(cm, dtype=bool)
    mask[:, missing_preds] = True
    mask[missing_actuals, :] = True

    plt.figure(figsize=figsize)
    sns.set_style(style="white")

    ax = sns.heatmap(
        cm,
        mask=mask,
        cmap="rocket_r",
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        vmin=0, vmax=1  # ensure low values show as sand
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
