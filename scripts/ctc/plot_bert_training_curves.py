import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Root directory containing run subfolders
tensorboard_root = "/home/vsmith/PycharmProjects/PKTabNER_Draft/trained_models/ctc/bert/tensorboard"

# Tags to extract
tags = ["F1/val_step", "Loss/val_step"]

run_name_map = {
    "scibert_20epoch_earlystop": "SciBERT",
    "biobert_20epoch_earlystop_mindelta": "BioBERT",
    "pubmedbert_20epoch_earlystop_mindelta": "PubMedBERT"
}

# Apply rolling average (smoothing)
def smooth_curve(df, value_col, window=3):
    df = df.copy()
    df[value_col] = df[value_col].rolling(window=window, min_periods=1).mean()
    return df



# Collect data for all runs
results = {tag: {} for tag in tags}

def load_scalar_df(log_dir, tag, downsample_to=None):
    acc = EventAccumulator(log_dir)
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return None

    events = acc.Scalars(tag)
    df = pd.DataFrame(events)
    df["step"] = df["step"].astype(int)
    df = df.rename(columns={"value": tag})

    if downsample_to is not None:
        # Round steps to nearest multiple of `downsample_to`
        df["step_group"] = (df["step"] // downsample_to) * downsample_to
        df = df.groupby("step_group")[tag].mean().reset_index().rename(columns={"step_group": "step"})

    return df[["step", tag]]



# Iterate through subdirectories
for run_name in os.listdir(tensorboard_root):
    run_path = os.path.join(tensorboard_root, run_name)
    if not os.path.isdir(run_path):
        continue

    # Use downsample_to=300 for all, including scibert
    for tag in tags:
        df = load_scalar_df(run_path, tag, downsample_to=300)
        if df is not None:
            results[tag][run_name] = df

# --- Plot F1/val_step ---
plt.figure(figsize=(10, 6))
for run_name, df in results["F1/val_step"].items():
    #df = df[df["step"] <= 7000]  # cap x-axis
    df = smooth_curve(df, "F1/val_step", window=3)
    label = run_name_map.get(run_name, run_name)  # fallback to original name
    plt.plot(df["step"], df["F1/val_step"], label=label)
plt.xlabel("Training Step")
plt.ylabel("Validation F1 Score")
#plt.title("Validation F1 Score Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot Loss/val_step ---
plt.figure(figsize=(10, 6))
for run_name, df in results["Loss/val_step"].items():
    #df = df[df["step"] <= 7000]
    df = smooth_curve(df, "Loss/val_step", window=3)
    label = run_name_map.get(run_name, run_name)
    plt.plot(df["step"], df["Loss/val_step"], label=label)

plt.xlabel("Training Step")
plt.ylabel("Validation Loss")
#plt.title("Validation Loss Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

a = 1