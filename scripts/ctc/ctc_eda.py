from pathlib import Path

import pandas as pd
import typer

from pktabner.utils import read_jsonl


def main(
        train_data_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/annotated_ready/ctc_train.jsonl",
                                             help="Path to training data file, expects .pkl file."),
        val_data_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/annotated_ready/ctc_val.jsonl",
                                           help="Path to validation data file, expects .pkl file."),
        test_data_path: Path = typer.Option(
            "/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/annotated_ready/ctc_test.jsonl",
            help="Path to training data file, expects .pkl file."),

):
    # Load data
    raw_train_data = list(read_jsonl(train_data_path))
    raw_val_data = list(read_jsonl(val_data_path))
    raw_test_data = list(read_jsonl(test_data_path))

    train_df = pd.DataFrame(raw_train_data)
    val_df = pd.DataFrame(raw_val_data)
    test_df = pd.DataFrame(raw_test_data)

    # Dataset sizes
    train_size = len(train_df)
    val_size = len(val_df)
    test_size = len(test_df)
    total_size = train_size + val_size + test_size

    # Label counts
    train_counts = train_df["label"].value_counts()
    val_counts = val_df["label"].value_counts()
    test_counts = test_df["label"].value_counts()

    # Combine label counts
    combined_counts = train_counts.add(val_counts, fill_value=0)
    combined_counts = combined_counts.add(test_counts, fill_value=0)

    # Print summary
    print("============= Dataset Statistics =============\n")
    print(f"Training Size: {train_size}")
    print(train_counts, "\n")

    print(f"Validation Size: {val_size}")
    print(val_counts, "\n")

    print(f"Test Size: {test_size}")
    print(test_counts, "\n")

    print("============= Combined Totals =============")
    print(f"Total Size: {total_size}")
    print("Total Label Counts:")
    print(combined_counts.astype(int))  # Convert float back to int for display

if __name__ == "__main__":
    typer.run(main)
a = 1