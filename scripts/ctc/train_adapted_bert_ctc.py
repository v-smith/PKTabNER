import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from sklearn.utils import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_scheduler

from pktabner.ctc.bert_classifier import SciBertWithAdditionalFeatures, CellDataset, \
    train_one_epoch, evaluate
from pktabner.ctc.ctc_classifier import join_list_elements, ensure_numeric_pair, \
    join_list_elements_row_tokens, join_list_elements_col_tokens
from pktabner.utils import read_jsonl


def main(
        path_to_config: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/configs/bert_ctc_config.json",
                                            help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        train_data_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/annotated_ready/ctc_train.jsonl",
                                             help="Path to training data file, expects .pkl file."),
        val_data_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/annotated_ready/ctc_val.jsonl",
                                           help="Path to validation data file, expects .pkl file."),
        model_save_dir: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/trained_models/ctc/bert/",
                                            help="Directory to save trained model to."),
        run_name: str = typer.Option("scibert_20epoch_earlystop_mindelta", help="Name of run name."),
        debug: bool = typer.Option(False, "--debug", help="Debug mode."),
):
    # ============= Read in config and get args ================= #
    with open(path_to_config, 'r') as file:
        config = json.load(file)
    label2id = {0: 1, 3: 0}
    id2label = {1: "PK", 0: "Other"}
    config["input_cols"] = ["<CELL>", "<ROW_HEADERS>", "<COL_HEADERS>", "<POS>", "<RPOS>"]
    config["num_labels"] = len(set(label2id.values()))
    config["num_MLP_layers"] = config.get("num_MLP_layers", 2)

    # ============= Read in and prepare data ================= #
    raw_train_data = list(read_jsonl(train_data_path))
    raw_val_data = list(read_jsonl(val_data_path))
    if debug:
        raw_train_data = raw_train_data[:10]
        raw_val_data = raw_val_data[:10]
    train_df = pd.DataFrame(raw_train_data)
    val_df = pd.DataFrame(raw_val_data)

    print("=============Dataset Statistics==============\n")
    print(train_df["label"].value_counts())
    print("\nValidation Data:")
    print(val_df["label"].value_counts())
    print("===========================\n")

    # ============= Set device ================= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print("===========================\n")

    # ============= Prep Labels ================= #
    y_train = [label2id[label] for label in train_df["label"]]
    y_val = [label2id[label] for label in val_df["label"]]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
    config["class_weight"] = class_weights

    # ============= Prep Features ================= #
    train_df['row_headers_flat'] = train_df['row_headers'].apply(join_list_elements)
    train_df['col_headers_flat'] = train_df['col_headers'].apply(join_list_elements)
    train_df['effective_parent_clean'] = train_df['effective_parent'].fillna('').astype(str)

    val_df['row_headers_flat'] = val_df['row_headers'].apply(join_list_elements_row_tokens).str.replace(r'\s+', ' ', regex=True).str.strip()
    val_df['col_headers_flat'] = val_df['col_headers'].apply(join_list_elements_col_tokens).str.replace(r'\s+', ' ', regex=True).str.strip()

    # Apply the helper function to create temporary columns with numeric pairs
    train_df['_pos_pair'] = train_df['pos'].apply(ensure_numeric_pair)
    train_df['_rev_pos_pair'] = train_df['rev_pos'].apply(ensure_numeric_pair)

    val_df['_pos_pair'] = val_df['pos'].apply(ensure_numeric_pair)
    val_df['_rev_pos_pair'] = val_df['rev_pos'].apply(ensure_numeric_pair)

    # Expand 'pos' tuples into two new columns for train_df
    train_df['pos_row_idx'] = train_df['_pos_pair'].apply(lambda x: x[0])
    train_df['pos_col_idx'] = train_df['_pos_pair'].apply(lambda x: x[1])
    train_df['rev_pos_row_idx'] = train_df['_rev_pos_pair'].apply(lambda x: x[0])
    train_df['rev_pos_col_idx'] = train_df['_rev_pos_pair'].apply(lambda x: x[1])

    val_df['pos_row_idx'] = val_df['_pos_pair'].apply(lambda x: x[0])
    val_df['pos_col_idx'] = val_df['_pos_pair'].apply(lambda x: x[1])
    val_df['rev_pos_row_idx'] = val_df['_rev_pos_pair'].apply(lambda x: x[0])
    val_df['rev_pos_col_idx'] = val_df['_rev_pos_pair'].apply(lambda x: x[1])

    train_df = train_df.drop(columns=['_pos_pair', '_rev_pos_pair'])
    val_df = val_df.drop(columns=['_pos_pair', '_rev_pos_pair'])

    # ============= Prep Tokens & Dataset ================= #
    SPECIAL_TOKENS = {
        "additional_special_tokens": ["<CELL>", "<ROW_HEADERS>", "<COL_HEADERS>", "<POS>", "<RPOS>", "<COL_SEP>", "<ROW_SEP>"]
    }
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    train_dataset = CellDataset(train_df, tokenizer, y_train, config["input_cols"])
    val_dataset = CellDataset(val_df, tokenizer, y_val, config["input_cols"])

    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 8), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 8), shuffle=False)

    # ============= Define the Classifier ================= #
    model = SciBertWithAdditionalFeatures(config)
    model.text_encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.get("learning_rate", 2e-5))

    # ============= Fit & Save the Classifier ================= #
    writer = SummaryWriter(log_dir=str(model_save_dir / "tensorboard" /run_name))

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    num_training_steps = len(train_loader) * config["epochs"]
    num_warmup_steps = int(config.get("warmup_ratio", 0.1) * num_training_steps)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    min_delta = 0.001
    global_step = 0
    best_val_acc = 0
    epochs_no_improve = 0
    patience = config.get("early_stopping_patience", 2)

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}")

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch=epoch,
            writer=writer,
            val_loader=val_loader,
            scaler=scaler,
            eval_every=300,
            global_step_start=global_step
        )

        val_loss, val_f1 = evaluate(model, val_loader, device)

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        if (val_f1 - best_val_acc) > min_delta:
            best_val_acc = val_f1
            torch.save(model.state_dict(), model_save_dir / f"{run_name}_best_model.pt")
            print("Saved new best model!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} consecutive epoch(s).")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best Val F1: {best_val_acc:.4f}")
                break

    writer.close()

    print("Training complete.")
    a = 1


if __name__ == "__main__":
    typer.run(main)