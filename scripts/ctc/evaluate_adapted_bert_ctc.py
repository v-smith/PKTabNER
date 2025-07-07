import json
from pathlib import Path

import pandas as pd
import torch
import typer
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from pktabner.ctc.bert_classifier import SciBertWithAdditionalFeatures, CellDataset
from pktabner.ctc.ctc_classifier import join_list_elements, ensure_numeric_pair
from pktabner.evaluation import print_ctc_model_scores
from pktabner.utils import read_jsonl


def main(
        path_to_config: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/configs/bert_ctc_config.json",
                                            help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        test_data_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/annotated_ready/ctc_test.jsonl",
                                             help="Path to training data file, expects .pkl file."),
        model_save_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/trained_models/ctc/bert/pubmedbert_20epoch_earlystop_mindelta_best_model.pt",
                                            help="Directory to save trained model to."),
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
    raw_test_data = list(read_jsonl(test_data_path))
    if debug:
        raw_test_data = raw_test_data[:10]
    test_df = pd.DataFrame(raw_test_data)

    print("=============Dataset Statistics==============\n")
    print(test_df["label"].value_counts())

    # ============= Set device ================= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print("===========================\n")

    # ============= Prep Labels ================= #
    y_train = [label2id[label] for label in test_df["label"]]

    # ============= Prep Features ================= #
    test_df['row_headers_flat'] = test_df['row_headers'].apply(join_list_elements)
    test_df['col_headers_flat'] = test_df['col_headers'].apply(join_list_elements)
    test_df['effective_parent_clean'] = test_df['effective_parent'].fillna('').astype(str)

    # Apply the helper function to create temporary columns with numeric pairs
    test_df['_pos_pair'] = test_df['pos'].apply(ensure_numeric_pair)
    test_df['_rev_pos_pair'] = test_df['rev_pos'].apply(ensure_numeric_pair)

    # Expand 'pos' tuples into two new columns for test_df
    test_df['pos_row_idx'] = test_df['_pos_pair'].apply(lambda x: x[0])
    test_df['pos_col_idx'] = test_df['_pos_pair'].apply(lambda x: x[1])
    test_df['rev_pos_row_idx'] = test_df['_rev_pos_pair'].apply(lambda x: x[0])
    test_df['rev_pos_col_idx'] = test_df['_rev_pos_pair'].apply(lambda x: x[1])

    test_df = test_df.drop(columns=['_pos_pair', '_rev_pos_pair'])
    # ============= Prep Tokens & Dataset ================= #
    SPECIAL_TOKENS = {
        "additional_special_tokens": ["<CELL>", "<ROW_HEADERS>", "<COL_HEADERS>", "<POS>", "<RPOS>", "<COL_SEP>", "<ROW_SEP>"]
    }
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    test_dataset = CellDataset(test_df, tokenizer, y_train, config["input_cols"])

    test_loader = DataLoader(test_dataset, batch_size=config.get("batch_size", 8), shuffle=True)

    # ============= Load Model ================= #
    print("Loading trained model...")
    model = SciBertWithAdditionalFeatures(config)
    model.text_encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()

    # ============= Run Inference ================= #
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"].to(dtype=torch.float),
                input_type_ids=batch["input_type_ids"],
                labels=None  # No need for labels in inference
            )

            sequence_output = outputs.last_hidden_state
            pooled_output = torch.mean(sequence_output, dim=1)
            pooled_output = model.dropout(pooled_output)
            logits = model.classifier(pooled_output)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    # ============= Evaluate ================= #
    print("\n===== Classification Report =====")
    print_ctc_model_scores(y_labels=all_labels, y_preds=all_preds, id2label=id2label, condition_name="Best Classifier")

    print("===== Confusion Matrix =====")
    print(confusion_matrix(all_labels, all_preds))
    a = 1


if __name__ == "__main__":
    typer.run(main)