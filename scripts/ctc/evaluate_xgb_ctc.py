import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from scipy.sparse import csr_matrix, hstack

from pktabner.ctc.ctc_classifier import ensure_numeric_pair, join_list_elements
from pktabner.evaluation import print_ctc_model_scores
from pktabner.utils import read_jsonl


def main(
        path_to_config: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/configs/xgb_ctc_config.json",
                                            help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        test_data_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/annotated_ready/ctc_test.jsonl",
                                             help="Path to training data file, expects .pkl file."),
        path_to_trained_model: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/trained_models/ctc/full_feature_pipeline_and_classifier.pkl",
                                            help="trained model dir"),
):
    # ============= Read in config and get args ================= #
    with open(path_to_config, 'r') as file:
        args = json.load(file)
    label2id = {0: 1, 3: 0}
    id2label = {1: "PK", 0: "Other"}

    # ============= Read in and prepare data ================= #
    raw_test_data = list(read_jsonl(test_data_path))
    test_df = pd.DataFrame(raw_test_data)

    print("=============Dataset Statistics==============\n")
    print("Test Data:")
    print(test_df["label"].value_counts())

    # ============= Set device ================= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print("===========================\n")

    # ============= Load the Classifier ================= #
    print(f"Loading pipeline components from: {path_to_trained_model}")
    with open(path_to_trained_model, 'rb') as f:
        loaded_components = pickle.load(f)

    # Extract individual components
    loaded_tfidf_cell_text_vec = loaded_components['tfidf_cell_text_vec']
    loaded_tfidf_header_vec = loaded_components['tfidf_header_vec']
    loaded_ohe_coltypes = loaded_components['ohe_coltype']
    loaded_ohe_rowtypes = loaded_components['ohe_rowtype']
    classifier = loaded_components['xgb_classifier']


    # ============= Prep Test Features & Labels ================= #
    y_test = [label2id[label] for label in test_df["label"]]

    # 1. Cell text Feature
    X_cell_text_test = loaded_tfidf_cell_text_vec.transform(test_df['cell_text'])

    # 2. Row header + Col header + Parent Cell Text feature
    test_df['row_headers_flat'] = test_df['row_headers'].apply(join_list_elements)
    test_df['col_headers_flat'] = test_df['col_headers'].apply(join_list_elements)
    test_df['effective_parent_clean'] = test_df['effective_parent'].fillna('').astype(str)

    test_df['combined_headers'] = (
            test_df['row_headers_flat'] + " " +
            test_df['col_headers_flat'] + " " +
            test_df['effective_parent_clean']
    )
    test_df['combined_headers'] = (
        test_df['combined_headers']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    X_header_text_test = loaded_tfidf_header_vec.transform(test_df['combined_headers'])

    # Combine all text features
    X_text_features_test = hstack([X_cell_text_test, X_header_text_test])

    # 3. Cell Types (Categorical - OneHotEncoding)
    test_df['col_type_imputed'] = test_df['col_type'].fillna('missing_col_type').astype('category')
    test_df['row_type_imputed'] = test_df['row_type'].fillna('missing_row_type').astype('category')

    X_coltype_test = loaded_ohe_coltypes.transform(test_df[['col_type_imputed']])
    X_rowtype_test = loaded_ohe_rowtypes.transform(test_df[['row_type_imputed']])

    # 4. Cell indexes (Numerical - Tuple expansion)
    test_df['_pos_pair'] = test_df['pos'].apply(ensure_numeric_pair)
    test_df['_rev_pos_pair'] = test_df['rev_pos'].apply(ensure_numeric_pair)

    test_df['pos_row_idx'] = test_df['_pos_pair'].apply(lambda x: x[0])
    test_df['pos_col_idx'] = test_df['_pos_pair'].apply(lambda x: x[1])
    test_df['rev_pos_row_idx'] = test_df['_rev_pos_pair'].apply(lambda x: x[0])
    test_df['rev_pos_col_idx'] = test_df['_rev_pos_pair'].apply(lambda x: x[1])

    test_df = test_df.drop(columns=['_pos_pair', '_rev_pos_pair'])  # Drop temp columns

    numerical_cols_to_use = ['pos_row_idx', 'pos_col_idx', 'rev_pos_row_idx', 'rev_pos_col_idx']
    X_numerical_test = csr_matrix(test_df[numerical_cols_to_use].values.astype(np.float32))

    # 5. Final Combined Features
    X_test = hstack([
        X_text_features_test,
        X_coltype_test,
        X_rowtype_test,
        X_numerical_test
    ])

    print(f"\nFinal X_test shape: {X_test.shape}")
    print("===========================\n")

    # ============= Get Predictions and Scores ================= #
    y_test_pred = classifier.predict(X_test)
    print_ctc_model_scores(y_labels=y_test, y_preds=y_test_pred, id2label=id2label, condition_name="Best Classifier")


if __name__ == "__main__":
    typer.run(main)