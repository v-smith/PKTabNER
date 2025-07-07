import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import OneHotEncoder

from pktabner.ctc.ctc_classifier import join_list_elements, ensure_numeric_pair
from pktabner.utils import read_jsonl


def main(
        path_to_config: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/configs/xgb_ctc_config.json",
                                            help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        train_data_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/annotated_ready/ctc_train.jsonl",
                                             help="Path to training data file, expects .pkl file."),
        val_data_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/annotated_ready/ctc_val.jsonl",
                                           help="Path to validation data file, expects .pkl file."),
):
    # ============= Read in config and get args ================= #
    with open(path_to_config, 'r') as file:
        args = json.load(file)
    label2id = {0: 1, 3: 0}
    id2label = {1: "PK", 0: "Other"}

    # ============= Read in and prepare data ================= #
    raw_train_data = list(read_jsonl(train_data_path))
    raw_val_data = list(read_jsonl(val_data_path))
    train_df = pd.DataFrame(raw_train_data)
    val_df = pd.DataFrame(raw_val_data)

    print("=============Dataset Statistics==============\n")
    print("Training Data:")
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
    counter = Counter(y_train)
    pos_count = counter[1]
    neg_count = counter[0]
    scale_pos_weight = neg_count / pos_count
    y_val = [label2id[label] for label in val_df["label"]]

    # ============= Prep Features ================= #

    # --- Cell text Feature ---
    tfidf_cell_text_vec = TfidfVectorizer(max_features=args.get("bow_max_features"))  # Use .get() for safer access
    X_cell_text_train = tfidf_cell_text_vec.fit_transform(train_df['cell_text'])
    # TRANSFORM validation data using the vectorizer FITTED ON TRAINING DATA
    X_cell_text_val = tfidf_cell_text_vec.transform(val_df['cell_text'])

    # --- Row header + Col header + Parent Cell Text feature ---
    # Apply join_list_elements to both train and val dataframes
    train_df['row_headers_flat'] = train_df['row_headers'].apply(join_list_elements)
    train_df['col_headers_flat'] = train_df['col_headers'].apply(join_list_elements)
    train_df['effective_parent_clean'] = train_df['effective_parent'].fillna('').astype(str)

    val_df['row_headers_flat'] = val_df['row_headers'].apply(join_list_elements)
    val_df['col_headers_flat'] = val_df['col_headers'].apply(join_list_elements)
    val_df['effective_parent_clean'] = val_df['effective_parent'].fillna('').astype(str)

    train_df['combined_headers'] = (
            train_df['row_headers_flat'] + " " +
            train_df['col_headers_flat'] + " " +
            train_df['effective_parent_clean']
    )
    val_df['combined_headers'] = (
            val_df['row_headers_flat'] + " " +
            val_df['col_headers_flat'] + " " +
            val_df['effective_parent_clean']
    )

    train_df['combined_headers'] = (
        train_df['combined_headers']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    val_df['combined_headers'] = (
        val_df['combined_headers']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    tfidf_header_vec = TfidfVectorizer(max_features=args.get("bow_max_features"))
    X_header_text_train = tfidf_header_vec.fit_transform(train_df['combined_headers'])
    # TRANSFORM validation data using the vectorizer FITTED ON TRAINING DATA
    X_header_text_val = tfidf_header_vec.transform(val_df['combined_headers'])

    # Combine all text features
    X_text_features_train = hstack([X_cell_text_train, X_header_text_train])
    X_text_features_val = hstack([X_cell_text_val, X_header_text_val])

    # --- CELL TYPES (Categorical - OneHotEncoding) ---
    train_df['col_type_imputed'] = train_df['col_type'].fillna('missing_col_type').astype('category')
    val_df['col_type_imputed'] = val_df['col_type'].fillna('missing_col_type').astype('category')
    ohe_coltypes = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_coltype_train = ohe_coltypes.fit_transform(train_df[['col_type_imputed']])
    X_coltype_val = ohe_coltypes.transform(val_df[['col_type_imputed']])

    train_df['row_type_imputed'] = train_df['row_type'].fillna('missing_row_type').astype('category')
    val_df['row_type_imputed'] = val_df['row_type'].fillna('missing_row_type').astype('category')
    ohe_rowtypes = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_rowtype_train = ohe_rowtypes.fit_transform(train_df[['row_type_imputed']])
    X_rowtype_val = ohe_rowtypes.transform(val_df[['row_type_imputed']])

    # --- Cell indexes (Numerical) ---
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

    numerical_cols_to_use = ['pos_row_idx', 'pos_col_idx', 'rev_pos_row_idx', 'rev_pos_col_idx']

    # Convert to CSR matrix, explicitly ensuring float32 type
    X_numerical_train = csr_matrix(train_df[numerical_cols_to_use].values.astype(np.float32))
    X_numerical_val = csr_matrix(val_df[numerical_cols_to_use].values.astype(np.float32))

    # --- Final Combined Features ---
    X_train = hstack([
        X_text_features_train,
        X_coltype_train,
        X_rowtype_train,
        X_numerical_train
    ])

    X_val = hstack([
        X_text_features_val,
        X_coltype_val,
        X_rowtype_val,
        X_numerical_val
    ])

    print("\nFinal Feature Matrix Shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print("===========================\n")

    # ============= Define the Classifier ================= #

    clf = xgb.XGBClassifier(
        n_estimators=300,
        objective="binary:logistic",
        nthread=4,
        seed=args["seed"],
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    # ============= Define the Hyperparameters ================= #
    params = {
        "xgbclassifier__max_depth": range(2, 10, 2),
        "xgbclassifier__min_child_weight": range(1, 6, 2),
        "xgbclassifier__gamma": [i / 10.0 for i in range(0, 5)],
        "xgbclassifier__subsample": [i / 10.0 for i in range(5, 10)],
        "xgbclassifier__colsample_bytree": [i / 10.0 for i in range(3, 10)],
        'xgbclassifier__reg_alpha': [0, 1e-5, 1e-2, 0.1, 1],
        'xgbclassifier__reg_lambda': [0, 1e-5, 1e-2, 0.1, 1],
    }

    # ============= Perform Grid Search ================= #
    gsearch = RandomizedSearchCV(clf, params, scoring='f1', cv=5, verbose=1)
    gsearch.fit(X_train, y_train)

    print('\n Best estimator:')
    print(gsearch.best_estimator_)
    print('\n Best hyperparameters:')
    print(gsearch.best_params_)

    a = 1


if __name__ == "__main__":
    typer.run(main)