import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
import xgboost as xgb
from matplotlib import pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
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

    # --- TABLE REGION (Categorical) ---
    # Convert to category BEFORE OneHotEncoding (good practice)
    train_df['region'] = train_df['region'].astype('category')
    val_df['region'] = val_df['region'].astype('category')  # Also convert val_df

    ohe_regions = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_region_train = ohe_regions.fit_transform(train_df[['region']])
    # TRANSFORM validation data using the encoder FITTED ON TRAINING DATA
    X_region_val = ohe_regions.transform(val_df[['region']])

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
        #X_region_train,
        X_coltype_train,
        X_rowtype_train,
        X_numerical_train
    ])

    X_val = hstack([
        X_text_features_val,
        #X_region_val,
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
        learning_rate=args["classifier_lr"],
        max_depth=args["classifier_max_depth"],
        min_child_weight=args["classifier_min_child_weight"],
        gamma=args["classifier_gamma"],
        subsample=args["classifier_subsample"],
        colsample_bytree=args["classifier_colsample_bytree"],
        n_estimators=args["classifier_n_estimators"],
        objective="binary:logistic",
        nthread=4,
        seed=args["seed"],
        use_label_encoder=False,
        eval_metric=args["early_stopping_eval_metric"],
        early_stopping_rounds=args["early_stopping_rounds"],
        scale_pos_weight=scale_pos_weight,
    )

    # ============= Fit the Classifier ================= #
    clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True,
            )

    # ============= Validation metrics ================= #
    y_val_pred = clf.predict(X_val)
    print("\n=============Validation Scores=============.\n")
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    print('Confusion Matrix:')
    print(conf_matrix)
    class_report = classification_report(y_val, y_val_pred, target_names=[id2label[0], id2label[1]])
    print('\nClassification Report:')
    print(class_report)
    print("===========================\n")

    # cross validations
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=args["seed"])
    cv_scoring = {'prec_macro': 'precision_macro',
                  'rec_macro': "recall_macro",
                  "f1_macro": "f1_macro",
                  "f1_micro": "f1_micro",
                  "prec_micro": "precision_micro",
                  "rec_micro": "recall_micro",
                  "f1_weighted": "f1_weighted",
                  "prec_weighted": "precision_weighted",
                  "rec_weighted": "recall_weighted"
                  }

    cv_scores = cross_validate(clf, X_train, y_train, cv=cv, scoring=cv_scoring,
                               params={"sample_weight": class_weights})
    formatted_cv_scores = {k: (round(v.mean() * 100, 2), round(v.std() * 100, 2)) for k, v in cv_scores.items()}
    print(formatted_cv_scores)

    # ============= Feature Importance Analysis ================= #
    print("\n=============Macro Feature Importance=============\n")
    importances = clf.feature_importances_

    # Check if importances are all zeros (can happen with very few features or perfect fit)
    if np.all(importances == 0):
        print(
            "All feature importances are zero. Model might not have learned distinct feature contributions or converged perfectly.")
        print("This can happen if feature_importances_ type is 'weight' (default in some XGBoost versions) "
              "and not 'gain' or 'cover'. Ensure importance_type is set correctly if needed.")
    else:
        num_cell_text = tfidf_cell_text_vec.get_feature_names_out().shape[0]
        num_header_text = tfidf_header_vec.get_feature_names_out().shape[0]
        num_region = ohe_regions.get_feature_names_out().shape[0]
        num_coltype_imputed = ohe_coltypes.get_feature_names_out().shape[0]
        num_rowtype_imputed = ohe_rowtypes.get_feature_names_out().shape[0]
        num_numerical = len(
            ['pos_row_idx', 'pos_col_idx', 'rev_pos_row_idx', 'rev_pos_col_idx'])  # Numerical features are constant

        # Verify total feature count matches
        total_features_from_components = (
                num_cell_text + num_header_text + num_region +
                num_coltype_imputed + num_rowtype_imputed + num_numerical
        )

        if total_features_from_components != len(importances):
            print(f"Mismatch in total feature counts for importance aggregation!")
        else:
            # 2. Slice the importances array according to the component sizes
            current_idx = 0
            imp_cell_text = importances[current_idx: current_idx + num_cell_text]
            current_idx += num_cell_text

            imp_header_text = importances[current_idx: current_idx + num_header_text]
            current_idx += num_header_text

            imp_region = importances[current_idx: current_idx + num_region]
            current_idx += num_region

            imp_coltype = importances[current_idx: current_idx + num_coltype_imputed]
            current_idx += num_coltype_imputed

            imp_rowtype = importances[current_idx: current_idx + num_rowtype_imputed]
            current_idx += num_rowtype_imputed

            imp_numerical = importances[current_idx: current_idx + num_numerical]
            # current_idx += num_numerical # No need to update current_idx after the last slice

            # 3. Sum importances for each macro category
            macro_importances = {
                "Cell Text": imp_cell_text.sum(),
                "Headers Text": imp_header_text.sum(),
                "Region": imp_region.sum(),
                "Column Type": imp_coltype.sum(),
                "Row Type": imp_rowtype.sum(),
                "Cell Coordinates": imp_numerical.sum(),
            }

            # 4. Create a pandas Series, sort, and display
            macro_importance_series = pd.Series(macro_importances).sort_values(ascending=False)

            print("Macro-Level Feature Importances (Sum of F-scores):")
            print(macro_importance_series)

            # 5. Visualize Macro Importance's
            plt.figure(figsize=(10, 6))
            sns.barplot(x=macro_importance_series.values, y=macro_importance_series.index, palette='viridis')
            plt.xlabel('Aggregated F-score Importance')
            plt.ylabel('Feature Category')
            plt.tight_layout()
            plt.show()

    print("===========================\n")


if __name__ == "__main__":
    typer.run(main)