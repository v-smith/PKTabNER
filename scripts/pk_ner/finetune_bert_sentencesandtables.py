import json
from pathlib import Path

import numpy as np
import torch
import typer
from datasets import Dataset
from nervaluate import Evaluator
from seqeval.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification

from pktabner.ner.bert_ner import char_spans_to_token_labels, iob_to_char_spans, display_ner_errors, \
    print_ner_scores
from pktabner.utils import read_jsonl


def main(
        path_to_config: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/configs/bert_ner_config.json",
                                            help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        train_data_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/ner_annotated_ready/pkner_train.jsonl",
                                             help="Path to training data file, expects .pkl file."),
        val_data_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/ner_annotated_ready/pkner_val.jsonl",
                                           help="Path to validation data file, expects .pkl file."),
        model_save_dir: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/trained_models/pk_ner/",
                                            help="Directory to save trained model to."),
        debug: bool = typer.Option(False, "--debug", help="Debug mode."),
):
    # ============= Read in config and get args ================= #
    with open(path_to_config, 'r') as file:
        config = json.load(file)
    run_name = config["run_name"]
    logging_dir = "/home/vsmith/PycharmProjects/PKTabNER_Draft/trained_models/pk_ner/" + run_name

    # ============= Read in and prepare data ================= #
    raw_train_data = list(read_jsonl(train_data_path))
    raw_val_data = list(read_jsonl(val_data_path))
    if debug:
        raw_train_data = raw_train_data[:20]
        raw_val_data = raw_val_data[:20]

    # ============= Set device ================= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print("===========================\n")

    # ============= Prep Labels ================= #
    # Extract unique labels from training data
    all_labels = set(span['label'] for example in raw_train_data for span in example['spans'])
    label_list = ["O", "B-PK", "I-PK"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label_list)

    # ============= Prep Tokens ================= #
    train_dataset = Dataset.from_list(raw_train_data)
    val_dataset = Dataset.from_list(raw_val_data)

    tokenizer = AutoTokenizer.from_pretrained(config["pretrained"])
    train_dataset = train_dataset.map(lambda x: char_spans_to_token_labels(x, tokenizer, label2id), batched=False)
    val_dataset = val_dataset.map(lambda x: char_spans_to_token_labels(x, tokenizer, label2id), batched=False)

    # ============= Define the Model ================= #
    model = AutoModelForTokenClassification.from_pretrained(config["pretrained"], num_labels=num_labels)
    model.config.label2id = label2id
    model.config.id2label = id2label

    model_save_dir = str(model_save_dir / run_name)

    # ============= Fit & Save the Model ================= #
    training_args = TrainingArguments(
        output_dir=str(model_save_dir),
        logging_strategy="steps",
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=20,
        logging_dir=logging_dir,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
    )

    def compute_metrics(p):
        predictions, labels = p
        # Get the most likely prediction class for each token
        preds = np.argmax(predictions, axis=2)

        # The 'id2label' map should be available from your main script's scope
        # Remove ignored index (-100) and convert ids to label strings
        true_labels = [
            [id2label[l] for l in label if l != -100]
            for label in labels
        ]

        true_preds = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]

        # Return a dictionary of metrics
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Training complete.")

    # ======== Evaluation with Scoring and Error Visualization ========
    print("--- Starting Evaluation ---")

    # 1. Get model predictions on the validation set
    predictions, labels, _ = trainer.predict(val_dataset)
    preds_argmax = np.argmax(predictions, axis=2)

    # 2. Convert IOB tags to character-level spans for both true and predicted labels
    all_true_spans = []
    all_pred_spans = []

    for i in range(len(val_dataset)):
        offset_mapping = val_dataset[i]["offset_mapping"]

        # Convert true labels to spans
        true_labels_for_instance = labels[i]
        true_spans = iob_to_char_spans(true_labels_for_instance, offset_mapping, id2label)
        all_true_spans.append(true_spans)

        # Convert predicted labels to spans
        pred_labels_for_instance = preds_argmax[i]
        pred_spans = iob_to_char_spans(pred_labels_for_instance, offset_mapping, id2label)
        all_pred_spans.append(pred_spans)

    # 3. Run nervaluate scoring using the default span loader
    evaluator = Evaluator(all_true_spans, all_pred_spans, tags=['PK'])
    _, results_per_tag, _, _ = evaluator.evaluate()

    print("\n--- Evaluation Scores ---")
    print_ner_scores(inp_dict=results_per_tag)

    # 4. Display detailed errors
    display_ner_errors(
        true_spans=all_true_spans,
        pred_spans=all_pred_spans,
        original_data=raw_val_data
    )
    a = 1


if __name__ == "__main__":
    typer.run(main)