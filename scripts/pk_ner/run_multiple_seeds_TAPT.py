import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from nervaluate import Evaluator
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,
                          DataCollatorForTokenClassification, EarlyStoppingCallback, set_seed)

from pktabner.ner.bert_ner import char_spans_to_token_labels, iob_to_char_spans
from pktabner.utils import read_jsonl


def evaluate_with_nervaluate(trainer, val_dataset, id2label, raw_val_data):
    predictions, labels, _ = trainer.predict(val_dataset)
    preds_argmax = np.argmax(predictions, axis=2)

    all_true_spans = []
    all_pred_spans = []

    for i in range(len(val_dataset)):
        offset_mapping = val_dataset[i]["offset_mapping"]

        true_spans = iob_to_char_spans(labels[i], offset_mapping, id2label)
        pred_spans = iob_to_char_spans(preds_argmax[i], offset_mapping, id2label)

        all_true_spans.append(true_spans)
        all_pred_spans.append(pred_spans)

    evaluator = Evaluator(all_true_spans, all_pred_spans, tags=["PK"])
    _, results_per_tag, _, _ = evaluator.evaluate()
    return results_per_tag


def compute_metrics_fn(id2label):
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        labels = p.label_ids

        true_labels = [
            [id2label[l] for l in label if l != -100]
            for label in labels
        ]

        true_preds = [
            [id2label[pred] for pred, l in zip(pred_seq, label) if l != -100]
            for pred_seq, label in zip(preds, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds)
        }
    return compute_metrics

def run_experiment(seed, config, raw_train_data_2, raw_val_data, aggregated_scores):
    print(f"\n=== Running TAPT experiment with seed {seed} ===\n")
    set_seed(seed)

    label_list = ["O", "B-PK", "I-PK"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    # --- Load external task-adaptive training data ---
    raw_train_data_1 = list(read_jsonl(config["train_path_1"]))
    for ex in raw_train_data_1 + raw_train_data_2:
        ex.pop('_input_hash', None)
        ex.pop('metadata', None)
        if "pmid" in ex:
            ex["pmid"] = str(ex["pmid"])
        if "_task_hash" in ex:
            ex["_task_hash"] = str(ex["_task_hash"])

    tokenizer = AutoTokenizer.from_pretrained(config["pretrained"])

    # === Stage 1: Task-Adaptive Fine-tuning ===
    dataset_1 = Dataset.from_list(raw_train_data_1)
    dataset_1 = dataset_1.map(lambda x: char_spans_to_token_labels(x, tokenizer, label2id), batched=False)

    model = AutoModelForTokenClassification.from_pretrained(config["pretrained"], num_labels=len(label_list))
    model.config.label2id = label2id
    model.config.id2label = id2label

    training_args_1 = TrainingArguments(
        output_dir=f"./tmp_seed_{seed}_stage1",
        evaluation_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config.get("tapt_epochs", 1),  # You can set this separately in config
        weight_decay=config["weight_decay"],
    )

    trainer_1 = Trainer(
        model=model,
        args=training_args_1,
        train_dataset=dataset_1,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    print("Stage 1: TAPT on external data")
    trainer_1.train()

    # === Stage 2: Fine-tuning on target data ===
    dataset_2 = Dataset.from_list(raw_train_data_2)
    val_dataset = Dataset.from_list(raw_val_data)
    dataset_2 = dataset_2.map(lambda x: char_spans_to_token_labels(x, tokenizer, label2id), batched=False)
    val_dataset = val_dataset.map(lambda x: char_spans_to_token_labels(x, tokenizer, label2id), batched=False)

    training_args_2 = TrainingArguments(
        output_dir=f"./tmp_seed_{seed}_stage2",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="no",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
    )

    trainer_2 = Trainer(
        model=model,
        args=training_args_2,
        train_dataset=dataset_2,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics_fn(id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Stage 2: Fine-tuning on PK tables")
    trainer_2.train()
    trainer_2.evaluate()

    # === Evaluation ===
    nerva_scores = evaluate_with_nervaluate(trainer_2, val_dataset, id2label, raw_val_data)
    for match_type in ["partial", "strict"]:
        metrics = nerva_scores["PK"][match_type]
        for metric in ["precision", "recall", "f1"]:
            aggregated_scores[match_type][metric].append(metrics[metric])

    print(f"Seed {seed} results: {nerva_scores}")
    return nerva_scores

def main():
    config_path = Path("/home/vsmith/PycharmProjects/PKTabNER_Draft/configs/bert_ner_config_tapt.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Read in both external and main training datasets
    raw_train_data_1 = list(read_jsonl(config["train_path_1"]))  # TAPT data
    raw_train_data_2 = list(read_jsonl(config["train_path_2"]))  # PK table NER data
    raw_val_data = list(read_jsonl(config["val_path"]))

    seeds = [42, 123, 456, 2025, 999]

    aggregated_scores = {
        "partial": defaultdict(list),
        "strict": defaultdict(list)
    }

    for seed in seeds:
        run_experiment(seed, config, raw_train_data_2, raw_val_data, aggregated_scores)

    # === Summary ===
    print("\n===== Nervaluate Summary Over Seeds =====")
    for match_type in ["partial", "strict"]:
        print(f"\n-- {match_type.upper()} MATCH --")
        for metric in ["precision", "recall", "f1"]:
            scores = aggregated_scores[match_type][metric]
            mean = np.mean(scores) * 100
            std = np.std(scores) * 100
            print(f"{metric.capitalize()}: {mean:.2f}% ± {std:.2f}%")


if __name__ == "__main__":
    main()
