from pathlib import Path

import numpy as np
import torch
import typer
from datasets import Dataset
from nervaluate import Evaluator
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification

from pktabner.ner.bert_ner import (
    char_spans_to_token_labels,
    iob_to_char_spans,
    print_ner_scores,
    display_ner_correct,
    display_ner_errors
)
from pktabner.utils import read_jsonl


def load_and_prepare_dataset(data_path, tokenizer, label2id):
    raw_data = list(read_jsonl(data_path))
    for ex in raw_data:
        if "pmid" in ex:
            ex["pmid"] = str(ex["pmid"])
        ex.pop('_input_hash', None)
        ex.pop("metadata", None)
        if "_task_hash" in ex:
            ex["_task_hash"] = str(ex["_task_hash"])
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(lambda x: char_spans_to_token_labels(x, tokenizer, label2id), batched=False)
    return dataset, raw_data


def main(test_data_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/ner_annotated_ready/pkner_val.jsonl"),
         model_dir: Path = typer.Option("/home/vsmith/PycharmProjects/PKTabNER_Draft/trained_models/pk_ner/pubmedbert_10epochs_cell-externaldata100_earlystopping_final/checkpoint-717"),
         pretrained: str = typer.Option("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
         ):
    label_list = ["O", "B-PK", "I-PK"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ============= Load tokenizer and model ============= #
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=len(label_list))
    model.to(device)

    # ============= Load and prepare test data ============= #
    test_dataset, raw_test_data = load_and_prepare_dataset(test_data_path, tokenizer, label2id)

    # ============= Evaluate ============= #
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    predictions, labels, _ = trainer.predict(test_dataset)
    preds_argmax = np.argmax(predictions, axis=2)

    all_true_spans = []
    all_pred_spans = []

    for i in range(len(test_dataset)):
        offset_mapping = test_dataset[i]["offset_mapping"]

        true_spans = iob_to_char_spans(labels[i], offset_mapping, id2label)
        pred_spans = iob_to_char_spans(preds_argmax[i], offset_mapping, id2label)

        all_true_spans.append(true_spans)
        all_pred_spans.append(pred_spans)

    # Evaluate with nervaluate
    evaluator = Evaluator(all_true_spans, all_pred_spans, tags=["PK"])
    _, results_per_tag, _, _ = evaluator.evaluate()

    print("\n=== Test Set Evaluation Results ===")
    print_ner_scores(results_per_tag)

    print("\n=== Correct Predictions ===")
    display_ner_correct(
        true_spans=all_true_spans,
        pred_spans=all_pred_spans,
        original_data=raw_test_data,
        display_number=20,
    )

    print("\n=== Errors ===")
    display_ner_errors(
        true_spans=all_true_spans,
        pred_spans=all_pred_spans,
        original_data=raw_test_data
    )

    a = 1


if __name__ == "__main__":
    typer.run(main)
