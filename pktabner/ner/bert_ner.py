from typing import Dict

import torch


def char_spans_to_token_labels(example, tokenizer, label2id):
    text = example["text"]
    spans = example["spans"]

    # Create a char-level label list
    char_labels = ['O'] * len(text)
    for span in spans:
        start, end, label = span['start'], span['end'], span['label']
        char_labels[start] = f'B-{label}'
        for i in range(start + 1, end):
            char_labels[i] = f'I-{label}'

    # Tokenize with word alignment
    tokenized = tokenizer(text, return_offsets_mapping=True, truncation=True)
    labels = []

    for offset in tokenized['offset_mapping']:
        if offset == (0, 0):
            labels.append(-100)
        else:
            start, end = offset
            if start < len(char_labels):
                label = char_labels[start]
            else:
                label = 'O'
            labels.append(label2id.get(label, label2id['O']))

    tokenized['labels'] = labels
    return tokenized


def iob_to_char_spans(token_labels, offset_mapping, id2label, debug=False):
    """
    Converts token-level IOB predictions to character-level spans.
    Includes a debug mode and filters out invalid zero-width spans.
    """
    if debug:
        print("\n--- Starting Span Conversion ---")

    spans = []
    in_entity = False
    start, end = 0, 0
    current_label = ""

    # (The main loop logic remains exactly the same as before)
    for i in range(len(offset_mapping)):
        label_id = token_labels[i]
        label = id2label.get(label_id, "O")
        offset = offset_mapping[i]

        if offset == (0, 0):
            if in_entity:
                spans.append({"label": current_label, "start": start, "end": end})
                in_entity = False
            continue

        if label.startswith("B-"):
            if in_entity:
                spans.append({"label": current_label, "start": start, "end": end})
            in_entity = True
            current_label = label.split("-")[1]
            start, end = offset
        elif label.startswith("I-") and in_entity:
            if label.split("-")[1] == current_label:
                _, end = offset
            else:
                spans.append({"label": current_label, "start": start, "end": end})
                in_entity = False
        elif label == "O" and in_entity:
            spans.append({"label": current_label, "start": start, "end": end})
            in_entity = False

    if in_entity:
        spans.append({"label": current_label, "start": start, "end": end})

    # --- THE FIX IS HERE ---
    # Filter out any invalid spans where start >= end.
    valid_spans = [span for span in spans if span['start'] < span['end']]

    if debug:
        print(f"--- Result: Found {len(valid_spans)} valid spans: {valid_spans} ---")

    return valid_spans


def get_f1(p, r):
    if p + r == 0.:
        return 0.
    else:
        return (2 * p * r) / (p + r)

def get_metrics(inp_dict):
    p = inp_dict['precision']
    r = inp_dict['recall']
    if "f1" in inp_dict.keys():
        f1 = inp_dict['f1']
    else:
        f1 = get_f1(p=p, r=r)
    return torch.FloatTensor([p]), torch.FloatTensor([r]), torch.FloatTensor([f1])



def print_ner_scores(inp_dict: Dict):
    """

    @param inp_dict: Dictionary with keys corresponding to entity types and subkeys to metrics
    e.g. {'PK': {'ent_type': {..},{'partial': {..},{'strict': {..} }}
    @return: Prints summary of metrics
    """
    for ent_type in inp_dict.keys():
        print(f"====== Stats for entity {ent_type} ======")
        for metric_type in inp_dict[ent_type].keys():
            if metric_type in ['partial', 'strict']:
                print(f" === {metric_type} match: === ")
                precision = inp_dict[ent_type][metric_type]['precision']
                recall = inp_dict[ent_type][metric_type]['recall']
                f1 = inp_dict[ent_type][metric_type]['f1']
                p = round(precision * 100, 2)
                r = round(recall * 100, 2)
                f1 = round(f1 * 100, 2)
                print(f" Precision:\t {p}%")
                print(f" Recall:\t {r}%")
                print(f" F1:\t\t {f1}%")


class AnsiColors:
    """Helper class for terminal colors."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def view_entities(text: str, spans: list[dict]):
    """Creates a color-coded string to view entities in the terminal."""

    # If there are no spans, just return the original text
    if not spans:
        return text

    # Sort spans by start index to handle them in order
    sorted_spans = sorted(spans, key=lambda x: x['start'])

    result = []
    last_idx = 0

    for span in sorted_spans:
        # THE FIX: This line appends the text BEFORE the current span.
        # It must use span['start'], not span['end'].
        result.append(text[last_idx:span['start']])

        # Add the highlighted entity
        result.append(f"{AnsiColors.OKGREEN}")  # Or AnsiColors.FAIL for red
        result.append(text[span['start']:span['end']])
        result.append(f"{AnsiColors.ENDC}")

        last_idx = span['end']

    # Add any remaining text after the last span
    result.append(text[last_idx:])

    return "".join(result)


def display_ner_correct(
        true_spans: list[list[dict]],
        pred_spans: list[list[dict]],
        original_data: list[dict],
        display_number=20,
):
    """
    Finds and prints discrepancies between true and predicted entity spans.
    """
    true_spans = true_spans[:display_number]
    pred_spans = pred_spans[:display_number]
    print("\n" + "=" * 20 + " True Analysis " + "=" * 20)
    true_count = 0
    for i, (true_list, pred_list) in enumerate(zip(true_spans, pred_spans)):
        # Check if the set of predicted spans is different from the true ones
        # We convert to tuples to make them hashable for set comparison
        true_set = set(tuple(sorted(d.items())) for d in true_list)
        pred_set = set(tuple(sorted(d.items())) for d in pred_list)

        if true_set == pred_set:
            true_count += 1
            instance_text = original_data[i]["text"]
            print(f"\n--- True #{true_count} (Example Index: {i}) ---")

            print("TRUE LABELS:")
            print(view_entities(instance_text, true_list))

            print("\nMODEL PREDICTIONS:")
            print(view_entities(instance_text, pred_list))
            print("-" * 50)

    if true_count == 0:
        print("No true!")
    else:
        print(f"\nFound a total of {true_count} correct examples.")


def display_ner_errors(
        true_spans: list[list[dict]],
        pred_spans: list[list[dict]],
        original_data: list[dict],
):
    """
    Finds and prints discrepancies between true and predicted entity spans.
    """
    print("\n" + "=" * 20 + " Error Analysis " + "=" * 20)
    error_count = 0
    for i, (true_list, pred_list) in enumerate(zip(true_spans, pred_spans)):
        # Check if the set of predicted spans is different from the true ones
        # We convert to tuples to make them hashable for set comparison
        true_set = set(tuple(sorted(d.items())) for d in true_list)
        pred_set = set(tuple(sorted(d.items())) for d in pred_list)

        if true_set != pred_set:
            error_count += 1
            instance_text = original_data[i]["text"]
            print(f"\n--- Discrepancy #{error_count} (Example Index: {i}) ---")

            print("TRUE LABELS:")
            print(view_entities(instance_text, true_list))

            print("\nMODEL PREDICTIONS:")
            print(view_entities(instance_text, pred_list))
            print("-" * 50)

    if error_count == 0:
        print("No discrepancies found between predictions and true labels. Congratulations!")
    else:
        print(f"\nFound a total of {error_count} examples with discrepancies.")