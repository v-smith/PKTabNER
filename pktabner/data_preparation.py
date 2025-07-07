import random
from collections import defaultdict, Counter

import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def build_cell_grid(soup):
    grid = []
    rows = soup.find_all("tr")
    occupied = defaultdict(lambda: defaultdict(bool))

    for row_idx, row in enumerate(rows):
        col_idx = 0
        while len(grid) <= row_idx:
            grid.append([])

        cells = row.find_all(["td", "th"])
        for cell in cells:
            while occupied[row_idx].get(col_idx, False):
                col_idx += 1

            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            # Fill grid only at anchor position
            while len(grid[row_idx]) <= col_idx:
                grid[row_idx].append(None)
            grid[row_idx][col_idx] = cell

            # Mark all spanned positions as occupied
            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied[row_idx + dr][col_idx + dc] = True

            col_idx += colspan

    return grid




def wrap_table_with_caption_footer(table_html: str, caption: str = "", footer: str = "") -> str:
    return f"""
    <div style="font-family: sans-serif; font-size: 14px; line-height: 1.5; margin-bottom: 1em;">
        <div style="margin-bottom: 0.5em;"><strong>Caption:</strong> {caption}</div>
        {table_html}
        <div style="margin-top: 0.5em;"><strong>Footer:</strong> {footer}</div>
    </div>
    """


def filter_overlapping_spans(spans):
    spans = sorted(spans, key=lambda s: (s["start"], -s["end"]))
    filtered = []
    last_end = -1
    for span in spans:
        if span["start"] >= last_end:
            filtered.append(span)
            last_end = span["end"]
    return filtered


def spacy_tokenize(text: str, nlp):
    doc = nlp.make_doc(text)  # use .make_doc to avoid running full pipeline
    return [
        {
            "text": token.text,
            "start": token.idx,
            "end": token.idx + len(token),
            "id": i,
            "ws": token.whitespace_ != ""
        }
        for i, token in enumerate(doc)
    ]

def find_spans(text, compiled_patterns, tokens):
    spans = []
    seen = set()

    for entry in compiled_patterns:
        for match in entry["pattern"].finditer(text):
            start_char, end_char = match.span()

            if (start_char, end_char) in seen:
                continue

            # Find the token indices
            token_start = token_end = None
            for i, tok in enumerate(tokens):
                if token_start is None and tok["start"] <= start_char < tok["end"]:
                    token_start = i
                if tok["start"] < end_char <= tok["end"]:
                    token_end = i
                    break

            if token_start is not None and token_end is not None:
                spans.append({
                    "start": start_char,
                    "end": end_char,
                    "token_start": token_start,
                    "token_end": token_end,
                    "label": entry["label"],
                })
                seen.add((start_char, end_char))

    return sorted(spans, key=lambda s: s["start"])


nlp = spacy.load("en_ner_bc5cdr_md")


def extract_drugs_spacy(text):
    """Use spaCy to extract CHEMICAL entities and return character-level and token-level spans."""
    doc = nlp(text)
    spans = []

    for ent in doc.ents:
        if ent.label_ == "CHEMICAL":
            spans.append({
                "start": ent.start_char,
                "end": ent.end_char,
                "token_start": ent.start,  # token index where entity starts
                "token_end": ent.end - 1,  # token index where entity ends (inclusive)
                "label": "CHEMICAL",  # or use "CHEMICAL" if you want to match spaCy label
            })

    return spans


def fill_missing_indices(my_data, original_data):
    # Add rows and cols if missing from original data, also add captions and footers to all.
    for item in tqdm(my_data):
        if "col_idx" not in item or "row_idx" not in item:
            table_id = item.get("table_id")
            text = item.get("text")

            # Look for matching entry in original_data
            match = next((
                orig for orig in original_data
                if orig.get("table_id") == table_id and orig.get("text") == text
                   and "col_idx" in orig and "row_idx" in orig
            ), None)

            if match:
                item["col_idx"] = match["col_idx"]
                item["row_idx"] = match["row_idx"]

    return my_data


def merge_adjacent_spans(data):
    merge_counts = defaultdict(int)

    for entry in data:
        spans = entry.get("spans", [])
        if not spans:
            continue

        # Sort spans by token_start
        spans.sort(key=lambda x: x["token_start"])
        merged_spans = []
        prev = spans[0]

        for curr in spans[1:]:
            if (
                prev["label"] == curr["label"]
                and prev["token_end"] + 1 == curr["token_start"]
            ):
                # Merge current span into prev
                prev["end"] = curr["end"]
                prev["token_end"] = curr["token_end"]
                merge_counts[prev["label"]] += 1
            else:
                merged_spans.append(prev)
                prev = curr

        merged_spans.append(prev)
        entry["spans"] = merged_spans

    return data, dict(merge_counts)


def print_unique_texts_for_data(my_data):
    value_texts = []

    for item in my_data:
        text = item["text"]
        if not item.get("spans"):
            value_texts.append(text.lower())

    unique_value_texts = sorted(list(set(value_texts)))
    print(f"Unique value texts: {len(unique_value_texts)}")
    for text in unique_value_texts:
        print("  -", text)


def print_label_counts_ctc(my_data):
    # Extract labels and count occurrences
    label_counts = Counter()

    for item in my_data:
        accept = item.get("accept", [])
        if len(accept) == 1:
            label = accept[0]
            label_counts[label] += 1
        else:
            print(f"Skipping malformed entry: {item['text']}, {item['accept']}")

    print(label_counts)


def print_label_counts_ner(my_data):
    # Extract labels and count occurrences
    label_counts = Counter(
        span["label"]
        for item in my_data
        for span in item.get("spans", [])
    )

    print(f"Labels: {label_counts}")



def print_unique_spans_for_label(my_data, label_name):
    cleaned_data = []

    for item in my_data:
        original_spans = item.get("spans", [])

        # Filter out Num_Value spans
        filtered_spans = [span for span in original_spans if span["label"] not in [label_name]]

        # Only keep the cell if at least one span remains
        if filtered_spans:
            new_item = item.copy()
            new_item["spans"] = filtered_spans
            cleaned_data.append(new_item)

    value_texts = []

    for item in my_data:
        text = item["text"]
        for span in item.get("spans", []):
            if span["label"] == "WEIGHT":
                span_text = text[span["start"]:span["end"]]
                value_texts.append(span_text.lower())

    unique_value_texts = list(set(value_texts))
    print(f"Unique value texts: {len(unique_value_texts)}")
    for text in unique_value_texts:
        print("  -", text)


def remove_spans_by_label(my_data, label_to_remove):
    for item in my_data:
        item["spans"] = [
            span for span in item.get("spans", [])
            if span.get("label") != label_to_remove
        ]



def remove_non_spacy_compatible_spans(my_data):
    dropped_spans = 0
    for item in tqdm(my_data):
        doc = nlp(item["text"])
        valid_spans = []

        for span in item.get("spans", []):
            start, end = span["start"], span["end"]
            spacy_span = doc.char_span(start, end, alignment_mode="contract")

            if spacy_span:
                span["start"] = spacy_span.start_char
                span["end"] = spacy_span.end_char
                valid_spans.append(span)
            else:
                dropped_spans += 1
                # print(f"Dropped invalid span: {item['text'][start:end]!r}")

        item["spans"] = valid_spans

    print(f"Dropped {dropped_spans} spans out of {len(my_data)}")


