import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import ujson
from sklearn.model_selection import train_test_split

from pktabner.ner.patterns import FRACTION_SLASH_RE, HTML_TAG_RE


def basic_preprocessing(text: str) -> str:
    text = text.lower()  # Convert to lowercase
    text = unicodedata.normalize('NFKC', text).replace('\xa0', ' ')  # Normalize Unicode & replace spaces
    text = FRACTION_SLASH_RE.sub("/", text)  # Normalize fraction slashes
    text = HTML_TAG_RE.sub("", text)  # Remove HTML tags
    #text = STOP_WORDS_RE.sub("", text)  # Remove stop words
    return text



def write_jsonl(file_path, lines):
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))


def read_jsonl(file_path):
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open(encoding='utf-8') as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue


def append_result_to_jsonl(filepath, data):
    with open(filepath, "a") as f:
        f.write(json.dumps(data) + "\n")


def filter_dicts_not_in_other(source_list, exclusion_list, key_fn):
    exclusion_keys = {key_fn(d) for d in exclusion_list}
    return [d for d in source_list if key_fn(d) not in exclusion_keys]


def deduplicate_dicts(data, key1, key2):
    """
    Deduplicates a list of dictionaries based on the combined value of two keys.

    Parameters:
        data (list): List of dictionaries.
        key1 (str): The first key to use for deduplication.
        key2 (str): The second key to use for deduplication.

    Returns:
        list: Deduplicated list of dictionaries.
    """
    seen = set()
    deduped = []

    for item in data:
        key_combo = (item.get(key1), item.get(key2))
        if key_combo not in seen:
            seen.add(key_combo)
            deduped.append(item)

    return deduped


def deduplicate_dicts_spans(data, key1, key="spans"):
    """
    Deduplicates a list of dictionaries based on the combined value of two keys.

    Parameters:
        data (list): List of dictionaries.
        key1 (str): The first key to use for deduplication.
        key2 (str): The second key to use for deduplication.

    Returns:
        list: Deduplicated list of dictionaries.
    """
    seen = set()
    deduped = []

    for item in data:
        key_combo = (item.get(key1), item["spans"][0]["start"])
        if key_combo not in seen:
            seen.add(key_combo)
            deduped.append(item)

    return deduped


def stratified_train_test_split_df(df_input, stratify_colname='y',
                                frac_train=0.8, frac_test=0.2,
                                random_state=42):
    if frac_train + frac_test != 1.0:
        raise ValueError(f'fractions {frac_train}, {frac_test} do not add up to 1.0')
    if stratify_colname not in df_input.columns:
        raise ValueError(f'{stratify_colname} is not a column in the dataframe')

    X = df_input
    y = df_input[stratify_colname]

    # Identify rare classes (with 1 or 2 examples)
    rare_classes = y.value_counts()[y.value_counts() <= 2].index.tolist()

    # Separate rare and non-rare classes
    df_rare = df_input[df_input[stratify_colname].isin(rare_classes)]
    df_non_rare = df_input[~df_input[stratify_colname].isin(rare_classes)]

    # Step 1: Split non-rare data into train and test (if not empty)
    if len(df_non_rare) > 0:
        df_train_non_rare, df_test_non_rare = train_test_split(
            df_non_rare,
            stratify=df_non_rare[stratify_colname],
            test_size=(1.0 - frac_train),
            random_state=random_state,
            shuffle=True
        )
    else:
        df_train_non_rare = pd.DataFrame(columns=df_input.columns)
        df_test_non_rare = pd.DataFrame(columns=df_input.columns)

    # Step 2: Split rare data into train and test (if not empty)
    if len(df_rare) > 0:
        df_rare_train, df_rare_test = train_test_split(
            df_rare,
            test_size=(1.0 - frac_train),
            random_state=random_state,
            shuffle=True
        )
    else:
        df_rare_train = pd.DataFrame(columns=df_input.columns)
        df_rare_test = pd.DataFrame(columns=df_input.columns)

    # Step 3: Combine non-rare and rare sets
    df_train = pd.concat([df_train_non_rare, df_rare_train]).sample(
        frac=1, random_state=random_state).reset_index(drop=True)
    df_test = pd.concat([df_test_non_rare, df_rare_test]).sample(
        frac=1, random_state=random_state).reset_index(drop=True)

    # Step 4: Check that the total size matches
    assert len(df_input) == len(df_train) + len(df_test), \
        "The total length of train and test sets does not match the input."

    return df_train, df_test


def stratified_train_test_split_dicts(data, stratify_key='label', frac_train=0.8, frac_test=0.2, random_state=42):
    if not 0.999 <= (frac_train + frac_test) <= 1.001:
        raise ValueError(f'Fractions {frac_train}, {frac_test} must sum to 1.0')

    label_counts = Counter(item[stratify_key] for item in data)
    rare_classes = {label for label, count in label_counts.items() if count <= 2}

    # Separate rare and non-rare
    rare = [item for item in data if item[stratify_key] in rare_classes]
    non_rare = [item for item in data if item[stratify_key] not in rare_classes]

    # Stratified split on non-rare
    if non_rare:
        non_rare_labels = [item[stratify_key] for item in non_rare]
        train_nr, test_nr = train_test_split(
            non_rare,
            stratify=non_rare_labels,
            test_size=frac_test,
            random_state=random_state,
            shuffle=True
        )
    else:
        train_nr, test_nr = [], []

    # Random split on rare
    if rare:
        random.seed(random_state)
        random.shuffle(rare)
        cutoff = int(len(rare) * frac_train)
        train_r = rare[:cutoff]
        test_r = rare[cutoff:]
    else:
        train_r, test_r = [], []

    # Combine and shuffle
    train = train_nr + train_r
    test = test_nr + test_r
    random.seed(random_state)
    random.shuffle(train)
    random.shuffle(test)

    # Sanity check
    assert len(train) + len(test) == len(data), "Mismatch in total data size after split."

    return train, test



def train_test_split_dicts(data, stratify_key='label', frac_train=0.8, frac_test=0.2, random_state=42):
    if not 0.999 <= (frac_train + frac_test) <= 1.001:
        raise ValueError(f'Fractions {frac_train}, {frac_test} must sum to 1.0')

    train, test = train_test_split(
        data,
        test_size=frac_test,
        random_state=random_state,
        shuffle=True
    )

    random.seed(random_state)
    random.shuffle(train)
    random.shuffle(test)

    # Sanity check
    assert len(train) + len(test) == len(data), "Mismatch in total data size after split."
    return train, test



