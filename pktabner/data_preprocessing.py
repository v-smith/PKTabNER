from dataclasses import dataclass
from typing import Dict, List

from bs4 import BeautifulSoup
from tqdm import tqdm


@dataclass
class PreprocessingConfig:
    special_tokens: bool = True
    window_size: int = 0  # How many words before/after mention
    include_combined_context: bool = True  # Include footer/caption
    mention_tokens: tuple = ("[MENTION]", "[/MENTION]")  # Custom special tokens
    truncate: bool = True
    max_items: int = 10


TABLE_DEFAULT_CONFIG = PreprocessingConfig(
    special_tokens=True,
    window_size=0,
    include_combined_context=True,
    truncate=True,
    max_items=10,
)

# --- Table Parsing and Context Extraction --- #

def is_index_sequence(values: List[str], allow_leading_blank: bool = False) -> bool:
    """Check if a list is a 0-based increasing integer sequence, optionally allowing an initial empty string."""
    if allow_leading_blank and values and values[0].strip() == "":
        values = values[1:]  # Strip blank header corner
    if not values:
        return False
    return all(cell.isdigit() for cell in values) and list(map(int, values)) == list(range(len(values)))


def parse_html_table(html_table: str) -> Dict[str, any]:
    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")

    if not table or not rows:
        return {"markdown": "", "table_data": []}

    # --- Parse all rows ---
    parsed_rows = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        parsed_rows.append([cell.get_text(strip=True) for cell in cells])

    # --- Detect and remove index row ---
    if is_index_sequence(parsed_rows[0], allow_leading_blank=True):
        parsed_rows = parsed_rows[1:]

    # --- Detect and remove index column ---
    first_col = [row[0] for row in parsed_rows if len(row) > 0]
    if is_index_sequence(first_col):
        parsed_rows = [row[1:] for row in parsed_rows]

    # --- Markdown ---
    markdown_lines = []
    if len(parsed_rows) > 0:
        markdown_lines.append("| " + " | ".join(parsed_rows[0]) + " |")
        #markdown_lines.append("|" + " --- |" * len(parsed_rows[0]))
        for row in parsed_rows[1:]:
            markdown_lines.append("| " + " | ".join(row) + " |")
    else:
        markdown_lines.append("")

    return {
        "markdown": "\n".join(markdown_lines),
        "table_data": parsed_rows,
    }


def extract_context_from_table(
    parsed_table: Dict,
    example: Dict,
    window: int = 3
) -> Dict[str, str]:
    """
    Extracts row and column context around a target cell, including up to `window` cells
    on either side of the target index. Highlights the target cell using [MENTION] tags.
    """
    full_table  = parsed_table["table_data"]
    total_rows = len(full_table)
    #mention = example["mention"]
    text_with_tagged_mention = example["text_with_tagged_mention"]
    #text = example["text"]
    row_idx = example.get("row_idx", -1)
    col_idx = example.get("col_idx", -1)
    target_cell = full_table[row_idx][col_idx]
    #print(f"Target cell: {target_cell}, text: {text}")

    # --- Row context ---
    row = full_table[row_idx] if 0 <= row_idx < total_rows else []
    row_start = max(0, col_idx - window)
    row_end = col_idx + window + 1
    row_context = []
    for i in range(row_start, row_end):
        if i < len(row):
            cell = row[i]
            if i == col_idx:
                cell = text_with_tagged_mention
            row_context.append(cell)

    # --- Column context ---
    col_context = []
    row_range_start = max(0, row_idx - window)
    row_range_end = row_idx + window + 1
    for i in range(row_range_start, row_range_end):
        if i < total_rows and col_idx < len(full_table[i]):
            cell = full_table[i][col_idx]
            if i == row_idx:
                cell = text_with_tagged_mention
            col_context.append(cell)

    return {
        "row_context": "| " + " | ".join(row_context) + " |" if row_context else "",
        "column_context": "| " + " | ".join(col_context) + " |" if col_context else "",
    }

def format_table_context_for_llm(sample: dict) -> str:
    """
    Formats table context for LLM input from a preprocessed sample.

    Args:
        sample (dict): A dictionary containing preprocessed table features.

    Returns:
        str: Formatted string ready for LLM input.
    """
    parts = ["The following context is provided to help you. \n It shows the table row and column from which the mention is derived (with mention tagged) and the table footer, if available."]

    # Include the row context
    row_context = sample.get("row_context", "")
    if row_context:
        parts.append(f"[ROW] {row_context.strip()}")

    # Always include the column context
    col_context = sample.get("col_context", "")
    if col_context:
        parts.append(f"[COLUMN] {col_context.strip()}")

    # Include footer if available
    footer = sample.get("footer", "")
    if footer:
        parts.append(f"[FOOTER] {footer.strip()}")

    """caption = sample.get("caption", "")
    if caption:
        caption = truncate_by_words(caption.strip(), max_words=20)
        parts.append(f"[CAPTION] {caption}")"""

    return "\n".join(parts)


# --- Preprocessing Pipeline --- #

def prep_table_features(data, config: PreprocessingConfig):
    for sample in tqdm(data):
        parsed_table = parse_html_table(sample["table_html"])

        context_parts = extract_context_from_table(parsed_table, sample)
        sample["row_context"] = context_parts["row_context"]
        sample["col_context"] = context_parts["column_context"]
        sample["table_context_llm"] = format_table_context_for_llm(sample)
    return data
