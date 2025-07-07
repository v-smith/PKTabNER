import re
from typing import List, Optional, Dict, Tuple

def is_numeric_cell(text: str) -> bool:
    """
    Determines if a string represents a numeric value or expression,
    handling formats like ±, scientific notation, ranges, brackets,
    comparisons, and footnote suffixes.
    """

    # Normalize Unicode characters and whitespace
    text = text.strip().lower()
    # Remove obvious trailing footnote letters after numbers: e.g. 6.4 (0.5)b,c
    text = re.sub(r"(?<=\d)[\s*\)\]]?[a-g],[a-g]$", "", text)

    replacements = ["fixed", "fix", "r^2=", "p=", "sd=", "range:", "(na)", "n.c.",
                    "e-", "+/-", "-", " ", " ", "to",
                    ",", ".",  "+", "(", ")", "[", "]", ";", ":",
                    "±", "×", "x", "%", "^", "=", "^", "/", "∼", "〜", "∆", "△",
                    "⩽", "⩾", "≤", "≥", ">", "<",
                    ]
    for x in replacements:
        text = text.replace(x, "")

    text = text.replace(" ", "")

    return text.isdigit()


class TableCell:
    def __init__(self, row: int, col: int, text: str,
                 html: Optional[str] = None, footnotes: Optional[str] = None,
                 rowspan: int = 1, colspan: int = 1,
                 cell_type: Optional[str] = None,
                 row_type: Optional[str] = None,
                 is_numeric: Optional[bool] = None):
        self.row = row
        self.col = col
        self.rowspan = rowspan
        self.colspan = colspan
        self.text = text
        self.html = html or ""
        self.footnotes: List[str] = footnotes if isinstance(footnotes, list) else ([footnotes] if footnotes else [])
        self.cell_type = cell_type
        self.row_type = row_type
        self.col_type = None
        self.is_empty = self.text.strip() == ""
        self.inferred_indent_level = 0
        self.effective_parent_cell = None
        self.semantic_children: List[Tuple[int, int]] = []
        self.labels: List[Dict] = []
        self.entities: List[Dict] = []
        self.is_numeric = is_numeric if is_numeric is not None else is_numeric_cell(text)

        # Track which (row, col) cells this one spans
        self.spanned_cells: List[Tuple[int, int]] = [(row + dr, col + dc)
                                                     for dr in range(rowspan)
                                                     for dc in range(colspan)]

    def to_dict(self):
        return {
            "row": self.row,
            "col": self.col,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
            "text": self.text,
            "html": self.html,
            "footnotes": self.footnotes,
            "is_numeric": self.is_numeric,
            "is_empty": self.is_empty,
            "cell_type": self.cell_type,
            "row_type": self.row_type,
            "col_type": self.col_type,
            "spanned_cells": self.spanned_cells,
            "effective_parent": self.effective_parent_cell.text if self.effective_parent_cell else None,
            "labels": self.labels,
            "entities": self.entities,

        }


class Table:
    def __init__(self, table_num: str, table_id = None, table_html=None, caption: Optional[str] = None,
                 footer: Optional[str] = None,
                 pmid: Optional[str] = None, pmcid: Optional[str] = None):
        self.table_id = table_id
        self.pmid = pmid
        self.pmcid = pmcid
        self.table_num = table_num
        self.caption = caption or ""
        self.footer = footer or ""
        self.table_html = table_html
        self.cells: List[TableCell] = []

    def add_cell(self, cell: TableCell):
        self.cells.append(cell)

    def to_dict(self):
        return {
            "table_id": self.table_id,
            "table_num": self.table_num,
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "caption": self.caption,
            "table_html": self.table_html,
            "footer": self.footer,
            "cells": [cell.to_dict() for cell in self.cells]
        }
