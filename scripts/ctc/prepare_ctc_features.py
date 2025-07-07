from collections import defaultdict

from tqdm import tqdm

from pktabner.table_data_structure import Table, TableCell
from pktabner.utils import read_jsonl, write_jsonl


def dict_to_table(table_dict):
    table = Table(
        table_num=table_dict.get("table_num"),
        table_id=table_dict.get("table_id"),
        caption=table_dict.get("caption"),
        footer=table_dict.get("footer"),
        pmid=table_dict.get("pmid"),
        pmcid=table_dict.get("pmcid"),
        table_html=table_dict.get("table_html"),
    )

    for cell_dict in table_dict["cells"]:
        cell = TableCell(
            row=cell_dict["row"],
            col=cell_dict["col"],
            text=cell_dict["text"],
            html=cell_dict.get("html", ""),
            footnotes=cell_dict.get("footnotes", []),
            rowspan=cell_dict.get("rowspan", 1),
            colspan=cell_dict.get("colspan", 1),
            cell_type=cell_dict.get("cell_type"),
            row_type=cell_dict.get("row_type"),
            is_numeric=cell_dict.get("is_numeric", None),
        )
        table.add_cell(cell)

    return table

def build_table_grid(table: Table):
    max_row = max(cell.row + cell.rowspan - 1 for cell in table.cells)
    max_col = max(cell.col + cell.colspan - 1 for cell in table.cells)

    grid = [[None for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    for cell in table.cells:
        for r, c in cell.spanned_cells:
            grid[r][c] = cell  # Non-empty cells only

    return grid


ROW_TYPE_PRIORITY = {
    "row_header_1": 1,
    "row_header_2": 2,
    "row_header_3": 3,
    "section_header": 4,
    None: 99  # for normal data cells
}


def build_header_lookup(table: Table):
    row_header_map = {}
    col_header_map = {}

    max_row = max(cell.row + cell.rowspan for cell in table.cells)
    max_col = max(cell.col + cell.colspan for cell in table.cells)

    # Step 1: row_header_N headers
    row_headers_by_col = defaultdict(list)
    for cell in table.cells:
        if cell.row_type and cell.row_type.startswith("row_header"):
            for c in range(cell.col, cell.col + cell.colspan):
                row_headers_by_col[c].append(cell)

    # Sort by row ascending (top-down) for each column
    for c in row_headers_by_col:
        row_headers_by_col[c].sort(key=lambda h: h.row)

    # Now apply most recent header of each type per row
    for r in range(max_row):
        for c in range(max_col):
            latest_by_type = {}
            for h in row_headers_by_col.get(c, []):
                if h.row + h.rowspan <= r:
                    latest_by_type[h.row_type] = h
                else:
                    break
            cell = next((cell for cell in table.cells if cell.row == r and cell.col == c), None)
            cell_priority = ROW_TYPE_PRIORITY.get(getattr(cell, "row_type", None), 99)

            for header in latest_by_type.values():
                header_priority = ROW_TYPE_PRIORITY.get(getattr(header, "row_type", None), 99)
                if header_priority < cell_priority:
                    row_header_map.setdefault((r, c), []).append(header)

    # Step 2: Section headers
    section_headers_by_col = {c: [] for c in range(max_col)}
    for cell in table.cells:
        if cell.row_type == "section_header":
            for c in range(cell.col, cell.col + cell.colspan):
                section_headers_by_col[c].append(cell)

    # Sort section headers by row for each column (top-down)
    for c in section_headers_by_col:
        section_headers_by_col[c].sort(key=lambda h: h.row)

    # Assign nearest section header above each cell
    for r in range(max_row):
        for c in range(max_col):
            nearest = None
            for header in section_headers_by_col.get(c, []):
                if header.row + header.rowspan <= r:
                    nearest = header  # latest valid one above
                else:
                    break
            cell = next((cell for cell in table.cells if cell.row == r and cell.col == c), None)
            cell_priority = ROW_TYPE_PRIORITY.get(getattr(cell, "row_type", None), 99)
            header_priority = ROW_TYPE_PRIORITY.get(getattr(nearest, "row_type", None), 99)

            if header_priority < cell_priority:
                row_header_map.setdefault((r, c), []).append(nearest)


    # Step 3: Column headers
    for header_cell in table.cells:
        if header_cell.col == 0 or header_cell.col_type:
            for r in range(header_cell.row, header_cell.row + header_cell.rowspan):
                for cc in range(header_cell.col + header_cell.colspan, max_col):
                    col_header_map.setdefault((r, cc), []).append(header_cell)

    return row_header_map, col_header_map



def build_cell_lookup(table: Table):
    lookup = {}
    for cell in table.cells:
        for r, c in cell.spanned_cells:
            lookup[(r, c)] = cell
    return lookup


def get_col_context(grid, col_idx, row_idx):
    current_cell = grid[row_idx][col_idx]
    seen = set()
    col_cells = []

    for i in range(len(grid)):
        if col_idx >= len(grid[i]):
            continue
        c = grid[i][col_idx]
        if not c or id(c) in seen:
            continue
        if i == row_idx:
            col_cells.append("[CELL]")
        elif c is current_cell:
            continue  # skip spanned self
        else:
            col_cells.append(c.text)
            seen.add(id(c))

    return col_cells


def get_row_context(grid, row_idx, col_idx):
    current_cell = grid[row_idx][col_idx]
    seen = set()
    row_cells = []

    for j, c in enumerate(grid[row_idx]):
        if not c or id(c) in seen:
            continue
        if j == col_idx:
            row_cells.append("[CELL]")
        elif c is current_cell:
            continue  # avoid re-adding spanned self
        else:
            row_cells.append(c.text)
            seen.add(id(c))

    return row_cells


def get_cell_quadrant(n_rows, n_cols, row_idx, col_idx):
    row_half = n_rows // 2
    col_half = n_cols // 2
    if row_idx <= row_half and col_idx <= col_half:
        region = "top-left"
    elif row_idx <= row_half and col_idx > col_half:
        region = "top-right"
    elif row_idx > row_half and col_idx <= col_half:
        region = "bottom-left"
    else:
        region = "bottom-right"

    return region

def enrich_cell_features(cell_data, table: Table):
    row_idx = cell_data["row_idx"]
    col_idx = cell_data["col_idx"]

    cell_lookup = build_cell_lookup(table)
    grid = build_table_grid(table)
    n_rows = len(grid)
    n_cols = len(grid[0])

    cell = cell_lookup.get((row_idx, col_idx))
    if not cell:
        print(f"Warning: Cell ({row_idx}, {col_idx}) missing in table {table.table_id}")
        return None
    assert cell_data["col_idx"] == cell.col, "column mismatch for retrieved cell"
    assert cell_data["row_idx"] == cell.row, "row mismatch for retrieved cell"
    if cell_data["text"] != cell.text:
        a = 1
        print(f"\ncell text mismatch in table: {table.table_id}, retrieved cell text: {cell.text}, annotated cell text: {cell_data['text']}\n")

    # Get unique row and column cell values
    row_cells = get_row_context(grid, row_idx, col_idx)
    col_cells = get_col_context(grid, col_idx, row_idx)

    # Get quadrant
    region = get_cell_quadrant(n_rows, n_cols, row_idx, col_idx)

    # Get Absolute and reverse position
    pos = (row_idx, col_idx)
    rev_pos = (n_rows - row_idx - 1, n_cols - col_idx - 1)

    # Get categorical cell info
    cell_type = "data" if cell.cell_type == "data_cell" else "header"
    row_type = cell.row_type
    col_type = "header" if cell.col == 0 or cell.col_type == "header_column" else None
    spanned_cells = cell.spanned_cells
    parent = cell.effective_parent_cell.text if cell.effective_parent_cell else None

    # Find header cells
    row_header_map, col_header_map = build_header_lookup(table)
    row_headers = [h.text for h in row_header_map.get((row_idx, col_idx), [])]
    col_headers = [h.text for h in col_header_map.get((row_idx, col_idx), [])]
    if col_idx == 0:
        assert col_headers == []
    if row_idx == 0:
        assert row_headers == []

    if parent in row_headers or parent in col_headers:
        parent = None

    return {
        "table_id": cell_data["table_id"],
        "row_idx": row_idx,
        "col_idx": col_idx,
        "cell_text": cell.text,
        "row_headers": row_headers,
        "col_headers": col_headers,
        "row_context": row_cells,
        "col_context": col_cells,
        "region": region,
        "pos": pos,
        "rev_pos": rev_pos,
        "cell_type": cell_type,
        "row_type": row_type,
        "col_type": col_type,
        "effective_parent": parent,
        "spans": cell_data.get("spans", []),
        "tokens": cell_data.get("tokens", []),
        "label": cell_data["accept"][0],
    }


annotated_cells = list(read_jsonl("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/annotated_ready/cleaned_ready_ctc_data.jsonl"))
parsed_tables = list(read_jsonl("/home/vsmith/PycharmProjects/E2E_PK_Table_IE_Draft/data/parsed_pk_tables/parsed_pk_tables_updated.jsonl"))

# Assume: tables is a list of Table objects
table_objs = [dict_to_table(td) for td in parsed_tables]
table_map = {table.table_id: table for table in table_objs}

enriched_cells = []
# todo: please note issue that (PMC6102747 , table_id "30015887 | unknown") is multiple places in table database - 2 tables with same name, real ids are roman numerals
annotated_cells = [x for x in annotated_cells if x["table_id"] not in ["30015887 | unknown", "31265522 | 3"]]


for cell in tqdm(annotated_cells):
    table_id = cell["table_id"]
    table = table_map.get(table_id)
    if not table:
        continue

    enriched = enrich_cell_features(cell, table)
    if enriched:
        enriched_cells.append(enriched)
    else:
        print(f"Failed for cell: {table_id}, row: {cell['row_idx']}, col: {cell['col_idx']}")
    a = 1

print(f"Saving cells: {len(enriched_cells)}")
write_jsonl("/home/vsmith/PycharmProjects/PKTabNER_Draft/data/annotated_cells/new/ner_annotated/ctc_to_annotate/annotated_ready/cleaned_ready_with_features_data.jsonl", enriched_cells)

a = 1