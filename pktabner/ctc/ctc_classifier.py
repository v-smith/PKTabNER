import pandas as pd


def join_list_elements(cell_content):
    if isinstance(cell_content, list):
        # Filter out any non-string elements within the list if necessary, then join
        return " ".join([str(item) for item in cell_content if pd.notna(item) and str(item).strip() != ''])
    elif pd.notna(cell_content) and str(cell_content).strip() != '':
        # If it's not a list but a valid string-like object, just return it
        return str(cell_content)
    return "" # Return empty string for NaN or empty/non-string content


def join_list_elements_row_tokens(cell_content):
    if isinstance(cell_content, list):
        # Filter out any non-string elements within the list if necessary, then join
        return f" <ROW_SEP> ".join([str(item) for item in cell_content if pd.notna(item) and str(item).strip() != ''])
    elif pd.notna(cell_content) and str(cell_content).strip() != '':
        # If it's not a list but a valid string-like object, just return it
        return str(cell_content)
    return "" # Return empty string for NaN or empty/non-string content


def join_list_elements_col_tokens(cell_content):
    if isinstance(cell_content, list):
        # Filter out any non-string elements within the list if necessary, then join
        return f" <COL_SEP> ".join([str(item) for item in cell_content if pd.notna(item) and str(item).strip() != ''])
    elif pd.notna(cell_content) and str(cell_content).strip() != '':
        # If it's not a list but a valid string-like object, just return it
        return str(cell_content)
    return "" # Return empty string for NaN or empty/non-string content


def ensure_numeric_pair(item):
    if isinstance(item, (tuple, list)):
        if len(item) == 2:
            try:
                # Attempt to convert to numbers, default to 0 if conversion fails
                return [int(item[0]) if pd.notna(item[0]) else 0,
                        int(item[1]) if pd.notna(item[1]) else 0]
            except (ValueError, TypeError):
                # Handle cases where elements inside the tuple/list aren't convertible
                return [0, 0]
        else:
            # Handle cases where the list/tuple length is not 2
            return [0, 0]
    elif pd.isna(item):
        return [0, 0] # Return default for NaN
    else:
        # If it's a single number, assume it's row_idx and col_idx is 0
        try:
            return [int(item) if pd.notna(item) else 0, 0]
        except (ValueError, TypeError):
            return [0, 0]

