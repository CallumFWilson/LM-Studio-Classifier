import ast

def parse_label_column(column):
    """Parse stringified lists safely into Python lists."""
    return column.apply(lambda val: val if isinstance(val, list) else safe_eval(val))

def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []
