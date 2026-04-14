import re

def fix_math(text: str) -> str:
    """
    Replace \_, \^, \{, \} with _, ^, {, } 
    only inside $...$ math segments.
    """
    # regex to match $...$ blocks (non-greedy)
    pattern = r"\$(.+?)\$"
    text = re.sub(r"\\\$", r"$", text)
    
    def fix_eq(match):
        content = match.group(1)
        # replace escaped sequences only inside math
        content = content.replace(r"\_", "_")
        content = content.replace(r"\^", "^")
        content = content.replace(r"\{", "{").replace(r"\}", "}")
        # Reduce double backslashes to single before commands
        content = re.sub(r"\\\\([a-zA-Z]+)", r"\\\1", content)
        return f"${content}$"

    # replace all $...$ occurrences
    return re.sub(pattern, fix_eq, text)

def add_column_bars(match):
    cols = match.group(1)
    cols_with_bars = '|' + '|'.join(list(cols)) + '|'
    return f'\\begin{{tabular}}{{{cols_with_bars}}}'

def make_latex(latex_str):
    
    # 1. Fix math
    latex_str = fix_math(latex_str)
    latex_str = replace_superscripts(latex_str)
    
    # 2. Delete the fontzie setting
    latex_str = re.sub(r"\\fontsize.*?\\selectfont" , "", latex_str, flags=re.DOTALL)

    # 3. Strip \begin and \end{table}
    latex_str = re.sub(r"(\\begin\{table\}\[[^\]]*\])", r"", latex_str)
    latex_str = re.sub(r"(\\end\{table\})", r"", latex_str)
    
    # 4. Change tabular* to tabular environment
    latex_str = re.sub(r'\\begin{tabular\*}\{[^}]*\}\{@\{\\extracolsep\{\\fill\}\}([lcr|]+)\}',     r'\\begin{tabular}{\1}', latex_str)
    latex_str = re.sub(r"\\end\{tabular\*}" , r"\\end{tabular}", latex_str)

    # 5. Change to boxed table
    latex_str = re.sub(r'\\begin{tabular}\{([lcr|]+)\}', add_column_bars, latex_str) # vlines!
    latex_str = re.sub(r"\\addlinespace\[\d+\.?\d*pt\]", r"", latex_str, flags=re.DOTALL) # remove addlinspace
    latex_str = re.sub(r"\\midrule\n", r"", latex_str, flags=re.DOTALL) # remove midrule
    latex_str = re.sub(r"\\toprule\n", r"", latex_str, flags=re.DOTALL) # remove toprule
    latex_str = re.sub(r"\\bottomrule\n", r"", latex_str, flags=re.DOTALL) # remove bottomrule
    latex_str = re.sub(r"(\\multicolumn\{\d+\}\{[clr])\}", r"\1|}", latex_str) # fix multicolumn
    latex_str = re.sub(r"\\cmidrule\(lr\)\{\d+-\d+\}", "", latex_str) # remove cmidrule (spanners)
    latex_str = re.sub(r"\\\\\s*\n", r"\\\ \\hline\n", latex_str) # everywhere hlines!
    latex_str = re.sub(r"(\\begin\{tabular\}\{[^}]+\})", r"\1 \\hline", latex_str) # add hline above table
 
    # 6. Strip \n
    latex_str = latex_str.lstrip("\n")
    latex_str = latex_str.strip("\n")
    
    return latex_str

def insert_rows(latex_str: str, insertions: dict[int, str]) -> str:
    """
    Inserts multiple LaTeX rows at specified row numbers in a tabular environment.

    Parameters:
        latex_str (str): The LaTeX tabular string.
        insertions (dict[int, str]): Dictionary where keys are row indices (0-based, after header),
                                     and values are the LaTeX rows to insert (should end with '\\\\ \\hline').

    Returns:
        str: The updated LaTeX string.
    """
    lines = latex_str.strip().splitlines()

    # Locate \begin{tabular} and \end{tabular}
    begin_idx = next(i for i, line in enumerate(lines) if line.strip().startswith(r"\begin{tabular}"))
    end_idx = next(i for i, line in enumerate(lines) if line.strip().startswith(r"\end{tabular}"))

    tabular_lines = lines[begin_idx + 1:end_idx]

    # Sort insertions in ascending order so later inserts adjust to earlier ones
    for idx in sorted(insertions.keys()):
        if idx < 0 or idx > len(tabular_lines)+1:
            raise IndexError(f"Row index {idx} is out of bounds.")
        tabular_lines.insert(idx, insertions[idx])

    # Reconstruct the full LaTeX
    new_latex = (
        lines[:begin_idx + 1] +
        tabular_lines +
        lines[end_idx:]
    )
    return "\n".join(new_latex)

def delete_rows(latex_str: str, row_numbers: list[int]) -> str:
    """
    Deletes multiple rows from a LaTeX tabular environment.

    Parameters:
        latex_str (str): The original LaTeX tabular string.
        row_numbers (list[int]): Row indices to delete (0-based, excluding the header).

    Returns:
        str: The updated LaTeX string with specified rows removed.
    """
    lines = latex_str.strip().splitlines()

    # Find tabular start and end
    begin_idx = next(i for i, line in enumerate(lines) if line.strip().startswith(r"\begin{tabular}"))
    end_idx = next(i for i, line in enumerate(lines) if line.strip().startswith(r"\end{tabular}"))

    # Extract body of the table
    tabular_lines = lines[begin_idx + 1:end_idx]

    # Convert row_numbers to actual indices in tabular_lines
    delete_indices = [r for r in row_numbers]  # +1 skips the header

    # Sort in reverse so deletion doesn't affect next indices
    for idx in sorted(delete_indices, reverse=True):
        if 0 <= idx < len(tabular_lines)+1:
            del tabular_lines[idx]
        else:
            raise IndexError(f"Row index {idx-1} is out of range for deletion.")

    # Reconstruct table
    return "\n".join(lines[:begin_idx + 1] + tabular_lines + lines[end_idx:])


def replace_latex_table_cell(latex_str: str, new_text: str, row: int, col: int) -> str:
    """
    Replace the content of a LaTeX table cell by (row, col) position.
    
    Args:
        latex_str (str): Full LaTeX table string (e.g., from DataFrame.to_latex()).
        row (int): Zero-based row index (excluding LaTeX formatting lines).
        col (int): Zero-based column index.
        new_text (str): Replacement text to insert.

    Returns:
        str: Updated LaTeX string.
    """
    lines = latex_str.strip().splitlines()
    
    # Identify lines that contain table rows (ignore hline, tabular, etc.)
    table_lines = [i for i, line in enumerate(lines) if '&' in line and '\\' in line]
    
    if row >= len(table_lines):
        raise IndexError(f"Row {row} out of range. Table has {len(table_lines)} data rows.")
    
    # Modify the target line
    target_line_idx = table_lines[row]
    cells = lines[target_line_idx].split('&')
    
    if col >= len(cells):
        raise IndexError(f"Column {col} out of range. Row has {len(cells)} columns.")
    
    # Strip off trailing \\, if it's part of the last cell
    cells[-1] = cells[-1].rstrip().rstrip('\\').rstrip()
    
    # Replace the content
    cells[col] = f' {new_text} '
    
    # Reconstruct the line and replace it
    new_line = ' &'.join(cells) + r' '
    lines[target_line_idx] = new_line
    
    return '\n'.join(lines)

def insert_multicolumn(latex_str: str, row: int, col_start: int, col_span: int, text: str, align: str = 'c') -> str:
    """
    Insert a \multicolumn at a given row and column position in a LaTeX table.

    Args:
        latex_str (str): Full LaTeX table string (e.g., from DataFrame.to_latex()).
        row (int): Zero-based row index (excluding \hline, \toprule, etc.).
        col_start (int): Zero-based column index where the multicolumn starts.
        col_span (int): How many columns the multicolumn should span.
        text (str): The content inside the multicolumn.
        align (str): Alignment of the multicolumn ('l', 'c', or 'r').

    Returns:
        str: Modified LaTeX table with inserted multicolumn.
    """
    lines = latex_str.strip().splitlines()

    # Identify which lines are actual table rows (not \hline, \toprule, etc.)
    table_lines = [i for i, line in enumerate(lines) if '&' in line and '\\' in line]
    
    if row >= len(table_lines):
        raise IndexError(f"Row {row} out of range. Table has {len(table_lines)} data rows.")

    # Get target line and split into cells
    target_line_idx = table_lines[row]
    cells = lines[target_line_idx].split('&')

    # Safety check for column range
    if col_start + col_span > len(cells):
        raise IndexError("Multicolumn exceeds number of columns in this row.")

    # Build multicolumn string
    multicol_str = f"\\multicolumn{{{col_span}}}{{{align}}}{{{text}}}"

    # Replace the selected range with the multicolumn
    new_cells = (
        cells[:col_start] +
        [multicol_str] +
        cells[col_start + col_span:]
    )

    # Clean up last cell (remove \\ if embedded)
    new_cells[-1] = new_cells[-1].rstrip().rstrip('\\').rstrip()

    # Reconstruct the line
    new_line = ' &'.join(new_cells) + r' '
    lines[target_line_idx] = new_line

    return '\n'.join(lines)

def fix_reference(text: str, citation_map: dict[str, str]) -> str:
    """
    Replace @Key references in the text with \citeproc{ref-Key}{Custom text}, using a mapping.

    """
    def replacer(match):
        raw_key = match.group(1)
        # Convert LaTeX escaped underscores (\_) back to plain underscores
        normalized_key = raw_key.replace(r'\_', '_')
        if normalized_key in citation_map:
            return f"\\citeproc{{ref-{normalized_key}}}{{{citation_map[normalized_key]}}}"
        else:
            return match.group(0)  # Leave unchanged if not found

    # Allow letters, digits, underscores, backslashes, and hyphens in the key
    return re.sub(r'@([A-Za-z0-9\\_\-]+)', replacer, text)


def replace_superscripts(text):
    # Match one or more digits, followed by \\^, followed by one or more digits
    # return re.sub(r'([^\s\\]+)\\\^(\d+)', r'\1\\textsuperscript{\2}', text)
    return re.sub(r'([^\s\\]+)\\\^(\d+)\\\^', r'\1\\textsuperscript{\2}', text)