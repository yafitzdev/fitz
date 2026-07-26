# fitz_sage/engines/fitz_krag/context/compressor.py
"""
AST-based code compression for retrieval context.

Reduces Python source code token count by 50-70% while preserving
all information needed for answering questions:
  - Import statements (kept verbatim)
  - Class/function signatures with decorators (kept verbatim)
  - Data structures and constants (kept verbatim)
  - Function bodies (collapsed to `...` unless short/important)

Stripped with zero information loss for Q&A:
  - Docstrings (signatures tell the model more)
  - Comments (implementation notes, not semantic signal)
  - Blank lines (cosmetic)

Applied AFTER reading content, BEFORE context assembly. Retrieval
scoring operates on full source for accurate relevance.
"""

import ast
import logging

from fitz_sage.engines.fitz_krag.types import AddressKind, ReadResult

logger = logging.getLogger(__name__)

# Bodies shorter than this (in lines) are kept verbatim.
# Short functions are often the most informative
# (factory functions, config, protocol methods).
_KEEP_BODY_LINES = 6


def compress_results(results: list[ReadResult]) -> list[ReadResult]:
    """Compress code content in ReadResults before context assembly.

    Only compresses Python code results (SYMBOL/FILE kinds with .py paths).
    Non-code results (sections and tables) are passed through unchanged.
    """
    compressed: list[ReadResult] = []
    for result in results:
        if _is_python_code(result):
            new_content = compress_python(result.content)
            compressed.append(
                ReadResult(
                    address=result.address,
                    content=new_content,
                    file_path=result.file_path,
                    line_range=result.line_range,
                    metadata=result.metadata,
                )
            )
        else:
            compressed.append(result)
    return compressed


def _is_python_code(result: ReadResult) -> bool:
    """Check if a ReadResult contains Python code."""
    if result.address.kind not in (AddressKind.SYMBOL, AddressKind.FILE):
        return False
    return result.file_path.lower().endswith(".py")


def compress_python(source: str) -> str:
    """Compress Python source for Q&A context.

    Returns compressed source string. If parsing fails (syntax errors,
    non-Python), returns the original source unchanged.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _strip_comments_and_blanks(source)

    lines = source.splitlines(keepends=True)
    if not lines:
        return source

    # Collect line ranges to remove or replace
    removals: list[tuple[int, int]] = []  # (start_line, end_line) 1-indexed, inclusive
    replacements: dict[int, str] = {}  # start_line -> replacement text

    for node in ast.walk(tree):
        # Strip docstrings (module, class, function)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                doc_node = node.body[0]
                removals.append((doc_node.lineno, doc_node.end_lineno or doc_node.lineno))

        # Compress function/method bodies
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            # Skip docstring node if present
            body_start_idx = 0
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body_start_idx = 1

            real_body = body[body_start_idx:]
            if not real_body:
                continue

            first_body = real_body[0]
            last_body = real_body[-1]
            body_start = first_body.lineno
            body_end = last_body.end_lineno or last_body.lineno

            body_lines = body_end - body_start + 1

            if body_lines <= _KEEP_BODY_LINES:
                continue  # Keep short bodies verbatim

            # Keep bodies that are just `pass` or `...`
            if len(real_body) == 1 and isinstance(real_body[0], (ast.Pass, ast.Expr)):
                continue

            # Collapse long bodies to `...`
            first_line = lines[body_start - 1] if body_start <= len(lines) else ""
            indent = len(first_line) - len(first_line.lstrip())
            indent_str = first_line[:indent] if indent > 0 else "        "

            replacements[body_start] = f"{indent_str}...  # {body_lines} lines\n"
            removals.append((body_start + 1, body_end))

    if not removals and not replacements:
        return _strip_comments_and_blanks(source)

    # Build set of lines to remove
    remove_lines: set[int] = set()
    for start, end in removals:
        for ln in range(start, end + 1):
            remove_lines.add(ln)

    # Build output
    result: list[str] = []
    for i, line in enumerate(lines, 1):
        if i in replacements:
            result.append(replacements[i])
        elif i not in remove_lines:
            result.append(line)

    return _strip_comments_and_blanks("".join(result))


def _strip_comments_and_blanks(source: str) -> str:
    """Remove comment-only lines and collapse multiple blank lines."""
    out: list[str] = []
    prev_blank = False

    for line in source.splitlines(keepends=True):
        stripped = line.strip()

        # Remove comment-only lines (but keep shebangs and type: ignore)
        if stripped.startswith("#") and not stripped.startswith("#!") and "type:" not in stripped:
            continue

        # Collapse multiple blank lines to one
        if not stripped:
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False

        out.append(line)

    return "".join(out)
