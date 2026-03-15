import ast
import os
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(".")   # change if needed, e.g. Path(r"C:\Users\...\your_app")

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
}

MIN_STRING_LEN = 4
MIN_DUPLICATE_COUNT = 2
IGNORE_STRINGS = {
    "PC1", "PC2", "PC3",
    "Q1", "Q2", "Q3", "Q4",
    "ticker", "permno",
    "__main__",
}

IGNORE_NUMBERS = {0, 1, -1, 2, 3, 4, 5, 10, 100}


def iter_python_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
            if filename.endswith(".py"):
                yield Path(dirpath) / filename


class ConstantCollector(ast.NodeVisitor):
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.strings = []
        self.numbers = []
        self.lists = []
        self.dicts = []
        self.assigned_constants = []

    def visit_Assign(self, node):
        # Capture UPPER_CASE constant assignments
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                self.assigned_constants.append({
                    "file": str(self.filepath),
                    "line": node.lineno,
                    "name": target.id,
                    "value_type": type(node.value).__name__,
                    "preview": safe_preview(node.value),
                })
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            val = node.value.strip()
            if len(val) >= MIN_STRING_LEN and val not in IGNORE_STRINGS:
                self.strings.append((val, node.lineno))
        elif isinstance(node.value, (int, float)):
            if node.value not in IGNORE_NUMBERS:
                self.numbers.append((node.value, node.lineno))
        self.generic_visit(node)

    def visit_List(self, node):
        preview = safe_preview(node)
        if preview:
            self.lists.append((preview, node.lineno))
        self.generic_visit(node)

    def visit_Tuple(self, node):
        preview = safe_preview(node)
        if preview and len(node.elts) >= 3:
            self.lists.append((preview, node.lineno))
        self.generic_visit(node)

    def visit_Dict(self, node):
        preview = safe_preview(node)
        if preview:
            self.dicts.append((preview, node.lineno))
        self.generic_visit(node)


def safe_preview(node, max_len=160):
    try:
        text = ast.unparse(node)
        text = " ".join(text.split())
        if len(text) > max_len:
            text = text[:max_len] + " ..."
        return text
    except Exception:
        return None


def main():
    string_index = defaultdict(list)
    number_index = defaultdict(list)
    list_index = defaultdict(list)
    dict_index = defaultdict(list)
    assigned_constants = []

    for pyfile in iter_python_files(PROJECT_ROOT):
        try:
            source = pyfile.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(pyfile))
            collector = ConstantCollector(pyfile)
            collector.visit(tree)

            for value, line in collector.strings:
                string_index[value].append((str(pyfile), line))

            for value, line in collector.numbers:
                number_index[value].append((str(pyfile), line))

            for value, line in collector.lists:
                list_index[value].append((str(pyfile), line))

            for value, line in collector.dicts:
                dict_index[value].append((str(pyfile), line))

            assigned_constants.extend(collector.assigned_constants)

        except Exception as e:
            print(f"[WARN] Could not parse {pyfile}: {e}")

    print("\n" + "=" * 90)
    print("UPPER_CASE CONSTANT ASSIGNMENTS")
    print("=" * 90)
    for item in sorted(assigned_constants, key=lambda x: (x["file"], x["line"])):
        print(f'{item["file"]}:{item["line"]}  {item["name"]} = {item["preview"]}')

    print("\n" + "=" * 90)
    print("DUPLICATE STRING LITERALS")
    print("=" * 90)
    for value, refs in sorted(string_index.items(), key=lambda x: (-len(x[1]), x[0])):
        if len(refs) >= MIN_DUPLICATE_COUNT:
            print(f'\n"{value}"  [{len(refs)} hits]')
            for file, line in refs[:12]:
                print(f"  - {file}:{line}")

    print("\n" + "=" * 90)
    print("DUPLICATE NUMERIC LITERALS")
    print("=" * 90)
    for value, refs in sorted(number_index.items(), key=lambda x: (-len(x[1]), str(x[0]))):
        if len(refs) >= MIN_DUPLICATE_COUNT:
            print(f"\n{value}  [{len(refs)} hits]")
            for file, line in refs[:12]:
                print(f"  - {file}:{line}")

    print("\n" + "=" * 90)
    print("REPEATED LIST / TUPLE LITERALS")
    print("=" * 90)
    for value, refs in sorted(list_index.items(), key=lambda x: (-len(x[1]), x[0])):
        if len(refs) >= MIN_DUPLICATE_COUNT:
            print(f"\n{value}  [{len(refs)} hits]")
            for file, line in refs[:8]:
                print(f"  - {file}:{line}")

    print("\n" + "=" * 90)
    print("REPEATED DICT LITERALS")
    print("=" * 90)
    for value, refs in sorted(dict_index.items(), key=lambda x: (-len(x[1]), x[0])):
        if len(refs) >= MIN_DUPLICATE_COUNT:
            print(f"\n{value}  [{len(refs)} hits]")
            for file, line in refs[:8]:
                print(f"  - {file}:{line}")

    print("\n" + "=" * 90)
    print("TOP CANDIDATES TO CENTRALIZE")
    print("=" * 90)

    candidates = []

    for value, refs in string_index.items():
        if len(refs) >= MIN_DUPLICATE_COUNT:
            candidates.append(("STRING", len(refs), value))

    for value, refs in number_index.items():
        if len(refs) >= MIN_DUPLICATE_COUNT:
            candidates.append(("NUMBER", len(refs), value))

    for value, refs in list_index.items():
        if len(refs) >= MIN_DUPLICATE_COUNT:
            candidates.append(("LIST/TUPLE", len(refs), value))

    for value, refs in dict_index.items():
        if len(refs) >= MIN_DUPLICATE_COUNT:
            candidates.append(("DICT", len(refs), value))

    for kind, count, value in sorted(candidates, key=lambda x: (-x[1], x[0], str(x[2])))[:30]:
        print(f"[{kind}] ({count} hits) {value}")


if __name__ == "__main__":
    main()