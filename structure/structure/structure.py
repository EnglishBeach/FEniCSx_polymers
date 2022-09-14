# Импорты стандартной библиотеки
import pathlib
import sys
# Локальные импорты
from . import files
def main():
    # Считывание пути из командной строки
    try:
        root = pathlib.Path(sys.argv[1]).resolve()
    except IndexError:
        print("Need one argument: the root of the original file tree")
        raise SystemExit()
    # Воссоздание файловой структуры
    new_root = files.unique_path(pathlib.Path.cwd(), "{:03d}")
    for path in root.rglob("*"):
        if path.is_file() and new_root not in path.parents:
            rel_path = path.relative_to(root)
            files.add_empty_file(new_root / rel_path)
if __name__ == "__main__":
    main()