import re
from pathlib import Path
from typing import Generator
from typing import Iterable
from typing import Set
from typing import TextIO
from typing import Union


class ScriptGenImportException(ValueError):
    pass


class ScriptGenImport:
    """
    Class to handle the insertion of lines from an imported file into the main
    script. Pass each line of the main script to `detect`. If an instance of
    `ScriptGenImport` is returned, call its `read` method to get the contents
    of the file being extracted.

    `detect` looks for an import statement followed by a comment of the form:
    "from module import item # script-gen: [/path/to/module.py]

    The contents of that file will be inserted in place of the import
    statement.
    """

    def __init__(self, file_to_import: Path) -> None:
        self.file_to_import = file_to_import

    @staticmethod
    def detect(line: str) -> Union[None, "ScriptGenImport"]:
        pattern = re.compile(r"(.*)# script-gen: (.+\.py)")
        match = re.match(pattern, line)
        if match is None:
            return None
        groups = match.groups()
        if "import" not in groups[0]:
            raise ScriptGenImportException(
                f"Not an import statement: '{groups[0]}'"
            )
        if groups[0].startswith("import"):
            raise ScriptGenImportException(
                "\n".join(
                    [
                        f"Cannot handle whole-module imports: '{groups[0]}'",
                        "Use 'from x import y' or 'from x import *'",
                    ]
                )
            )
        return ScriptGenImport(file_to_import=Path(groups[1]))

    def read(self, main_script_dir: Path) -> Generator[str, None, None]:
        with open(main_script_dir / self.file_to_import, "r") as f:
            for line in f:
                yield line


def insert_lines_into_file(
    lines: Iterable[str],
    file: TextIO,
    imported_modules: Set[str],
    main_script_dir: Path,
) -> None:
    for line in lines:
        script_gen_import = ScriptGenImport.detect(line)
        if script_gen_import:
            if script_gen_import.file_to_import in imported_modules:
                continue
            imported_modules.add(script_gen_import.file_to_import)
            insert_lines_into_file(
                script_gen_import.read(main_script_dir),
                file,
                imported_modules,
                main_script_dir
            )
            continue
        print(line, file=file, end="")


def generate_kaggle_script(main_script: Path, output_file: Path) -> None:
    """
    Insert the contents of all imported local modules, so that a script
    contained in a single file can be uploaded and run on a Kaggle kernel
    """
    imported_modules = set()
    with open(main_script, "r") as infile, open(output_file, "w") as outfile:
        insert_lines_into_file(
            infile,
            outfile,
            imported_modules,
            main_script.parent,
        )
