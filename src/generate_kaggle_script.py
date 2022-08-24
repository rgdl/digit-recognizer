import re
from pathlib import Path
from typing import Generator
from typing import Iterable
from typing import Optional
from typing import Set
from typing import TextIO


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
    def detect(line: str) -> Optional[Path]:
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
        return Path(groups[1])

    def read(self) -> Generator[str, None, None]:
        with open(self.file_to_import, "r") as f:
            for line in f:
                yield line


class ScriptGenerator:
    def __init__(self, main_script_path: Path, output_file_path: Path):
        self.main_script_path = main_script_path
        self.root_dir = main_script_path.parent
        self.output_file_path = output_file_path
        self.imported_modules = set()

    def insert_lines(self, lines: Iterable[str]) -> None:
        for line in lines:
            file_to_import = ScriptGenImport.detect(line)
            if file_to_import:
                script_gen_import = ScriptGenImport(self.root_dir / file_to_import)
                if file_to_import in self.imported_modules:
                    continue
                self.imported_modules.add(file_to_import)
                self.insert_lines(script_gen_import.read())
                continue
            print(line, file=self._outfile, end="")


    def run(self) -> None:
        """
        Insert the contents of all imported local modules, so that a script
        contained in a single file can be uploaded and run on a Kaggle kernel
        """
        self._infile = open(self.main_script_path, "r")
        self._outfile = open(self.output_file_path, "w")
        try:
            self.insert_lines(self._infile)
        finally:
            self._infile.close()
            self._outfile.close()
