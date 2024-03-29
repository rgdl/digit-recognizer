import re
import sys
from pathlib import Path
from typing import Iterable
from typing import Optional
from typing import Set


class ScriptGeneratorException(ValueError):
    pass


class ScriptGenerator:
    """
    Class to handle the insertion of lines from an imported file into the main
    script.

    Looks for an import statement followed by a comment of the form:
    "from module import item # script-gen: [/path/to/module.py]

    The contents of that file will be inserted in place of the import
    statement.
    """

    def __init__(self, main_script_path: Path, output_file_path: Path):
        self.main_script_path = main_script_path
        self.root_dir = main_script_path.parent
        self.output_file_path = output_file_path
        self.imported_modules: Set[Path] = set()
        self.in_name_eq_main_block = False  # No printing while this is True

    @staticmethod
    def detect_script_gen_tag(line: str) -> Optional[Path]:
        pattern = re.compile(r"(.*)# script-gen: (.+\.py)")
        match = re.match(pattern, line)
        if match is None:
            return None
        groups = match.groups()
        if "import" not in groups[0]:
            raise ScriptGeneratorException(
                f"Not an import statement: '{groups[0]}'"
            )
        if groups[0].startswith("import"):
            raise ScriptGeneratorException(
                "\n".join(
                    [
                        f"Cannot handle whole-module imports: '{groups[0]}'",
                        "Use 'from x import y' or 'from x import *'",
                    ]
                )
            )
        return Path(groups[1])

    def _insert_one_line(self, line: str) -> None:
        print(line, file=self._outfile, end="")

    def _is_name_eq_main_start(self, line: str) -> bool:
        return not self.in_name_eq_main_block and bool(
            re.search("""if __name__ == ['"]+__main__['"]+:""", line)
        )

    def _is_name_eq_main_end(self, line: str) -> bool:
        return self.in_name_eq_main_block and bool(re.search(r"^\S", line))

    def insert_lines(
        self,
        lines: Iterable[str],
        in_main_file: bool = True,
        header: Optional[str] = None,
        footer: Optional[str] = None,
    ) -> None:
        if header is not None:
            self._insert_one_line(header)

        for line in lines:
            if self._is_name_eq_main_start(line):
                self.in_name_eq_main_block = True
            elif self._is_name_eq_main_end(line):
                self.in_name_eq_main_block = False
            if self.in_name_eq_main_block and not in_main_file:
                continue

            file_to_import = self.detect_script_gen_tag(line)

            if file_to_import is not None:
                if file_to_import in self.imported_modules:
                    continue
                self.imported_modules.add(file_to_import)

                with open(self.root_dir / file_to_import, "r") as f:
                    self.insert_lines(
                        f,
                        in_main_file=False,
                        header=f"### Contents of '{file_to_import}' ###\n",
                        footer=f"### End of '{file_to_import}' ###\n",
                    )
                continue

            self._insert_one_line(line)

        if footer is not None:
            self._insert_one_line(footer)

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


if __name__ == "__main__":
    assert (
        len(sys.argv) > 2
    ), "Please provide an input file and output destination"
    ScriptGenerator(Path(sys.argv[1]), Path(sys.argv[2])).run()
