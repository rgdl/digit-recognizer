from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pytest

from generate_kaggle_script import ScriptGenerator
from generate_kaggle_script import ScriptGeneratorException


def _write_lines_to_file(lines: List[str], path: Path) -> None:
    with open(path, "w") as f:
        for line in lines:
            print(line, file=f)


def _build_tag(main_file: Path, imported_file: Path) -> str:
    """Build a tag indicating that an import needs to be in-lined"""
    relative_path_to_imported = imported_file.relative_to(main_file.parent)
    return f" # script-gen: {relative_path_to_imported}"


def _get_file_lines(file: Path) -> List[str]:
    with open(file, "r") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def test_with_no_imports():
    with TemporaryDirectory() as td:
        main = Path(td, "main.py")
        output = Path(td, "output.py")

        original_script_lines = ["x = 'hello!'", "", "print(x)"]
        _write_lines_to_file(original_script_lines, main)

        ScriptGenerator(main, output).run()
        assert _get_file_lines(output) == original_script_lines


def test_with_simple_import():
    with TemporaryDirectory() as td:
        imported = Path(td, "imported.py")
        main = Path(td, "main.py")
        output = Path(td, "output.py")

        _write_lines_to_file(["def func():", "    return 'hi!'"], imported)
        _write_lines_to_file(
            [
                "from imported import func" + _build_tag(main, imported),
                "",
                "x = func()",
            ],
            main,
        )

        ScriptGenerator(main, output).run()
        assert _get_file_lines(output) == [
            "### Contents of 'imported.py' ###",
            "def func():",
            "    return 'hi!'",
            "### End of 'imported.py' ###",
            "",
            "x = func()",
        ]


def test_detect_script_gen_instruction():
    negative_cases = ("import something", "from something import *", "")
    positive_cases = (
        "from something import func  # script-gen: something.py",
        "from something import * # script-gen: something.py",
    )
    error_cases = ("import something # script-gen: something.py",)

    for case in negative_cases:
        assert ScriptGenerator.detect_script_gen_tag(case) is None

    for case in positive_cases:
        assert ScriptGenerator.detect_script_gen_tag(case) is not None

    for case in error_cases:
        with pytest.raises(ScriptGeneratorException):
            ScriptGenerator.detect_script_gen_tag(case)


def test_only_insert_a_module_once():
    with TemporaryDirectory() as td:
        imported = Path(td, "imported.py")
        main = Path(td, "main.py")
        output = Path(td, "output.py")
        import_tag = _build_tag(main, imported)

        _write_lines_to_file(
            ["def imported_func():", "    return 'hi!'"],
            imported,
        )
        _write_lines_to_file(
            [
                "from imported import imported_func" + import_tag,
                "from imported import imported_func" + import_tag,
                "",
                "x = imported_func()",
            ],
            main,
        )

        ScriptGenerator(main, output).run()
        assert _get_file_lines(output) == [
            "### Contents of 'imported.py' ###",
            "def imported_func():",
            "    return 'hi!'",
            "### End of 'imported.py' ###",
            "",
            "x = imported_func()",
        ]


def test_nested_import():
    with TemporaryDirectory() as td:
        nested_import = Path(td, "nested_import.py")
        imported = Path(td, "imported.py")
        main = Path(td, "main.py")
        output = Path(td, "output.py")

        _write_lines_to_file(
            ["def nested_import_func():", "    return 'bye!'"],
            nested_import,
        )
        _write_lines_to_file(
            [
                (
                    "from nested_import import nested_import_func"
                    + _build_tag(imported, nested_import)
                ),
                "",
                "print(nested_import_func())",
                "def imported_func():",
                "    return 'hi!'",
            ],
            imported,
        )
        _write_lines_to_file(
            [
                (
                    "from imported import imported_func"
                    + _build_tag(main, imported)
                ),
                "",
                "x = imported_func()",
            ],
            main,
        )
        ScriptGenerator(main, output).run()
        assert _get_file_lines(output) == [
            "### Contents of 'imported.py' ###",
            "### Contents of 'nested_import.py' ###",
            "def nested_import_func():",
            "    return 'bye!'",
            "### End of 'nested_import.py' ###",
            "",
            "print(nested_import_func())",
            "def imported_func():",
            "    return 'hi!'",
            "### End of 'imported.py' ###",
            "",
            "x = imported_func()",
        ]
