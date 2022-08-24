from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from generate_kaggle_script import ScriptGenImport
from generate_kaggle_script import ScriptGenImportException
from generate_kaggle_script import generate_kaggle_script


def test_with_no_imports():
    with TemporaryDirectory() as td:
        original_script_lines = ["x = 'hello!'", "", "print(x)"]
        main_script = Path(td, "main.py")
        with open(main_script, "w") as f:
            for line in original_script_lines:
                print(line, file=f)

        output = Path(td, "output.py")
        generate_kaggle_script(main_script, output)
        with open(output, "r") as f:
            output_lines = [line.rstrip("\n") for line in f.readlines()]

    assert output_lines == original_script_lines


def test_with_simple_import():
    with TemporaryDirectory() as td:
        imported = Path(td, "imported.py")
        with open(imported, "w") as f:
            for line in [
                "def imported_func():",
                "    return 'hi!'",
            ]:
                print(line, file=f)

        main_script = Path(td, "main.py")
        relative_path_to_imported = imported.relative_to(main_script.parent)
        script_gen_instruction = f" # script-gen: {relative_path_to_imported}"
        with open(main_script, "w") as f:
            for line in [
                "from imported import imported_func" + script_gen_instruction,
                "",
                "x = imported_func()",
            ]:
                print(line, file=f)

        output = Path(td, "output.py")
        generate_kaggle_script(main_script, output)
        with open(output, "r") as f:
            output_lines = [line.rstrip("\n") for line in f]

    expected_lines = [
        "def imported_func():",
        "    return 'hi!'",
        "",
        "x = imported_func()",
    ]

    assert output_lines == expected_lines


def test_detect_script_gen_instruction():
    negative_cases = (
        "import something",
        "from something import *",
        "",
    )
    positive_cases = (
        "from something import func  # script-gen: something.py",
        "from something import * # script-gen: something.py",
    )
    error_cases = ("import something # script-gen: something.py",)

    for case in negative_cases:
        assert ScriptGenImport.detect(case) is None

    for case in positive_cases:
        assert isinstance(ScriptGenImport.detect(case), ScriptGenImport)

    for case in error_cases:
        with pytest.raises(ScriptGenImportException):
            ScriptGenImport.detect(case)


def test_only_insert_a_module_once():
    with TemporaryDirectory() as td:
        imported = Path(td, "imported.py")
        with open(imported, "w") as f:
            for line in [
                "def imported_func():",
                "    return 'hi!'",
            ]:
                print(line, file=f)

        main_script = Path(td, "main.py")
        relative_path_to_imported = imported.relative_to(main_script.parent)
        script_gen_instruction = f" # script-gen: {relative_path_to_imported}"
        with open(main_script, "w") as f:
            for line in [
                "from imported import imported_func" + script_gen_instruction,
                "from imported import imported_func" + script_gen_instruction,
                "",
                "x = imported_func()",
            ]:
                print(line, file=f)

        output = Path(td, "output.py")
        generate_kaggle_script(main_script, output)
        with open(output, "r") as f:
            output_lines = [line.rstrip("\n") for line in f]

    expected_lines = [
        "def imported_func():",
        "    return 'hi!'",
        "",
        "x = imported_func()",
    ]

    assert output_lines == expected_lines


def test_nested_import():
    with TemporaryDirectory() as td:
        nested_import = Path(td, "nested_import.py")
        with open(nested_import, "w") as f:
            for line in [
                "def nested_import_func():",
                "    return 'bye!'",
            ]:
                print(line, file=f)

        imported = Path(td, "imported.py")
        relative_path_to_nested_import = nested_import.relative_to(
            imported.parent
        )
        nested_script_gen_instruction = (
            f"   # script-gen: {relative_path_to_nested_import}"
        )
        with open(imported, "w") as f:
            for line in [
                (
                    "from nested_import import nested_import_func"
                    + nested_script_gen_instruction
                ),
                "",
                "print(nested_import_func())",
                "def imported_func():",
                "    return 'hi!'",
            ]:
                print(line, file=f)

        main_script = Path(td, "main.py")
        relative_path_to_imported = imported.relative_to(main_script.parent)
        script_gen_instruction = f" # script-gen: {relative_path_to_imported}"
        with open(main_script, "w") as f:
            for line in [
                "from imported import imported_func" + script_gen_instruction,
                "",
                "x = imported_func()",
            ]:
                print(line, file=f)

        output = Path(td, "output.py")
        generate_kaggle_script(main_script, output)
        with open(output, "r") as f:
            output_lines = [line.rstrip("\n") for line in f]

    expected_lines = [
        "def nested_import_func():",
        "    return 'bye!'",
        "",
        "print(nested_import_func())",
        "def imported_func():",
        "    return 'hi!'",
        "",
        "x = imported_func()",
    ]

    assert output_lines == expected_lines
