"""Command line interface."""

from __future__ import annotations

import json

from conftest import requires_rdkit

from formulate.cli.main import main


def test_experts_command_lists_the_panel(capsys):
    assert main(["experts"]) == 0
    out = capsys.readouterr().out
    assert "joback" in out and "interfacial" in out


def test_properties_command_lists_the_registry(capsys):
    assert main(["properties"]) == 0
    out = capsys.readouterr().out
    assert "normal_boiling_point" in out and "thermal" in out


def test_example_command_emits_a_loadable_spec(capsys, tmp_path):
    assert main(["example"]) == 0
    text = capsys.readouterr().out

    path = tmp_path / "spec.yaml"
    path.write_text(text, encoding="utf-8")

    from formulate.targets.spec import TargetSpec

    spec = TargetSpec.from_file(path)
    assert spec.requirements
    assert spec.assumptions


def test_a_missing_specification_is_an_error_not_a_crash(capsys):
    assert main(["run", "/nonexistent/spec.yaml"]) == 2
    assert "no such specification" in capsys.readouterr().err


def test_a_malformed_specification_is_reported(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("requirements:\n  - property: not_a_property\n", encoding="utf-8")
    assert main(["run", str(path)]) == 2
    err = capsys.readouterr().err
    assert "could not read" in err
    assert "not_a_property" in err


@requires_rdkit
def test_run_command_prints_a_report(tmp_path, capsys):
    path = tmp_path / "spec.yaml"
    path.write_text(
        """
name: cli test
conditions: {temperature: 25 degC, pressure: 1 atm}
requirements:
  - property: normal_boiling_point
    direction: in_range
    lower: 60 degC
    upper: 160 degC
    hard: true
""",
        encoding="utf-8",
    )
    assert main(["run", str(path), "--top", "2", "--pool", "20"]) == 0
    out = capsys.readouterr().out
    assert "FORMULATE DESIGN RUN" in out
    assert "WHAT THIS RUN DOES NOT ESTABLISH" in out


@requires_rdkit
def test_run_command_emits_json(tmp_path, capsys):
    path = tmp_path / "spec.yaml"
    path.write_text(
        """
requirements:
  - property: logp
    direction: in_range
    lower: 0
    upper: 4
""",
        encoding="utf-8",
    )
    main(["run", str(path), "--json", "--pool", "10"])
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == "formulate/design-run/1"


@requires_rdkit
def test_an_infeasible_run_exits_nonzero(tmp_path, capsys):
    path = tmp_path / "spec.yaml"
    path.write_text(
        """
requirements:
  - property: normal_boiling_point
    direction: in_range
    lower: 900 degC
    upper: 1000 degC
    hard: true
""",
        encoding="utf-8",
    )
    assert main(["run", str(path), "--pool", "20"]) == 1


@requires_rdkit
def test_run_command_saves_a_record(tmp_path, capsys):
    path = tmp_path / "spec.yaml"
    path.write_text(
        """
requirements:
  - property: logp
    direction: in_range
    lower: 0
    upper: 4
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "runs"
    main(["run", str(path), "--pool", "10", "--save", str(out_dir)])
    assert list(out_dir.glob("*.json"))


@requires_rdkit
def test_calibrate_command_emits_json(capsys):
    assert main(["calibrate", "--json"]) == 0
    results = json.loads(capsys.readouterr().out)
    assert "normal_boiling_point" in results
    assert results["normal_boiling_point"]["count"] > 0
