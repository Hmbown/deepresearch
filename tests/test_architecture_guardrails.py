from pathlib import Path


def test_no_legacy_graph_patterns_in_source():
    """Ensure deprecated runtime patterns do not return."""
    forbidden_tokens = [
        "IntakeFirstApp",
        "@entrypoint",
        "Send(",
        "evaluate_wave",
        "classify_and_plan",
        "WAVE_EVALUATION_PROMPT",
        "CLASSIFIER_SYSTEM_PROMPT",
    ]
    for path in Path("src/deepresearch").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"Found legacy token '{token}' in {path}"


def test_no_machine_specific_paths_in_source_and_docs():
    forbidden_tokens = [
        "/Volumes/",
        "VIXinSSD",
    ]
    targets = [
        Path("src/deepresearch"),
        Path("docs"),
        Path("CLAUDE.md"),
        Path("agents.md"),
        Path("README.md"),
    ]
    files: list[Path] = []
    for target in targets:
        if target.is_dir():
            files.extend(path for path in target.rglob("*") if path.suffix in {".py", ".md"})
        else:
            files.append(target)
    for path in files:
        if path.is_dir():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"Found machine-specific token '{token}' in {path}"
