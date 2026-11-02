from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_starts_with_education_only_notice() -> None:
    lines = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    assert "For education only, not for clinical use" in lines[0]
