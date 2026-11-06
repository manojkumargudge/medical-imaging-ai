"""Write the viewer API's OpenAPI document to docs/openapi.json (no services needed).

tests/test_step7.py checks that the file matches the code, so run this after changing the API:
    python -m scripts.export_openapi
For education only, not for clinical use.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.api.app import create_app

OUT = Path("docs/openapi.json")


def document() -> str:
    return json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    OUT.write_text(document(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
