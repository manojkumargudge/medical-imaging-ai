"""Download the two trained checkpoints from the GitHub release and check their sha256.

    python -m scripts.fetch_checkpoints            # paths from .env
    python -m scripts.fetch_checkpoints --env-file .env.other

Each file goes to the path set in the env file (CLASSIFIER_CHECKPOINT, SEGMENTATION_CHECKPOINT;
.env.example uses C:/aiml02-checkpoints/). A file that is already there is kept if its sha256
matches; if it does not match, the script stops and changes nothing. A download is written to a
temporary name next to the target and renamed only after its sha256 matches.

Release v1.0-models. The classifier was trained on NIH ChestX-ray14 (cite Wang X, et al.,
CVPR 2017; acknowledge the NIH Clinical Center). The U-Net was trained on Medical Segmentation
Decathlon Task09 Spleen (CC BY-SA 4.0), so its weights are released under CC BY-SA 4.0 with
attribution. For education only, not for clinical use.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.service.orthanc import read_env

RELEASE = "https://github.com/manojkumargudge/medical-imaging-ai/releases/download/v1.0-models/"
CHUNK = 1 << 20


@dataclass(frozen=True)
class Checkpoint:
    setting: str  # the env file key holding the local path
    name: str  # the release asset
    sha256: str
    size: int


CHECKPOINTS = [
    Checkpoint(
        "CLASSIFIER_CHECKPOINT",
        "epoch_003.pt",
        "1a35108eeeac9726b9f9d718e737c9d42a7325b40fcbdc9e08d46de96b501708",
        84_560_522,
    ),
    Checkpoint(
        "SEGMENTATION_CHECKPOINT",
        "epoch_180.pt",
        "8839d5379168dd6c64e869356a86eeed72cfc9bb7409d3386d721096ac26cc36",
        57_874_023,
    ),
]

Download = Callable[[str, Path], None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "aiml02-fetch"})
    with urllib.request.urlopen(request, timeout=600) as response, path.open("wb") as out:  # noqa: S310
        while chunk := response.read(CHUNK):
            out.write(chunk)


def fetch(item: Checkpoint, target: Path, get: Download = download) -> str:
    """Returns "kept" or "downloaded"; raises SystemExit on a mismatch, changing nothing."""
    if target.is_dir():
        raise SystemExit(
            f"{target} is a folder, not a file; not touching it. Docker creates an empty folder "
            "there if the stack was started before the checkpoints were fetched: check it is "
            "empty, remove it yourself, and rerun."
        )
    if target.exists():
        found = sha256_file(target)
        if found != item.sha256:
            raise SystemExit(
                f"{target} exists with sha256 {found}, expected {item.sha256}; not overwritten"
            )
        return "kept"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if partial.exists():
        raise SystemExit(f"{partial} exists from an earlier run; remove it yourself and rerun")
    get(RELEASE + item.name, partial)
    found = sha256_file(partial)
    if found != item.sha256:
        raise SystemExit(
            f"{item.name}: downloaded sha256 {found}, expected {item.sha256}; "
            f"left as {partial}, not used"
        )
    partial.rename(target)
    return "downloaded"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    values = {**read_env(args.env_file), **os.environ}
    for item in CHECKPOINTS:
        if not values.get(item.setting):
            raise SystemExit(f"set {item.setting} in {args.env_file}")
        target = Path(values[item.setting])
        outcome = fetch(item, target)
        print(f"{outcome} {target} ({item.size} bytes, sha256 {item.sha256[:12]}... ok)")


if __name__ == "__main__":
    main()
