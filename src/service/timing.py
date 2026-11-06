"""Time the worker's models on the demo studies: several runs, the first ones dropped as warm-up.

Runs inside the worker container, so it uses the worker's image, checkpoints and threads:
    docker compose exec worker python -m src.service.timing --runs 6 --warm-up 1
It reads the demo studies from Orthanc (found by their StudyInstanceUIDs, src/service/stack.py)
and writes the model outputs to a temporary folder that is deleted; nothing goes to the results
database or the results folder. What is timed is the same as the worker's `seconds` column:
download from Orthanc, inference, writing the outputs. CPU only, so the times depend on the
machine and on what else it is doing; the median and the range of the warm runs are reported.
For education only, not for clinical use.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.service.orthanc import Orthanc, load_env
from src.service.stack import demo_study_uids


def summarise(seconds: Sequence[float], warm_up: int) -> dict[str, Any]:
    """Median and range of the runs after the first `warm_up` ones."""
    warm = list(seconds[warm_up:])
    if not warm:
        raise ValueError(f"{len(seconds)} runs leave none after {warm_up} warm-up runs")
    return {
        "runs": [round(s, 2) for s in seconds],
        "warm_runs": len(warm),
        "median": round(statistics.median(warm), 1),
        "min": round(min(warm), 1),
        "max": round(max(warm), 1),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--warm-up", type=int, default=1, help="first runs to leave out")
    args = parser.parse_args(argv)
    load_env()
    import torch

    from src.service.worker import analyzers_from_env

    threads = int(os.environ.get("WORKER_THREADS", "4"))
    torch.set_num_threads(threads)
    orthanc = Orthanc.from_env()
    analyzers = analyzers_from_env()
    report: dict[str, Any] = {"threads": threads, "cpus": os.cpu_count(), "studies": {}}
    for uid in sorted(demo_study_uids()):
        [study_id] = orthanc.find_study(uid)
        study = orthanc.get_json(f"/studies/{study_id}")
        name = f"{study['PatientMainDicomTags'].get('PatientID', '')} ({uid[-12:]})"
        series = orthanc.get_json(f"/studies/{study_id}/series")
        for analyzer in analyzers:
            matching = [s for s in series if s["MainDicomTags"]["Modality"] in analyzer.modalities]
            if not matching:
                continue
            seconds = []
            for run in range(args.runs):
                with tempfile.TemporaryDirectory() as folder:
                    started = time.monotonic()
                    analyzer.analyze(orthanc, matching, Path(folder) / "out", Path(folder))
                    seconds.append(time.monotonic() - started)
                print(f"{name} {analyzer.name} run {run + 1}: {seconds[-1]:.1f} s", flush=True)
            report["studies"][f"{name} {analyzer.name}"] = summarise(seconds, args.warm_up)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
