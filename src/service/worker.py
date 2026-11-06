"""Inference worker: run the models on every new stable study in Orthanc and store the results.

Polls Orthanc's `/changes` for StableStudy events. For each study:

- DX/CR series go to the classifier with Grad-CAM.
- CT series go to the 3D U-Net.

Results go to Postgres, one row per (StudyInstanceUID, checkpoint), so a rerun adds nothing. The
last processed change sequence is stored too, so a restart resumes where it stopped.

A model error is stored as a row with status "error" and the worker moves on. Delete that row
to retry. An Orthanc or database error stops the poll without advancing, so the change is
retried on the next poll.

Checkpoints are read from mounted paths (CLASSIFIER_CHECKPOINT, SEGMENTATION_CHECKPOINT), never
from the image or git. CPU only. For education only, not for clinical use.

Run: python -m src.service.worker [--once]
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from src.service.analyzers import Analyzer, ClassifierAnalyzer, Reader, SegmentationAnalyzer
from src.service.orthanc import Orthanc, OrthancError, load_env
from src.service.store import PostgresStore, Result, Store

Log = Callable[[str], None]


class Worker:
    def __init__(
        self,
        orthanc: Reader,
        store: Store,
        analyzers: Sequence[Analyzer],
        results_dir: Path,
        log: Log = print,
    ) -> None:
        if not analyzers:
            raise ValueError("no model configured")
        self.orthanc, self.store, self.analyzers = orthanc, store, list(analyzers)
        self.results_dir, self.log = results_dir, log

    def poll_once(self, limit: int = 100) -> int:
        """Handle the next changes; returns how many were read."""
        changes = self.orthanc.get_json(f"/changes?since={self.store.last_seq()}&limit={limit}")
        for change in changes["Changes"]:
            if change["ChangeType"] == "StableStudy":
                self.process_study(change["ID"])
            self.store.set_last_seq(int(change["Seq"]))
        if changes["Changes"]:
            self.store.set_last_seq(int(changes["Last"]))
        return len(changes["Changes"])

    def process_study(self, orthanc_id: str) -> list[str]:
        """Run every analyzer whose modality is in the study; returns the outcomes."""
        try:
            study = self.orthanc.get_json(f"/studies/{orthanc_id}")
        except OrthancError as error:
            if error.status != 404:
                raise
            # Deleted since it became stable (e.g. scripts/reset_demo.py): nothing to do.
            self.log(f"study {orthanc_id} is no longer in Orthanc, skipped")
            return ["gone"]
        study_uid = str(study["MainDicomTags"]["StudyInstanceUID"])
        series = self.orthanc.get_json(f"/studies/{orthanc_id}/series")
        outcomes = []
        for analyzer in self.analyzers:
            matching = [
                s for s in series if s["MainDicomTags"].get("Modality") in analyzer.modalities
            ]
            if not matching:
                continue
            if self.store.has(study_uid, analyzer.checkpoint_sha256):
                self.log(f"{study_uid}: {analyzer.name} already done, skipped")
                outcomes.append("skipped")
                continue
            started = time.monotonic()
            try:
                out_dir = self.results_dir / study_uid / analyzer.name
                results = analyzer.analyze(self.orthanc, matching, out_dir, self.results_dir)
                status = "ok"
            except Exception as error:  # a bad study must not stop the worker
                results, status = {"error": f"{type(error).__name__}: {error}"}, "error"
            modalities = sorted({s["MainDicomTags"]["Modality"] for s in matching})
            added = self.store.insert(
                Result(
                    study_uid=study_uid,
                    orthanc_study_id=orthanc_id,
                    modality=",".join(modalities),
                    model=analyzer.name,
                    checkpoint_name=analyzer.checkpoint_name,
                    checkpoint_sha256=analyzer.checkpoint_sha256,
                    status=status,
                    results=results,
                    seconds=round(time.monotonic() - started, 3),
                )
            )
            self.log(f"{study_uid}: {analyzer.name} {status}, stored {added}")
            outcomes.append(status if added else "skipped")
        return outcomes

    def run(self, interval: float, heartbeat: Path | None = None) -> None:
        """Poll forever. After each poll, successful or not, touch `heartbeat` (if given): the
        Compose healthcheck reads its age, since the worker serves no HTTP."""
        while True:
            try:
                if self.poll_once() == 0:
                    time.sleep(interval)
            except Exception as error:  # Orthanc or the database is down: retry later
                self.log(f"poll failed: {type(error).__name__}: {error}")
                time.sleep(interval)
            finally:
                if heartbeat is not None:
                    heartbeat.touch()


def analyzers_from_env() -> list[Analyzer]:
    analyzers: list[Analyzer] = []
    classifier = os.environ.get("CLASSIFIER_CHECKPOINT")
    if classifier:
        analyzers.append(ClassifierAnalyzer.from_checkpoint(Path(classifier)))
    segmentation = os.environ.get("SEGMENTATION_CHECKPOINT")
    if segmentation:
        analyzers.append(SegmentationAnalyzer.from_checkpoint(Path(segmentation)))
    return analyzers


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="handle pending changes, then exit")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between polls")
    args = parser.parse_args(argv)
    load_env()
    torch.set_num_threads(int(os.environ.get("WORKER_THREADS", "4")))
    analyzers = analyzers_from_env()
    for analyzer in analyzers:
        print(f"{analyzer.name}: {analyzer.checkpoint_name} sha256 {analyzer.checkpoint_sha256}")
    url = os.environ.get("RESULTS_DB_URL")
    if not url:
        raise ValueError("set RESULTS_DB_URL")
    store = PostgresStore(url)
    results_dir = Path(os.environ.get("RESULTS_DIR", "results/worker"))
    worker = Worker(Orthanc.from_env(), store, analyzers, results_dir)
    if args.once:
        while worker.poll_once():
            pass
    else:
        heartbeat = os.environ.get("WORKER_HEARTBEAT")
        worker.run(args.interval, Path(heartbeat) if heartbeat else None)


if __name__ == "__main__":
    main()
