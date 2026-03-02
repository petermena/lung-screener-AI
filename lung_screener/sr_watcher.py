"""Background watcher for radiologist-verified DICOM Structured Reports.

Monitors the PACS incoming storage directory for new DICOM files.
When a verified SR arrives, parses its feedback records and stores them
in the FeedbackStore. Optionally triggers incremental retraining when
enough feedback has accumulated.

Usage:
    lung-screener watch
    lung-screener watch --watch-dir ./data/incoming --feedback-dir ./data/feedback
    lung-screener watch --model best.pth --auto-retrain --retrain-threshold 20
"""

import json
import logging
import signal
import time
from pathlib import Path

import pydicom

from .feedback import FeedbackStore, parse_feedback_sr

logger = logging.getLogger(__name__)


class SRWatcher:
    """Polls a directory for verified DICOM SRs and records radiologist feedback.

    Maintains a persistent set of already-processed file paths so that
    restarting the watcher never double-counts records.

    Args:
        watch_dir:          Directory (recursively searched) for incoming .dcm files.
        feedback_dir:       Directory where FeedbackStore writes feedback.jsonl.
        poll_interval:      Seconds between directory scans.
        auto_retrain:       If True, triggers IncrementalRetrainer when the
                            number of new records since the last retrain reaches
                            retrain_threshold.
        retrain_threshold:  Minimum new feedback records required to auto-retrain.
        model_path:         Path to current best .pth checkpoint (required for
                            auto-retrain). Updated in-place if a new model is
                            promoted.
        checkpoint_dir:     Directory where retrain checkpoints are saved.
        config:             Full application config dict (passed to IncrementalRetrainer).
    """

    def __init__(
        self,
        watch_dir: str | Path,
        feedback_dir: str | Path,
        poll_interval: int = 30,
        auto_retrain: bool = False,
        retrain_threshold: int = 20,
        model_path: str | Path | None = None,
        checkpoint_dir: str | Path = "./checkpoints",
        config: dict | None = None,
    ):
        self.watch_dir = Path(watch_dir)
        self.feedback_dir = Path(feedback_dir)
        self.poll_interval = poll_interval
        self.auto_retrain = auto_retrain
        self.retrain_threshold = retrain_threshold
        self.model_path = Path(model_path) if model_path else None
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config or {}

        self.store = FeedbackStore(self.feedback_dir)

        # Persist seen-file state across restarts
        self._seen_file = self.feedback_dir / "watched_files.json"
        self._seen: set[str] = self._load_seen()

        self._running = False
        self._records_since_retrain = 0

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_seen(self) -> set[str]:
        if self._seen_file.exists():
            with open(self._seen_file) as f:
                return set(json.load(f))
        return set()

    def _save_seen(self):
        with open(self._seen_file, "w") as f:
            json.dump(sorted(self._seen), f, indent=2)

    # ------------------------------------------------------------------
    # Core scan
    # ------------------------------------------------------------------

    def _scan_once(self) -> int:
        """Scan watch_dir for new verified SRs.

        Returns:
            Number of new FeedbackRecords stored in this scan.
        """
        new_records = 0

        for dcm_path in sorted(self.watch_dir.rglob("*.dcm")):
            key = str(dcm_path.resolve())

            if key in self._seen:
                continue

            # Quick header-only read to check modality before full load
            try:
                ds_header = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            except Exception as e:
                logger.debug(f"Skipping unreadable file {dcm_path.name}: {e}")
                self._seen.add(key)
                continue

            # Skip non-SR files immediately and don't recheck them
            if getattr(ds_header, "Modality", "") != "SR":
                self._seen.add(key)
                continue

            # Leave unverified SRs out of _seen so we check them again next poll
            if getattr(ds_header, "VerificationFlag", "") != "VERIFIED":
                logger.debug(f"Unverified SR — will recheck next poll: {dcm_path.name}")
                continue

            # Full read for SR parsing
            try:
                ds_full = pydicom.dcmread(str(dcm_path))
            except Exception as e:
                logger.warning(f"Could not fully read SR {dcm_path.name}: {e}")
                self._seen.add(key)
                continue

            records = parse_feedback_sr(ds_full)
            if records:
                for record in records:
                    self.store.add_record(record)
                new_records += len(records)

                confirmed = sum(1 for r in records if r.confirmed)
                rejected = len(records) - confirmed
                patient = getattr(ds_full, "PatientID", "unknown")
                logger.info(
                    f"[SR] {dcm_path.name} — patient {patient}: "
                    f"{confirmed} confirmed, {rejected} rejected"
                )
            else:
                logger.debug(
                    f"[SR] {dcm_path.name}: verified but no parseable findings"
                )

            self._seen.add(key)

        self._save_seen()
        return new_records

    # ------------------------------------------------------------------
    # Auto-retrain
    # ------------------------------------------------------------------

    def _maybe_retrain(self):
        """Trigger incremental retrain if auto-retrain is enabled and threshold is met."""
        if not self.auto_retrain or self.model_path is None:
            return

        if self._records_since_retrain < self.retrain_threshold:
            return

        stats = self.store.get_stats()
        logger.info(
            f"Auto-retrain threshold reached "
            f"({self._records_since_retrain} new records, "
            f"{stats['total_feedback']} total). Starting retrain..."
        )
        self._records_since_retrain = 0
        self._run_retrain()

    def _run_retrain(self):
        from .retrain import IncrementalRetrainer

        retrainer = IncrementalRetrainer(
            config=self.config,
            base_checkpoint=self.model_path,
            feedback_dir=self.feedback_dir,
            checkpoint_dir=self.checkpoint_dir,
        )
        result = retrainer.retrain()
        logger.info(result.summary())

        # Track the promoted model path for future retrains
        if result.promoted:
            promoted_path = self.checkpoint_dir / "best.pth"
            if promoted_path.exists():
                self.model_path = promoted_path
                logger.info(f"Active model updated to: {self.model_path}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Start the polling loop. Blocks until SIGINT/SIGTERM or stop() is called."""
        self._running = True
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        def _handle_signal(signum, frame):
            logger.info("Received stop signal — shutting down watcher.")
            self._running = False

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        logger.info(f"SR watcher started — monitoring: {self.watch_dir}")
        logger.info(f"  Poll interval : {self.poll_interval}s")
        logger.info(f"  Feedback dir  : {self.feedback_dir}")
        logger.info(f"  Files seen    : {len(self._seen)} (from prior runs)")
        if self.auto_retrain:
            logger.info(
                f"  Auto-retrain  : ON  (threshold={self.retrain_threshold}, "
                f"model={self.model_path})"
            )
        else:
            logger.info("  Auto-retrain  : OFF")

        while self._running:
            try:
                new_records = self._scan_once()
                if new_records:
                    self._records_since_retrain += new_records
                    stats = self.store.get_stats()
                    logger.info(
                        f"Stored {new_records} new record(s). "
                        f"Total: {stats['total_feedback']} "
                        f"({stats['confirmed']} confirmed / "
                        f"{stats['rejected']} rejected, "
                        f"precision={stats['precision']:.1%})"
                    )
                    self._maybe_retrain()
            except Exception as e:
                logger.error(f"Watcher scan error: {e}", exc_info=True)

            # Sleep in 1-second ticks so SIGINT is responsive
            for _ in range(self.poll_interval):
                if not self._running:
                    break
                time.sleep(1)

        logger.info("SR watcher stopped.")

    def stop(self):
        """Programmatically stop the polling loop."""
        self._running = False
