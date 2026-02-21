"""Command-line interface for Lung Screener AI.

Provides commands for training, inference, PACS server management,
batch processing, feedback, risk scoring, active learning, calibration,
and metrics visualization.
"""

import json
import logging
import sys
from pathlib import Path

import click
import yaml


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from YAML file."""
    default_path = Path(__file__).parent.parent / "config" / "default.yaml"

    config = {}
    if default_path.exists():
        with open(default_path) as f:
            config = yaml.safe_load(f)

    if config_path:
        with open(config_path) as f:
            override = yaml.safe_load(f)
            _deep_merge(config, override)

    return config


def _deep_merge(base: dict, override: dict):
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config YAML file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.pass_context
def main(ctx, config, verbose):
    """Lung Screener AI - Lung nodule detection for CT scans."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@main.command()
@click.option("--epochs", type=int, help="Override number of training epochs")
@click.option("--batch-size", type=int, help="Override batch size")
@click.option("--lr", type=float, help="Override learning rate")
@click.option("--resume", type=click.Path(exists=True), help="Resume from checkpoint")
@click.option("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
@click.option(
    "--dataset",
    multiple=True,
    type=(click.Choice(["luna16", "luna25"]), click.Path(exists=True)),
    help="Add a dataset: --dataset luna25 ./data/luna25 (repeatable)",
)
@click.pass_context
def train(ctx, epochs, batch_size, lr, resume, checkpoint_dir, dataset):
    """Train the nodule detection model.

    By default trains on LUNA16 data.  To include additional datasets
    (e.g. LUNA25) pass one or more --dataset flags:

    \b
        lung-screener train --dataset luna25 ./data/luna25
        lung-screener train --dataset luna16 ./data/luna16 --dataset luna25 ./data/luna25
    """
    from .train import Trainer

    config = ctx.obj["config"]

    # Apply CLI overrides
    if epochs:
        config.setdefault("training", {})["epochs"] = epochs
    if batch_size:
        config.setdefault("training", {})["batch_size"] = batch_size
    if lr:
        config.setdefault("training", {})["learning_rate"] = lr

    # Build data.datasets list from --dataset flags (overrides YAML)
    if dataset:
        config.setdefault("data", {})["datasets"] = [
            {"type": kind, "dataset_dir": path} for kind, path in dataset
        ]

    trainer = Trainer(config, checkpoint_dir=checkpoint_dir)
    trainer.train(resume_from=resume)


@main.command(name="evaluate")
@click.option("--checkpoint", "-m", type=click.Path(exists=True), required=True, help="Model checkpoint")
@click.option("--output", "-o", type=click.Path(), help="Output JSON results file")
@click.option("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory for metrics")
@click.pass_context
def evaluate_cmd(ctx, checkpoint, output, checkpoint_dir):
    """Evaluate model on the validation set with comprehensive metrics.

    Produces AUC-ROC (with 95% CI), sensitivity, specificity, precision,
    F1, ECE, operating point table, and FROC sensitivity.

    \b
    Examples:
        lung-screener evaluate -m checkpoints/best.pth
        lung-screener evaluate -m checkpoints/best.pth -o eval_results.json
    """
    from .evaluate import evaluate, format_report

    config = ctx.obj["config"]
    results = evaluate(config, checkpoint, output_path=output)
    click.echo(format_report(results))


@main.command(name="train-kfold")
@click.option("--folds", "-k", type=int, default=5, help="Number of folds")
@click.option("--epochs", type=int, help="Override number of training epochs")
@click.option("--batch-size", type=int, help="Override batch size")
@click.option("--lr", type=float, help="Override learning rate")
@click.option("--checkpoint-dir", default="./checkpoints", help="Root checkpoint directory")
@click.option(
    "--dataset",
    multiple=True,
    type=(click.Choice(["luna16", "luna25"]), click.Path(exists=True)),
    help="Add a dataset (repeatable)",
)
@click.pass_context
def train_kfold_cmd(ctx, folds, epochs, batch_size, lr, checkpoint_dir, dataset):
    """Run k-fold cross-validation training.

    Trains K independent models, each validated on a different fold
    of the data.  Fold checkpoints are saved under the checkpoint
    directory as fold_0/, fold_1/, etc.

    Fold models can later be ensembled for the best possible inference.

    \b
    Examples:
        lung-screener train-kfold -k 5
        lung-screener train-kfold -k 5 --dataset luna16 ./data/luna16 --dataset luna25 ./data/luna25
    """
    from .train import train_kfold

    config = ctx.obj["config"]
    if epochs:
        config.setdefault("training", {})["epochs"] = epochs
    if batch_size:
        config.setdefault("training", {})["batch_size"] = batch_size
    if lr:
        config.setdefault("training", {})["learning_rate"] = lr
    if dataset:
        config.setdefault("data", {})["datasets"] = [
            {"type": kind, "dataset_dir": path} for kind, path in dataset
        ]

    summary = train_kfold(config, n_folds=folds, checkpoint_dir=checkpoint_dir)

    click.echo("")
    click.echo(f"K-Fold Training Complete ({summary['n_folds']} folds)")
    click.echo(f"  Mean AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    click.echo(f"  Range:    {summary['min_auc']:.4f} - {summary['max_auc']:.4f}")
    click.echo(f"\nFold checkpoints saved under {checkpoint_dir}/fold_*/best.pth")
    click.echo("Use 'lung-screener ensemble-predict' to run ensemble inference.")


@main.command(name="evaluate-kfold")
@click.option("--folds", "-k", type=int, default=5, help="Number of folds")
@click.option("--checkpoint-dir", default="./checkpoints", help="Root checkpoint directory")
@click.option("--output", "-o", type=click.Path(), help="Output JSON results file")
@click.pass_context
def evaluate_kfold_cmd(ctx, folds, checkpoint_dir, output):
    """Evaluate all k-fold models and aggregate metrics.

    \b
    Example:
        lung-screener evaluate-kfold -k 5 --checkpoint-dir ./checkpoints
    """
    from .evaluate import evaluate_kfold

    config = ctx.obj["config"]
    results = evaluate_kfold(config, checkpoint_dir, n_folds=folds, output_path=output)

    agg = results.get("aggregated_metrics", {})
    click.echo(f"\nK-Fold Evaluation ({results.get('n_folds', 0)} folds)")
    click.echo(f"  AUC:         {agg.get('auc_roc_mean', 0):.4f} ± {agg.get('auc_roc_std', 0):.4f}")
    click.echo(f"  Sensitivity: {agg.get('sensitivity_mean', 0):.4f} ± {agg.get('sensitivity_std', 0):.4f}")
    click.echo(f"  Specificity: {agg.get('specificity_mean', 0):.4f} ± {agg.get('specificity_std', 0):.4f}")
    click.echo(f"  F1:          {agg.get('f1_score_mean', 0):.4f} ± {agg.get('f1_score_std', 0):.4f}")


@main.command(name="ensemble-predict")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--models", "-m", type=click.Path(exists=True), multiple=True, required=True,
              help="Model checkpoint paths (repeat for each model)")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
@click.option("--format", "output_format", type=click.Choice(["json", "text", "report"]), default="text")
@click.pass_context
def ensemble_predict(ctx, input_path, models, output, output_format):
    """Run ensemble inference with multiple models.

    Averages predictions from multiple checkpoints (e.g. k-fold models)
    for more robust and accurate detection.

    \b
    Examples:
        lung-screener ensemble-predict /scan -m fold_0/best.pth -m fold_1/best.pth -m fold_2/best.pth
        lung-screener ensemble-predict /scan -m best.pth -m swa_best.pth
    """
    from .inference import EnsembleDetector
    from .preprocessing import load_dicom_series, load_mhd

    config = ctx.obj["config"]
    input_path = Path(input_path)

    detector = EnsembleDetector(config, model_paths=list(models))

    if input_path.suffix in (".mhd", ".mha"):
        image = load_mhd(input_path)
        series_uid = input_path.stem
    elif input_path.is_dir():
        image = load_dicom_series(input_path)
        series_uid = input_path.name
    else:
        click.echo(f"Unsupported input: {input_path}", err=True)
        sys.exit(1)

    result = detector.predict_scan(image, series_uid=series_uid)

    if output_format == "json":
        result_dict = result.to_dict()
        if output:
            with open(output, "w") as f:
                json.dump(result_dict, f, indent=2)
            click.echo(f"Results written to {output}")
        else:
            click.echo(json.dumps(result_dict, indent=2))
    elif output_format == "report":
        report_text = result.dictation()
        if output:
            with open(output, "w") as f:
                f.write(report_text)
            click.echo(f"Report written to {output}")
        else:
            click.echo(report_text)
    else:
        click.echo(result.summary())


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Model checkpoint")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
@click.option("--format", "output_format", type=click.Choice(["json", "text", "report"]), default="text")
@click.option("--validate/--no-validate", default=True, help="Validate DICOM input")
@click.pass_context
def predict(ctx, input_path, model, output, output_format, validate):
    """Run nodule detection on a CT scan.

    INPUT_PATH can be a directory of DICOM files or a .mhd file.
    """
    from .preprocessing import load_dicom_series, load_mhd

    config = ctx.obj["config"]
    model_path = Path(model)
    input_path = Path(input_path)

    # Input validation for DICOM directories
    if validate and input_path.is_dir():
        from .input_validation import validate_dicom_series

        validation = validate_dicom_series(input_path)
        if not validation.is_valid:
            click.echo(validation.summary(), err=True)
            sys.exit(1)
        if validation.warnings:
            for w in validation.warnings:
                click.echo(f"Warning: {w}", err=True)

    # Use ONNX runtime if model is .onnx, otherwise use PyTorch
    if model_path.suffix == ".onnx":
        from .inference_onnx import NoduleDetectorONNX
        detector = NoduleDetectorONNX(config, onnx_path=model_path)
    else:
        from .inference import NoduleDetector
        detector = NoduleDetector(config, model_path=model_path)

    # Load scan
    if input_path.suffix == ".mhd":
        image = load_mhd(input_path)
        series_uid = input_path.stem
    elif input_path.is_dir():
        image = load_dicom_series(input_path)
        series_uid = input_path.name
    else:
        click.echo(f"Unsupported input: {input_path}", err=True)
        sys.exit(1)

    # Run detection
    result = detector.predict_scan(image, series_uid=series_uid)

    # Output
    if output_format == "json":
        result_dict = result.to_dict()
        if output:
            with open(output, "w") as f:
                json.dump(result_dict, f, indent=2)
            click.echo(f"Results written to {output}")
        else:
            click.echo(json.dumps(result_dict, indent=2))
    elif output_format == "report":
        report_text = result.dictation()
        if output:
            with open(output, "w") as f:
                f.write(report_text)
            click.echo(f"Report written to {output}")
        else:
            click.echo(report_text)
    else:
        click.echo(result.summary())


@main.command()
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Model checkpoint")
@click.option("--port", type=int, help="Override listening port")
@click.pass_context
def serve(ctx, model, port):
    """Start DICOM SCP server to receive studies from PACS.

    The server listens for incoming CT studies, runs nodule detection,
    and sends DICOM Structured Reports back to the configured PACS.
    """
    from .pacs import DicomStorageSCP

    config = ctx.obj["config"]

    if port:
        config.setdefault("pacs", {})["local_port"] = port

    model_path = Path(model)
    if model_path.suffix == ".onnx":
        from .inference_onnx import NoduleDetectorONNX
        detector = NoduleDetectorONNX(config, onnx_path=model_path)
    else:
        from .inference import NoduleDetector
        detector = NoduleDetector(config, model_path=model_path)

    def on_result(result):
        click.echo(result.summary())

    scp = DicomStorageSCP(config, detector, on_result=on_result)

    try:
        scp.start()
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
        scp.stop()


@main.command(name="export")
@click.option("--checkpoint", type=click.Path(exists=True), required=True, help="Trained .pth checkpoint")
@click.option("--output", "-o", default="./model.onnx", help="Output ONNX file path")
@click.pass_context
def export_model(ctx, checkpoint, output):
    """Export trained model to ONNX for lightweight offline deployment.

    The ONNX model can run without PyTorch installed (~50MB vs ~2GB),
    making it ideal for distributing to workstations.
    """
    from .export import export_to_onnx

    config = ctx.obj["config"]
    path = export_to_onnx(checkpoint, output, config)
    click.echo(f"Model exported to {path}")
    click.echo("Use with: lung-screener predict /path/to/scan -m model.onnx")


@main.command(name="batch")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Model checkpoint")
@click.option("--output", "-o", type=click.Path(), help="Output JSON results file")
@click.option("--priority", type=click.Choice(["stat", "urgent", "routine", "low"]), default="routine")
@click.option("--no-validate", is_flag=True, help="Skip input validation")
@click.pass_context
def batch(ctx, input_dir, model, output, priority, no_validate):
    """Process multiple CT studies in batch mode.

    INPUT_DIR should contain subdirectories with DICOM files, or .mhd files.
    Studies are processed in priority order with progress tracking.
    """
    from .worklist import BatchProcessor, Priority

    config = ctx.obj["config"]
    model_path = Path(model)

    if model_path.suffix == ".onnx":
        from .inference_onnx import NoduleDetectorONNX
        detector = NoduleDetectorONNX(config, onnx_path=model_path)
    else:
        from .inference import NoduleDetector
        detector = NoduleDetector(config, model_path=model_path)

    priority_map = {
        "stat": Priority.STAT,
        "urgent": Priority.URGENT,
        "routine": Priority.ROUTINE,
        "low": Priority.LOW,
    }

    def on_progress(current, total, entry):
        click.echo(f"  [{current}/{total}] {entry.study_id} - {entry.status.value}")

    processor = BatchProcessor(
        detector,
        validate_input=not no_validate,
        on_progress=on_progress,
    )

    count = processor.add_directory(input_dir, priority=priority_map[priority])
    click.echo(f"Found {count} studies to process")
    click.echo("")

    processor.process_all()

    click.echo("")
    click.echo(processor.format_worklist_report())

    if output:
        processor.save_results(output)
        click.echo(f"Results saved to {output}")


@main.command(name="validate")
@click.argument("input_path", type=click.Path(exists=True))
def validate_input(input_path):
    """Validate that a DICOM directory contains a suitable chest CT."""
    from .input_validation import validate_dicom_series

    result = validate_dicom_series(input_path)
    click.echo(result.summary())
    if not result.is_valid:
        sys.exit(1)


@main.command(name="dashboard")
@click.option("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory with metrics.json")
@click.option("--output", "-o", type=click.Path(), help="Output HTML file")
@click.pass_context
def dashboard(ctx, checkpoint_dir, output):
    """Generate training metrics dashboard.

    Creates an interactive HTML page with training curves, validation
    metrics, and performance analysis.
    """
    from .metrics_dashboard import save_dashboard

    metrics_path = Path(checkpoint_dir) / "metrics.json"
    if not metrics_path.exists():
        click.echo(f"No metrics.json found in {checkpoint_dir}", err=True)
        click.echo("Run training first to generate metrics.", err=True)
        sys.exit(1)

    output_path = save_dashboard(metrics_path, output)
    click.echo(f"Dashboard saved to {output_path}")


@main.command(name="risk")
@click.option("--age", type=int, required=True, help="Patient age")
@click.option("--sex", type=click.Choice(["M", "F"]), required=True, help="Patient sex")
@click.option("--diameter", "-d", type=float, required=True, help="Nodule diameter in mm")
@click.option("--nodule-type", type=click.Choice(["solid", "part_solid", "ground_glass"]), default="solid")
@click.option("--pack-years", type=float, default=0.0, help="Smoking pack-years")
@click.option("--family-history", is_flag=True, help="Family history of lung cancer")
@click.option("--emphysema", is_flag=True, help="Emphysema present")
@click.option("--upper-lobe", is_flag=True, help="Nodule in upper lobe")
@click.pass_context
def risk(ctx, age, sex, diameter, nodule_type, pack_years, family_history, emphysema, upper_lobe):
    """Compute Brock/PanCan malignancy risk score for a nodule.

    Uses patient demographics and nodule characteristics to estimate
    malignancy probability based on the validated PanCan model.
    """
    from .risk_model import (
        PatientDemographics,
        compute_brock_score,
        compute_lung_rads_with_risk,
        screening_eligibility,
    )

    demographics = PatientDemographics(
        age=age,
        sex=sex,
        pack_years=pack_years,
        family_history_lung_cancer=family_history,
        emphysema=emphysema,
    )

    score = compute_brock_score(
        demographics,
        nodule_diameter_mm=diameter,
        nodule_type=nodule_type,
        upper_lobe=upper_lobe,
    )

    lung_rads = compute_lung_rads_with_risk(diameter, nodule_type, score)
    eligibility = screening_eligibility(demographics)

    click.echo(f"Brock/PanCan Risk Score")
    click.echo(f"  Malignancy probability: {score.malignancy_probability:.1%}")
    click.echo(f"  Risk category: {score.risk_category}")
    click.echo(f"  Lung-RADS (risk-adjusted): {lung_rads}")
    click.echo(f"")
    click.echo(f"Screening Eligibility ({eligibility['criteria']}):")
    click.echo(f"  {'Eligible' if eligibility['eligible'] else 'Not eligible'}")
    for reason in eligibility["reasons"]:
        click.echo(f"  - {reason}")


@main.command(name="feedback")
@click.option("--feedback-dir", default="./data/feedback", help="Feedback storage directory")
@click.option("--export", "export_dir", type=click.Path(), help="Export confirmed findings for retraining")
@click.pass_context
def feedback_cmd(ctx, feedback_dir, export_dir):
    """View feedback statistics or export confirmed findings for retraining."""
    from .feedback import FeedbackStore

    store = FeedbackStore(feedback_dir)
    stats = store.get_stats()

    click.echo("Radiologist Feedback Statistics")
    click.echo(f"  Total feedback records: {stats['total_feedback']}")
    click.echo(f"  Confirmed (true positive): {stats['confirmed']}")
    click.echo(f"  Rejected (false positive): {stats['rejected']}")
    click.echo(f"  AI precision: {stats['precision']:.1%}")
    click.echo(f"  Avg size disagreement: {stats['avg_size_disagreement_mm']:.1f} mm")
    click.echo(f"  Lung-RADS agreement: {stats['lung_rads_agreement']:.1%}")
    click.echo(f"  Avg TP confidence: {stats['avg_tp_confidence']:.3f}")
    click.echo(f"  Avg FP confidence: {stats['avg_fp_confidence']:.3f}")

    if export_dir:
        result = store.export_for_training(export_dir)
        click.echo(f"\nExported {result['confirmed_count']} positive and "
                    f"{result['rejected_count']} negative annotations to {export_dir}")


@main.command(name="retrain")
@click.option("--checkpoint", "-m", type=click.Path(exists=True), required=True, help="Current best model checkpoint")
@click.option("--feedback-dir", default="./data/feedback", help="Feedback storage directory")
@click.option("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
@click.option("--epochs", type=int, help="Override fine-tuning epochs")
@click.option("--lr", type=float, help="Override learning rate")
@click.option("--min-feedback", type=int, help="Minimum feedback records required")
@click.option("--history", is_flag=True, help="Show retrain history instead of running")
@click.pass_context
def retrain(ctx, checkpoint, feedback_dir, checkpoint_dir, epochs, lr, min_feedback, history):
    """Incrementally retrain the model using radiologist feedback.

    Loads the current best model, merges accumulated feedback with the
    original training data, fine-tunes with a low learning rate, and
    promotes the new model only if it doesn't regress on validation.

    All operations run fully offline.

    \b
    Example:
        lung-screener retrain -m checkpoints/best.pth
        lung-screener retrain -m checkpoints/best.pth --min-feedback 10
        lung-screener retrain -m checkpoints/best.pth --history
    """
    from .retrain import IncrementalRetrainer

    config = ctx.obj["config"]

    if epochs:
        config.setdefault("retrain", {})["epochs"] = epochs
    if lr:
        config.setdefault("retrain", {})["learning_rate"] = lr
    if min_feedback:
        config.setdefault("retrain", {})["min_feedback_records"] = min_feedback

    retrainer = IncrementalRetrainer(
        config,
        base_checkpoint=checkpoint,
        feedback_dir=feedback_dir,
        checkpoint_dir=checkpoint_dir,
    )

    if history:
        entries = retrainer.get_retrain_history()
        if not entries:
            click.echo("No retrain history found.")
            return
        click.echo(f"{'Retrain ID':<20} {'Records':>8} {'Baseline AUC':>13} {'New AUC':>9} {'Promoted':>9}")
        click.echo("-" * 65)
        for entry in entries:
            click.echo(
                f"{entry['retrain_id']:<20} "
                f"{entry['feedback_records_used']:>8} "
                f"{entry['baseline_auc']:>13.4f} "
                f"{entry['new_auc']:>9.4f} "
                f"{'YES' if entry['promoted'] else 'no':>9}"
            )
        return

    click.echo("Starting incremental retrain...")
    click.echo(f"  Base model: {checkpoint}")
    click.echo(f"  Feedback dir: {feedback_dir}")
    click.echo("")

    result = retrainer.retrain()
    click.echo("")
    click.echo(result.summary())


@main.command(name="annotate")
@click.option("--data-dir", default="./data/training", help="Training data directory")
@click.option("--port", "-p", type=int, default=8888, help="Web UI port")
@click.pass_context
def annotate_ui(ctx, data_dir, port):
    """Launch the browser-based annotation tool.

    Opens a local web UI where you can import DICOM scans, scroll
    through slices, click to mark nodules, and prepare training data.

    \b
    Example:
        lung-screener annotate
        lung-screener annotate --port 9000
    """
    import uvicorn

    from .annotator_ui import create_app

    config = ctx.obj["config"]
    app = create_app(data_dir, config)

    click.echo(f"Starting annotation tool at http://localhost:{port}")
    click.echo("Open this URL in your browser to begin annotating.")
    click.echo("Press Ctrl+C to stop.")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


@main.group()
@click.option("--data-dir", default="./data/training", help="Training data directory")
@click.pass_context
def data(ctx, data_dir):
    """Manage training data: import DICOM scans, annotate, and prepare."""
    from .data_manager import DataManager

    ctx.ensure_object(dict)
    ctx.obj["data_manager"] = DataManager(data_dir, ctx.obj["config"])


@data.command(name="import")
@click.argument("dicom_dir", type=click.Path(exists=True))
@click.option("--label", "-l", default="", help="Human-readable label for this scan")
@click.pass_context
def data_import(ctx, dicom_dir, label):
    """Import a DICOM series directory into the training dataset.

    DICOM_DIR should contain the .dcm files for a single CT series.
    The scan will be converted to MHD format and registered for annotation.

    Example:

        lung-screener data import /path/to/patient_042/CT_series/
    """
    dm = ctx.obj["data_manager"]
    scan = dm.import_dicom(dicom_dir, label=label)
    click.echo(f"Imported: {scan['series_uid']}")
    click.echo(f"  Patient: {scan['patient_id']}")
    click.echo(f"  Slices:  {scan['num_slices']}")
    click.echo(f"  Status:  {scan['status']}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  1. Annotate nodules:  lung-screener data annotate {scan['series_uid']} --x 10.0 --y 20.0 --z -150.0 --diameter 8.5")
    click.echo(f"  2. Or mark negative:  lung-screener data mark-negative {scan['series_uid']}")


@data.command(name="annotate")
@click.argument("series_uid")
@click.option("--x", type=float, required=True, help="X coordinate in mm")
@click.option("--y", type=float, required=True, help="Y coordinate in mm")
@click.option("--z", type=float, required=True, help="Z coordinate in mm (slice position)")
@click.option("--diameter", "-d", type=float, required=True, help="Nodule diameter in mm")
@click.option("--note", "-n", default="", help="Optional clinical note")
@click.pass_context
def data_annotate(ctx, series_uid, x, y, z, diameter, note):
    """Add a nodule annotation to an imported scan.

    Coordinates should be in world mm -- the same values shown by your
    DICOM viewer when you hover over the nodule.

    Example:

        lung-screener data annotate 1.3.6.1.4... --x -42.3 --y 118.7 --z -205.1 --diameter 8.5
    """
    dm = ctx.obj["data_manager"]
    dm.annotate(series_uid, x, y, z, diameter, note=note)
    click.echo(f"Annotation added: ({x}, {y}, {z}) mm, {diameter}mm diameter")


@data.command(name="mark-negative")
@click.argument("series_uid")
@click.pass_context
def data_mark_negative(ctx, series_uid):
    """Mark a scan as having no nodules (negative training case)."""
    dm = ctx.obj["data_manager"]
    dm.mark_negative(series_uid)
    click.echo(f"Marked {series_uid} as negative (no nodules)")


@data.command(name="list")
@click.pass_context
def data_list(ctx):
    """List all imported scans and their annotation status."""
    dm = ctx.obj["data_manager"]
    scans = dm.list_scans()

    if not scans:
        click.echo("No scans imported yet.")
        click.echo("Import with: lung-screener data import /path/to/dicom/")
        return

    click.echo(f"{'Series UID':<45} {'Label':<15} {'Slices':>6} {'Annotations':>12} {'Status':<12}")
    click.echo("-" * 95)
    for s in scans:
        uid_short = s['series_uid'][:42] + "..." if len(s['series_uid']) > 45 else s['series_uid']
        click.echo(
            f"{uid_short:<45} {s['label']:<15} {s['num_slices']:>6} "
            f"{s['num_annotations']:>12} {s['status']:<12}"
        )

    click.echo("")
    total = len(scans)
    ready = sum(1 for s in scans if s["status"] in ("annotated", "negative"))
    click.echo(f"Total: {total} scans, {ready} ready for training")


@data.command(name="prepare")
@click.option("--output-dir", "-o", default=None, help="Output directory (default: data_dir/prepared)")
@click.pass_context
def data_prepare(ctx, output_dir):
    """Compile annotations into training-ready CSV files.

    Generates annotations.csv + candidates_V2.csv from your imported and
    annotated scans. The output directory can be passed directly to the
    train command via config override.

    Example:

        lung-screener data prepare
        lung-screener train --dataset-dir ./data/training/prepared
    """
    dm = ctx.obj["data_manager"]
    result_dir = dm.prepare(output_dir)
    click.echo("")
    click.echo(f"Training data ready at: {result_dir}")
    click.echo("")
    click.echo("To train:")
    click.echo(f"  lung-screener train  (set data.dataset_dir to {result_dir} in config)")


@main.command()
@click.pass_context
def verify(ctx):
    """Verify PACS connectivity with a C-ECHO."""
    from pynetdicom import AE

    config = ctx.obj["config"]
    pacs_config = config.get("pacs", {})

    ae = AE(ae_title=pacs_config.get("local_ae_title", "LUNG_SCREEN_AI"))
    ae.add_requested_context("1.2.840.10008.1.1")  # Verification SOP

    remote_host = pacs_config.get("remote_host", "localhost")
    remote_port = pacs_config.get("remote_port", 4006)
    remote_ae = pacs_config.get("remote_ae_title", "GEPACS")

    click.echo(f"Sending C-ECHO to {remote_ae}@{remote_host}:{remote_port}...")

    assoc = ae.associate(remote_host, remote_port, ae_title=remote_ae)

    if assoc.is_established:
        status = assoc.send_c_echo()
        assoc.release()

        if status and status.Status == 0x0000:
            click.echo("C-ECHO successful - PACS connection verified!")
        else:
            click.echo(f"C-ECHO failed with status: {status}", err=True)
            sys.exit(1)
    else:
        click.echo("Could not establish association with PACS", err=True)
        sys.exit(1)


@main.command(name="gradcam")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Model checkpoint")
@click.option("--output", "-o", default="./gradcam_output", help="Output directory for saliency maps")
@click.option("--view", type=click.Choice(["slices", "three-plane", "both"]), default="both",
              help="Visualization mode")
@click.option("--num-slices", type=int, default=9, help="Number of axial slices (for 'slices' view)")
@click.option("--alpha", type=float, default=0.4, help="Heatmap overlay opacity (0-1)")
@click.option("--max-findings", type=int, default=10, help="Max findings to visualize")
@click.option("--validate/--no-validate", default=True, help="Validate DICOM input")
@click.pass_context
def gradcam(ctx, input_path, model, output, view, num_slices, alpha, max_findings, validate):
    """Generate GradCAM saliency maps for detected nodules.

    Shows which voxel regions drove the model's nodule prediction —
    useful for radiologist review and model debugging.

    \b
    Examples:
        lung-screener gradcam /path/to/scan -m best.pth
        lung-screener gradcam scan.mhd -m best.pth --view three-plane
        lung-screener gradcam /dicom/dir -m best.pth -o ./maps --alpha 0.5
    """
    from .gradcam import GradCAM3D, load_model_for_gradcam, render_slices, render_three_plane
    from .preprocessing import CTPreprocessor, extract_patch, load_dicom_series, load_mhd

    config = ctx.obj["config"]
    model_path = Path(model)
    input_path = Path(input_path)
    output_dir = Path(output)

    # Input validation for DICOM directories
    if validate and input_path.is_dir():
        from .input_validation import validate_dicom_series

        validation = validate_dicom_series(input_path)
        if not validation.is_valid:
            click.echo(validation.summary(), err=True)
            sys.exit(1)

    # Load model
    loaded_model, device = load_model_for_gradcam(config, model_path)
    gc = GradCAM3D(loaded_model)

    # Load and preprocess scan
    if input_path.suffix == ".mhd":
        image = load_mhd(input_path)
        series_uid = input_path.stem
    elif input_path.is_dir():
        image = load_dicom_series(input_path)
        series_uid = input_path.name
    else:
        click.echo(f"Unsupported input: {input_path}", err=True)
        sys.exit(1)

    click.echo(f"Processing: {series_uid}")

    preprocessor = CTPreprocessor(config)
    processed = preprocessor.process_scan(image)
    volume = processed["volume"]
    candidates = processed["candidates"]

    if not candidates:
        click.echo("No candidates found in scan.")
        gc.release()
        return

    click.echo(f"Found {len(candidates)} candidates, generating saliency maps...")

    patch_size = tuple(config.get("model", {}).get("patch_size", [48, 48, 48]))
    import numpy as np

    # Classify and generate GradCAM for each candidate
    count = 0
    for i, cand in enumerate(candidates):
        if count >= max_findings:
            break

        patch = extract_patch(volume, cand["center_voxel"], patch_size)
        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)

        heatmap, prediction = gc.generate(patch_tensor, target_class=1)  # Explain nodule class

        # Only visualize if model thinks it's a nodule or close
        if prediction["nodule_probability"] < 0.1:
            continue

        count += 1
        prefix = f"candidate_{i:03d}"
        conf = prediction["confidence"]
        label = "nodule" if prediction["class"] == 1 else "non_nodule"

        click.echo(
            f"  [{count}] Candidate {i}: {label} "
            f"(nodule prob: {prediction['nodule_probability']:.1%}, "
            f"diameter: {cand['diameter_mm']:.1f}mm)"
        )

        if view in ("slices", "both"):
            render_slices(
                patch, heatmap,
                output_dir / f"{prefix}_slices.png",
                num_slices=num_slices,
                alpha=alpha,
                prediction=prediction,
            )
        if view in ("three-plane", "both"):
            render_three_plane(
                patch, heatmap,
                output_dir / f"{prefix}_3plane.png",
                alpha=alpha,
                prediction=prediction,
            )

        # Save raw heatmap as numpy for further analysis
        np.save(output_dir / f"{prefix}_heatmap.npy", heatmap)

    gc.release()
    click.echo(f"\nSaved {count} GradCAM visualizations to {output_dir}/")


if __name__ == "__main__":
    main()
