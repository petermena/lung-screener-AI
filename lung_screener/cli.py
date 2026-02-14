"""Command-line interface for Lung Screener AI.

Provides commands for training, inference, and PACS server management.
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
@click.pass_context
def train(ctx, epochs, batch_size, lr, resume, checkpoint_dir):
    """Train the nodule detection model on LUNA16 data."""
    from .train import Trainer

    config = ctx.obj["config"]

    # Apply CLI overrides
    if epochs:
        config.setdefault("training", {})["epochs"] = epochs
    if batch_size:
        config.setdefault("training", {})["batch_size"] = batch_size
    if lr:
        config.setdefault("training", {})["learning_rate"] = lr

    trainer = Trainer(config, checkpoint_dir=checkpoint_dir)
    trainer.train(resume_from=resume)


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Path(exists=True), required=True, help="Model checkpoint")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
@click.option("--format", "output_format", type=click.Choice(["json", "text"]), default="text")
@click.pass_context
def predict(ctx, input_path, model, output, output_format):
    """Run nodule detection on a CT scan.

    INPUT_PATH can be a directory of DICOM files or a .mhd file.
    """
    from .preprocessing import load_dicom_series, load_mhd

    config = ctx.obj["config"]
    model_path = Path(model)

    # Use ONNX runtime if model is .onnx, otherwise use PyTorch
    if model_path.suffix == ".onnx":
        from .inference_onnx import NoduleDetectorONNX
        detector = NoduleDetectorONNX(config, onnx_path=model_path)
    else:
        from .inference import NoduleDetector
        detector = NoduleDetector(config, model_path=model_path)

    input_path = Path(input_path)

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

    Coordinates should be in world mm — the same values shown by your
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


if __name__ == "__main__":
    main()
