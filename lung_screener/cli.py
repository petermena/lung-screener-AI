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
    from .inference import NoduleDetector
    from .preprocessing import load_dicom_series, load_mhd

    config = ctx.obj["config"]
    detector = NoduleDetector(config, model_path=model)

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
    from .inference import NoduleDetector
    from .pacs import DicomStorageSCP

    config = ctx.obj["config"]

    if port:
        config.setdefault("pacs", {})["local_port"] = port

    detector = NoduleDetector(config, model_path=model)

    def on_result(result):
        click.echo(result.summary())

    scp = DicomStorageSCP(config, detector, on_result=on_result)

    try:
        scp.start()
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
        scp.stop()


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
