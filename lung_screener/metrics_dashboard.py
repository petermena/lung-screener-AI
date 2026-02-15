"""Validation metrics dashboard for monitoring model performance.

Provides a web-based dashboard for visualizing training progress,
validation metrics, and model performance over time.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Logs training and validation metrics to a JSON file.

    Stores per-epoch metrics for visualization in the dashboard.
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.json"
        self.history: dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_auc": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_auc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_sensitivity": [],
            "val_specificity": [],
            "learning_rate": [],
        }
        self._load_existing()

    def _load_existing(self):
        """Load existing metrics if available."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file) as f:
                    self.history = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        learning_rate: float = 0.0,
    ):
        """Log metrics for one epoch."""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(round(train_metrics.get("loss", 0), 6))
        self.history["train_accuracy"].append(round(train_metrics.get("accuracy", 0), 4))
        self.history["train_auc"].append(round(train_metrics.get("auc", 0), 4))
        self.history["val_loss"].append(round(val_metrics.get("loss", 0), 6))
        self.history["val_accuracy"].append(round(val_metrics.get("accuracy", 0), 4))
        self.history["val_auc"].append(round(val_metrics.get("auc", 0), 4))
        self.history["val_precision"].append(round(val_metrics.get("precision", 0), 4))
        self.history["val_recall"].append(round(val_metrics.get("recall", 0), 4))
        self.history["val_f1"].append(round(val_metrics.get("f1", 0), 4))
        self.history["val_sensitivity"].append(round(val_metrics.get("sensitivity", 0), 4))
        self.history["val_specificity"].append(round(val_metrics.get("specificity", 0), 4))
        self.history["learning_rate"].append(round(learning_rate, 8))

        self._save()

    def _save(self):
        """Persist metrics to disk."""
        with open(self.metrics_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_latest(self) -> dict:
        """Get the latest epoch's metrics."""
        if not self.history["epoch"]:
            return {}
        return {k: v[-1] for k, v in self.history.items()}

    def get_best(self, metric: str = "val_auc") -> dict:
        """Get the epoch with the best value of a given metric."""
        if not self.history.get(metric):
            return {}
        values = self.history[metric]
        best_idx = int(np.argmax(values))
        return {k: v[best_idx] for k, v in self.history.items()}


def generate_dashboard_html(metrics_path: str | Path) -> str:
    """Generate a self-contained HTML dashboard page.

    Creates an interactive HTML page with Chart.js graphs showing
    training curves, validation metrics, and performance analysis.

    Args:
        metrics_path: Path to metrics.json file.

    Returns:
        HTML string for the dashboard.
    """
    metrics_path = Path(metrics_path)
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics_json = json.dumps(metrics)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lung Screener AI - Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ text-align: center; margin-bottom: 10px; color: #00d4ff; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px; max-width: 1400px; margin: 0 auto; }}
        .card {{ background: #16213e; border-radius: 12px; padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .card h2 {{ font-size: 16px; color: #00d4ff; margin-bottom: 15px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;
                 max-width: 1400px; margin: 0 auto 20px; }}
        .stat {{ background: #16213e; border-radius: 12px; padding: 15px; text-align: center; }}
        .stat .value {{ font-size: 28px; font-weight: bold; color: #00d4ff; }}
        .stat .label {{ font-size: 12px; color: #888; margin-top: 5px; }}
        canvas {{ max-height: 300px; }}
        .no-data {{ text-align: center; color: #666; padding: 40px;
                   font-size: 18px; }}
    </style>
</head>
<body>
    <h1>Lung Screener AI</h1>
    <p class="subtitle">Training Metrics Dashboard</p>

    <div class="stats" id="statsGrid"></div>
    <div class="grid">
        <div class="card"><h2>Loss Curves</h2><canvas id="lossChart"></canvas></div>
        <div class="card"><h2>AUC-ROC</h2><canvas id="aucChart"></canvas></div>
        <div class="card"><h2>Sensitivity &amp; Specificity</h2><canvas id="sensSpecChart"></canvas></div>
        <div class="card"><h2>Precision / Recall / F1</h2><canvas id="prfChart"></canvas></div>
        <div class="card"><h2>Learning Rate Schedule</h2><canvas id="lrChart"></canvas></div>
        <div class="card"><h2>Accuracy</h2><canvas id="accChart"></canvas></div>
    </div>

    <script>
    const metrics = {metrics_json};

    if (!metrics.epoch || metrics.epoch.length === 0) {{
        document.querySelector('.grid').innerHTML =
            '<div class="no-data">No training data yet. Start training to see metrics.</div>';
    }} else {{
        const epochs = metrics.epoch;
        const latest = {{}};
        for (const key in metrics) {{
            if (metrics[key].length > 0) latest[key] = metrics[key][metrics[key].length - 1];
        }}

        // Stats cards
        const statsData = [
            {{ label: 'Best Val AUC', value: Math.max(...(metrics.val_auc || [0])).toFixed(4) }},
            {{ label: 'Latest Sensitivity', value: (latest.val_sensitivity || 0).toFixed(4) }},
            {{ label: 'Latest Specificity', value: (latest.val_specificity || 0).toFixed(4) }},
            {{ label: 'Epochs Trained', value: epochs.length }},
        ];
        const statsGrid = document.getElementById('statsGrid');
        statsData.forEach(s => {{
            statsGrid.innerHTML += `<div class="stat"><div class="value">${{s.value}}</div><div class="label">${{s.label}}</div></div>`;
        }});

        const chartOpts = {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#ccc' }} }} }},
            scales: {{
                x: {{ title: {{ display: true, text: 'Epoch', color: '#888' }}, ticks: {{ color: '#888' }} }},
                y: {{ ticks: {{ color: '#888' }} }}
            }}
        }};

        new Chart('lossChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Train Loss', data: metrics.train_loss, borderColor: '#ff6384', tension: 0.3 }},
                    {{ label: 'Val Loss', data: metrics.val_loss, borderColor: '#36a2eb', tension: 0.3 }},
                ]
            }}, options: chartOpts
        }});

        new Chart('aucChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Train AUC', data: metrics.train_auc, borderColor: '#ff6384', tension: 0.3 }},
                    {{ label: 'Val AUC', data: metrics.val_auc, borderColor: '#36a2eb', tension: 0.3 }},
                ]
            }}, options: chartOpts
        }});

        new Chart('sensSpecChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Sensitivity', data: metrics.val_sensitivity, borderColor: '#4bc0c0', tension: 0.3 }},
                    {{ label: 'Specificity', data: metrics.val_specificity, borderColor: '#ff9f40', tension: 0.3 }},
                ]
            }}, options: chartOpts
        }});

        new Chart('prfChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Precision', data: metrics.val_precision, borderColor: '#9966ff', tension: 0.3 }},
                    {{ label: 'Recall', data: metrics.val_recall, borderColor: '#ff6384', tension: 0.3 }},
                    {{ label: 'F1', data: metrics.val_f1, borderColor: '#4bc0c0', tension: 0.3 }},
                ]
            }}, options: chartOpts
        }});

        new Chart('lrChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Learning Rate', data: metrics.learning_rate, borderColor: '#ffcd56', tension: 0.3 }},
                ]
            }}, options: {{ ...chartOpts, scales: {{ ...chartOpts.scales, y: {{ type: 'logarithmic', ticks: {{ color: '#888' }} }} }} }}
        }});

        new Chart('accChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Train Accuracy', data: metrics.train_accuracy, borderColor: '#ff6384', tension: 0.3 }},
                    {{ label: 'Val Accuracy', data: metrics.val_accuracy, borderColor: '#36a2eb', tension: 0.3 }},
                ]
            }}, options: chartOpts
        }});
    }}
    </script>
</body>
</html>"""

    return html


def save_dashboard(
    metrics_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Generate and save the dashboard HTML.

    Args:
        metrics_path: Path to metrics.json.
        output_path: Where to save the HTML. Defaults to same directory as metrics.

    Returns:
        Path to the generated HTML file.
    """
    metrics_path = Path(metrics_path)
    if output_path is None:
        output_path = metrics_path.parent / "dashboard.html"
    else:
        output_path = Path(output_path)

    html = generate_dashboard_html(metrics_path)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Dashboard saved to {output_path}")
    return output_path
