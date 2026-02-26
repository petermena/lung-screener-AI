"""Validation metrics dashboard for monitoring model performance.

Provides a web-based dashboard for visualizing training progress,
validation metrics, evaluation results, calibration analysis, and
model performance over time.

The dashboard is a self-contained HTML file that uses Chart.js for
interactive visualizations and includes:
- Training curves (loss, AUC, accuracy)
- Evaluation metrics (ROC curve, PR curve, FROC, confusion matrix)
- Calibration analysis (reliability diagram, confidence histograms)
- Operating point analysis at multiple thresholds
- Optimization results (model size, latency)
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


def generate_dashboard_html(
    metrics_path: str | Path,
    eval_results_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    optimization_path: str | Path | None = None,
) -> str:
    """Generate a comprehensive self-contained HTML dashboard.

    Creates an interactive HTML page with Chart.js graphs showing training
    curves, evaluation results (ROC, PR, FROC, confusion matrix), calibration
    analysis, and optimization results.

    Args:
        metrics_path: Path to metrics.json (training history).
        eval_results_path: Path to eval_results.json (full evaluation).
        calibration_path: Path to calibration.json.
        optimization_path: Path to optimization_results.json.

    Returns:
        HTML string for the dashboard.
    """
    metrics_path = Path(metrics_path)
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    eval_results = {}
    if eval_results_path:
        eval_results_path = Path(eval_results_path)
        if eval_results_path.exists():
            with open(eval_results_path) as f:
                eval_results = json.load(f)

    calibration = {}
    if calibration_path:
        calibration_path = Path(calibration_path)
        if calibration_path.exists():
            with open(calibration_path) as f:
                calibration = json.load(f)

    optimization = {}
    if optimization_path:
        optimization_path = Path(optimization_path)
        if optimization_path.exists():
            with open(optimization_path) as f:
                optimization = json.load(f)

    # Auto-detect eval_results from same directory as metrics
    if not eval_results:
        auto_eval = metrics_path.parent / "eval_results.json"
        if auto_eval.exists():
            with open(auto_eval) as f:
                eval_results = json.load(f)

    if not calibration:
        auto_cal = metrics_path.parent / "calibration.json"
        if auto_cal.exists():
            with open(auto_cal) as f:
                calibration = json.load(f)

    if not optimization:
        auto_opt = metrics_path.parent / "optimization_results.json"
        if auto_opt.exists():
            with open(auto_opt) as f:
                optimization = json.load(f)

    metrics_json = json.dumps(metrics)
    eval_json = json.dumps(eval_results)
    cal_json = json.dumps(calibration)
    opt_json = json.dumps(optimization)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lung Screener AI - Comprehensive Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0f0f1a; color: #e0e0e0; padding: 20px; }}
        h1 {{ text-align: center; margin-bottom: 5px; color: #00d4ff;
             font-size: 28px; letter-spacing: 1px; }}
        .subtitle {{ text-align: center; color: #666; margin-bottom: 25px; font-size: 14px; }}
        .section-title {{ color: #00d4ff; font-size: 20px; margin: 30px auto 15px;
                         max-width: 1400px; padding-left: 5px;
                         border-left: 3px solid #00d4ff; padding-left: 12px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
                gap: 16px; max-width: 1400px; margin: 0 auto 20px; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                  gap: 16px; max-width: 1400px; margin: 0 auto 20px; }}
        .card {{ background: #1a1a2e; border-radius: 10px; padding: 18px;
                border: 1px solid #2a2a4a; }}
        .card h2 {{ font-size: 14px; color: #00d4ff; margin-bottom: 12px;
                   text-transform: uppercase; letter-spacing: 0.5px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                 gap: 12px; max-width: 1400px; margin: 0 auto 20px; }}
        .stat {{ background: #1a1a2e; border-radius: 10px; padding: 14px;
                text-align: center; border: 1px solid #2a2a4a; }}
        .stat .value {{ font-size: 26px; font-weight: bold; color: #00d4ff; }}
        .stat .value.good {{ color: #4bc0c0; }}
        .stat .value.warn {{ color: #ff9f40; }}
        .stat .label {{ font-size: 11px; color: #888; margin-top: 4px;
                       text-transform: uppercase; letter-spacing: 0.5px; }}
        canvas {{ max-height: 280px; }}
        .no-data {{ text-align: center; color: #444; padding: 30px; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #0f0f1a; color: #00d4ff; padding: 8px 6px; text-align: right;
             font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
        td {{ padding: 6px; text-align: right; border-bottom: 1px solid #2a2a4a; }}
        th:first-child, td:first-child {{ text-align: left; }}
        tr:hover td {{ background: #1f1f3a; }}
        .cm-grid {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 0;
                   max-width: 280px; margin: 10px auto; }}
        .cm-cell {{ padding: 12px 8px; text-align: center; font-size: 14px; font-weight: bold;
                   border: 1px solid #2a2a4a; }}
        .cm-header {{ background: #0f0f1a; color: #888; font-size: 11px; font-weight: 600;
                     text-transform: uppercase; padding: 8px; text-align: center;
                     border: 1px solid #2a2a4a; }}
        .cm-tp {{ background: #1a4a1a; color: #4bc0c0; }}
        .cm-tn {{ background: #1a3a4a; color: #36a2eb; }}
        .cm-fp {{ background: #4a2a1a; color: #ff9f40; }}
        .cm-fn {{ background: #4a1a2a; color: #ff6384; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
                 font-size: 11px; font-weight: 600; }}
        .badge-good {{ background: #1a4a1a; color: #4bc0c0; }}
        .badge-ok {{ background: #3a3a1a; color: #ffcd56; }}
        .badge-warn {{ background: #4a2a1a; color: #ff9f40; }}
        .footer {{ text-align: center; color: #444; margin-top: 30px; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Lung Screener AI</h1>
    <p class="subtitle">Comprehensive Model Performance Dashboard</p>

    <!-- ==================== EVALUATION SUMMARY ==================== -->
    <div class="stats" id="evalStats"></div>

    <!-- ==================== EVALUATION CURVES ==================== -->
    <h3 class="section-title" id="evalSection" style="display:none">Evaluation Results</h3>
    <div class="grid" id="evalGrid" style="display:none">
        <div class="card"><h2>ROC Curve</h2><canvas id="rocChart"></canvas></div>
        <div class="card"><h2>Precision-Recall Curve</h2><canvas id="prChart"></canvas></div>
        <div class="card"><h2>FROC Sensitivity</h2><canvas id="frocChart"></canvas></div>
        <div class="card">
            <h2>Confusion Matrix (Threshold: <span id="cmThreshold"></span>)</h2>
            <div class="cm-grid" id="cmGrid"></div>
        </div>
    </div>

    <!-- ==================== CALIBRATION & CONFIDENCE ==================== -->
    <h3 class="section-title" id="calSection" style="display:none">Calibration &amp; Confidence Analysis</h3>
    <div class="grid" id="calGrid" style="display:none">
        <div class="card"><h2>Reliability Diagram</h2><canvas id="reliabilityChart"></canvas></div>
        <div class="card"><h2>Confidence Distribution</h2><canvas id="confHistChart"></canvas></div>
    </div>

    <!-- ==================== OPERATING POINTS ==================== -->
    <h3 class="section-title" id="opSection" style="display:none">Operating Point Analysis</h3>
    <div class="grid" id="opGrid" style="display:none">
        <div class="card"><h2>Sensitivity vs Specificity by Threshold</h2><canvas id="sensSpecThreshChart"></canvas></div>
        <div class="card card-table"><h2>Operating Points Table</h2><div id="opTable"></div></div>
    </div>

    <!-- ==================== TRAINING CURVES ==================== -->
    <h3 class="section-title" id="trainSection" style="display:none">Training History</h3>
    <div class="grid" id="trainGrid" style="display:none">
        <div class="card"><h2>Loss Curves</h2><canvas id="lossChart"></canvas></div>
        <div class="card"><h2>AUC-ROC Over Epochs</h2><canvas id="aucChart"></canvas></div>
        <div class="card"><h2>Sensitivity &amp; Specificity</h2><canvas id="sensSpecChart"></canvas></div>
        <div class="card"><h2>Precision / Recall / F1</h2><canvas id="prfChart"></canvas></div>
        <div class="card"><h2>Learning Rate Schedule</h2><canvas id="lrChart"></canvas></div>
        <div class="card"><h2>Accuracy</h2><canvas id="accChart"></canvas></div>
    </div>

    <!-- ==================== OPTIMIZATION ==================== -->
    <h3 class="section-title" id="optSection" style="display:none">Optimization &amp; Deployment</h3>
    <div class="grid-3" id="optGrid" style="display:none">
        <div class="card" id="optCard"></div>
    </div>

    <div class="footer">
        Lung Screener AI &mdash; Generated <span id="genTime"></span>
    </div>

    <script>
    const metrics = {metrics_json};
    const evalData = {eval_json};
    const calData = {cal_json};
    const optData = {opt_json};

    document.getElementById('genTime').textContent = new Date().toLocaleString();

    const darkChartOpts = {{
        responsive: true,
        plugins: {{
            legend: {{ labels: {{ color: '#aaa', font: {{ size: 11 }} }} }},
        }},
        scales: {{
            x: {{ ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }} }},
            y: {{ ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }} }}
        }}
    }};

    // ==================== EVALUATION SECTION ====================
    if (evalData && evalData.metrics) {{
        const m = evalData.metrics;
        const cm = evalData.confusion_matrix || {{}};
        const opt_thresh = evalData.optimal_threshold || {{}};

        document.getElementById('evalSection').style.display = '';
        document.getElementById('evalGrid').style.display = '';

        // Stats cards
        const statsEl = document.getElementById('evalStats');
        const statsItems = [
            {{ label: 'AUC-ROC', value: m.auc_roc, cls: m.auc_roc >= 0.95 ? 'good' : '' }},
            {{ label: '95% CI', value: `${{m.auc_95ci_low}}-${{m.auc_95ci_high}}`, cls: '' }},
            {{ label: 'Sensitivity', value: m.sensitivity, cls: m.sensitivity >= 0.9 ? 'good' : 'warn' }},
            {{ label: 'Specificity', value: m.specificity, cls: m.specificity >= 0.9 ? 'good' : '' }},
            {{ label: 'Precision (PPV)', value: m.precision, cls: '' }},
            {{ label: 'NPV', value: m.npv, cls: m.npv >= 0.99 ? 'good' : '' }},
            {{ label: 'F1 Score', value: m.f1_score, cls: '' }},
            {{ label: 'ECE', value: m.ece, cls: m.ece <= 0.05 ? 'good' : 'warn' }},
        ];
        statsItems.forEach(s => {{
            const v = typeof s.value === 'number' ? s.value.toFixed(4) : s.value;
            statsEl.innerHTML += `<div class="stat"><div class="value ${{s.cls}}">${{v}}</div><div class="label">${{s.label}}</div></div>`;
        }});

        // ROC Curve
        if (evalData.roc_curve) {{
            new Chart('rocChart', {{
                type: 'line',
                data: {{
                    labels: evalData.roc_curve.fpr,
                    datasets: [
                        {{ label: `ROC (AUC=${{m.auc_roc}})`, data: evalData.roc_curve.tpr,
                          borderColor: '#00d4ff', borderWidth: 2, pointRadius: 0, tension: 0.1 }},
                        {{ label: 'Random', data: evalData.roc_curve.fpr,
                          borderColor: '#444', borderWidth: 1, borderDash: [5,5], pointRadius: 0 }},
                    ]
                }},
                options: {{
                    ...darkChartOpts,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'False Positive Rate', color: '#888' }},
                             ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }}, min: 0, max: 1 }},
                        y: {{ title: {{ display: true, text: 'True Positive Rate', color: '#888' }},
                             ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }}, min: 0, max: 1 }},
                    }}
                }}
            }});
        }}

        // PR Curve
        if (evalData.pr_curve) {{
            new Chart('prChart', {{
                type: 'line',
                data: {{
                    labels: evalData.pr_curve.recall,
                    datasets: [
                        {{ label: 'Precision-Recall', data: evalData.pr_curve.precision,
                          borderColor: '#9966ff', borderWidth: 2, pointRadius: 0, tension: 0.1 }},
                    ]
                }},
                options: {{
                    ...darkChartOpts,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Recall', color: '#888' }},
                             ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }}, min: 0, max: 1 }},
                        y: {{ title: {{ display: true, text: 'Precision', color: '#888' }},
                             ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }}, min: 0, max: 1 }},
                    }}
                }}
            }});
        }}

        // FROC Chart
        if (evalData.froc_sensitivity) {{
            const frocKeys = Object.keys(evalData.froc_sensitivity);
            const frocLabels = frocKeys.map(k => k.replace('sens_at_fpr_', ''));
            const frocValues = frocKeys.map(k => evalData.froc_sensitivity[k]);
            new Chart('frocChart', {{
                type: 'bar',
                data: {{
                    labels: frocLabels,
                    datasets: [{{
                        label: 'Sensitivity',
                        data: frocValues,
                        backgroundColor: frocValues.map(v => v >= 0.9 ? '#4bc0c088' : v >= 0.8 ? '#ffcd5688' : '#ff638488'),
                        borderColor: frocValues.map(v => v >= 0.9 ? '#4bc0c0' : v >= 0.8 ? '#ffcd56' : '#ff6384'),
                        borderWidth: 1,
                    }}]
                }},
                options: {{
                    ...darkChartOpts,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'False Positive Rate', color: '#888' }},
                             ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }} }},
                        y: {{ title: {{ display: true, text: 'Sensitivity', color: '#888' }},
                             ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }}, min: 0, max: 1 }},
                    }}
                }}
            }});
        }}

        // Confusion Matrix
        if (cm.tp !== undefined) {{
            document.getElementById('cmThreshold').textContent = evalData.threshold || 0.5;
            const total = cm.tp + cm.fp + cm.fn + cm.tn;
            document.getElementById('cmGrid').innerHTML = `
                <div class="cm-header"></div>
                <div class="cm-header">Pred +</div>
                <div class="cm-header">Pred -</div>
                <div class="cm-header">Actual +</div>
                <div class="cm-cell cm-tp">${{cm.tp}}<br><small>${{(cm.tp/total*100).toFixed(1)}}%</small></div>
                <div class="cm-cell cm-fn">${{cm.fn}}<br><small>${{(cm.fn/total*100).toFixed(1)}}%</small></div>
                <div class="cm-header">Actual -</div>
                <div class="cm-cell cm-fp">${{cm.fp}}<br><small>${{(cm.fp/total*100).toFixed(1)}}%</small></div>
                <div class="cm-cell cm-tn">${{cm.tn}}<br><small>${{(cm.tn/total*100).toFixed(1)}}%</small></div>
            `;
        }}
    }}

    // ==================== CALIBRATION SECTION ====================
    if (evalData.reliability_diagram || evalData.confidence_histogram) {{
        document.getElementById('calSection').style.display = '';
        document.getElementById('calGrid').style.display = '';

        // Reliability Diagram
        if (evalData.reliability_diagram) {{
            const rd = evalData.reliability_diagram;
            new Chart('reliabilityChart', {{
                type: 'bar',
                data: {{
                    labels: rd.bin_centers.map(c => c.toFixed(2)),
                    datasets: [
                        {{ label: 'Accuracy', data: rd.bin_accuracies, backgroundColor: '#4bc0c088',
                          borderColor: '#4bc0c0', borderWidth: 1 }},
                        {{ label: 'Perfect Calibration', data: rd.bin_centers,
                          type: 'line', borderColor: '#ff638488', borderWidth: 2,
                          borderDash: [5,5], pointRadius: 0 }},
                    ]
                }},
                options: {{
                    ...darkChartOpts,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Predicted Confidence', color: '#888' }},
                             ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }} }},
                        y: {{ title: {{ display: true, text: 'Actual Accuracy', color: '#888' }},
                             ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }}, min: 0, max: 1 }},
                    }},
                    plugins: {{
                        ...darkChartOpts.plugins,
                        subtitle: {{ display: true,
                            text: `ECE = ${{(evalData.reliability_diagram.ece || evalData.metrics?.ece || 0).toFixed(4)}}${{calData.temperature ? '  |  Temperature = ' + calData.temperature.toFixed(4) : ''}}`,
                            color: '#888', font: {{ size: 12 }} }}
                    }}
                }}
            }});
        }}

        // Confidence Histogram
        if (evalData.confidence_histogram) {{
            const ch = evalData.confidence_histogram;
            new Chart('confHistChart', {{
                type: 'bar',
                data: {{
                    labels: ch.bin_centers.map(c => c.toFixed(2)),
                    datasets: [
                        {{ label: 'True Nodules', data: ch.positive_counts,
                          backgroundColor: '#4bc0c066', borderColor: '#4bc0c0', borderWidth: 1 }},
                        {{ label: 'Non-Nodules', data: ch.negative_counts,
                          backgroundColor: '#ff638466', borderColor: '#ff6384', borderWidth: 1 }},
                    ]
                }},
                options: {{
                    ...darkChartOpts,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Confidence Score', color: '#888' }},
                             ticks: {{ color: '#666', maxTicksLimit: 10 }}, grid: {{ color: '#1f1f3a' }} }},
                        y: {{ title: {{ display: true, text: 'Count', color: '#888' }},
                             ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }} }},
                    }}
                }}
            }});
        }}
    }}

    // ==================== OPERATING POINTS SECTION ====================
    if (evalData.operating_points && evalData.operating_points.length > 0) {{
        document.getElementById('opSection').style.display = '';
        document.getElementById('opGrid').style.display = '';

        const ops = evalData.operating_points;
        const thresholds = ops.map(o => o.threshold);

        // Sensitivity vs Specificity by threshold
        new Chart('sensSpecThreshChart', {{
            type: 'line',
            data: {{
                labels: thresholds,
                datasets: [
                    {{ label: 'Sensitivity', data: ops.map(o => o.sensitivity),
                      borderColor: '#4bc0c0', borderWidth: 2, tension: 0.3 }},
                    {{ label: 'Specificity', data: ops.map(o => o.specificity),
                      borderColor: '#ff9f40', borderWidth: 2, tension: 0.3 }},
                    {{ label: 'Precision', data: ops.map(o => o.precision),
                      borderColor: '#9966ff', borderWidth: 2, tension: 0.3 }},
                ]
            }},
            options: {{
                ...darkChartOpts,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Threshold', color: '#888' }},
                         ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }} }},
                    y: {{ ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }}, min: 0, max: 1 }},
                }}
            }}
        }});

        // Operating points table
        let tableHtml = '<table><tr><th>Threshold</th><th>Sens</th><th>Spec</th><th>Prec</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th></tr>';
        ops.forEach(o => {{
            const highlight = o.threshold === (evalData.threshold || 0.5) ? ' style="background:#1a2a4a"' : '';
            tableHtml += `<tr${{highlight}}><td>${{o.threshold.toFixed(1)}}</td><td>${{o.sensitivity.toFixed(4)}}</td><td>${{o.specificity.toFixed(4)}}</td><td>${{o.precision.toFixed(4)}}</td><td>${{o.tp}}</td><td>${{o.fp}}</td><td>${{o.fn}}</td><td>${{o.tn}}</td></tr>`;
        }});
        tableHtml += '</table>';
        if (evalData.optimal_threshold) {{
            tableHtml += `<p style="margin-top:10px;color:#888;font-size:12px">Optimal threshold (Youden J = ${{evalData.optimal_threshold.youden_j.toFixed(4)}}): <strong style="color:#00d4ff">${{evalData.optimal_threshold.threshold.toFixed(4)}}</strong></p>`;
        }}
        document.getElementById('opTable').innerHTML = tableHtml;
    }}

    // ==================== TRAINING SECTION ====================
    if (metrics.epoch && metrics.epoch.length > 0) {{
        document.getElementById('trainSection').style.display = '';
        document.getElementById('trainGrid').style.display = '';

        const epochs = metrics.epoch;
        const epochOpts = {{
            ...darkChartOpts,
            scales: {{
                x: {{ title: {{ display: true, text: 'Epoch', color: '#888' }},
                     ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }} }},
                y: {{ ticks: {{ color: '#666' }}, grid: {{ color: '#1f1f3a' }} }}
            }}
        }};

        new Chart('lossChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Train Loss', data: metrics.train_loss, borderColor: '#ff6384',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                    {{ label: 'Val Loss', data: metrics.val_loss, borderColor: '#36a2eb',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                ]
            }}, options: epochOpts
        }});

        new Chart('aucChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Train AUC', data: metrics.train_auc, borderColor: '#ff6384',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                    {{ label: 'Val AUC', data: metrics.val_auc, borderColor: '#36a2eb',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                ]
            }}, options: epochOpts
        }});

        new Chart('sensSpecChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Sensitivity', data: metrics.val_sensitivity, borderColor: '#4bc0c0',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                    {{ label: 'Specificity', data: metrics.val_specificity, borderColor: '#ff9f40',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                ]
            }}, options: epochOpts
        }});

        new Chart('prfChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Precision', data: metrics.val_precision, borderColor: '#9966ff',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                    {{ label: 'Recall', data: metrics.val_recall, borderColor: '#ff6384',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                    {{ label: 'F1', data: metrics.val_f1, borderColor: '#4bc0c0',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                ]
            }}, options: epochOpts
        }});

        new Chart('lrChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Learning Rate', data: metrics.learning_rate, borderColor: '#ffcd56',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                ]
            }}, options: {{ ...epochOpts,
                scales: {{ ...epochOpts.scales, y: {{ type: 'logarithmic', ticks: {{ color: '#666' }},
                          grid: {{ color: '#1f1f3a' }} }} }}
            }}
        }});

        new Chart('accChart', {{
            type: 'line', data: {{
                labels: epochs,
                datasets: [
                    {{ label: 'Train Accuracy', data: metrics.train_accuracy, borderColor: '#ff6384',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                    {{ label: 'Val Accuracy', data: metrics.val_accuracy, borderColor: '#36a2eb',
                      borderWidth: 1.5, pointRadius: 0, tension: 0.3 }},
                ]
            }}, options: epochOpts
        }});
    }}

    // ==================== OPTIMIZATION SECTION ====================
    if (optData && optData.original_size_mb) {{
        document.getElementById('optSection').style.display = '';
        document.getElementById('optGrid').style.display = '';

        let optHtml = '<h2>Model Optimization</h2>';
        optHtml += '<table>';
        optHtml += `<tr><td>Original Size</td><td>${{optData.original_size_mb}} MB</td></tr>`;
        optHtml += `<tr><td>Optimized Size</td><td>${{optData.optimized_size_mb}} MB</td></tr>`;
        optHtml += `<tr><td>Size Reduction</td><td>${{optData.reduction_pct}}%</td></tr>`;
        optHtml += `<tr><td>Graph Optimized</td><td>${{optData.graph_optimized ? 'Yes' : 'No'}}</td></tr>`;
        optHtml += `<tr><td>INT8 Quantized</td><td>${{optData.quantized ? 'Yes' : 'No'}}</td></tr>`;
        if (optData.benchmark) {{
            const bm = optData.benchmark;
            optHtml += `<tr><td>Avg Latency</td><td>${{bm.avg_latency_ms.toFixed(1)}} ms</td></tr>`;
            optHtml += `<tr><td>Throughput</td><td>${{bm.throughput_patches_per_sec.toFixed(0)}} patches/sec</td></tr>`;
            optHtml += `<tr><td>P95 Latency</td><td>${{bm.p95_latency_ms.toFixed(1)}} ms</td></tr>`;
        }}
        optHtml += '</table>';
        document.getElementById('optCard').innerHTML = optHtml;
    }}
    </script>
</body>
</html>"""

    return html


def save_dashboard(
    metrics_path: str | Path,
    output_path: str | Path | None = None,
    eval_results_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    optimization_path: str | Path | None = None,
) -> Path:
    """Generate and save the comprehensive dashboard HTML.

    Args:
        metrics_path: Path to metrics.json.
        output_path: Where to save the HTML. Defaults to same directory as metrics.
        eval_results_path: Optional path to eval_results.json.
        calibration_path: Optional path to calibration.json.
        optimization_path: Optional path to optimization_results.json.

    Returns:
        Path to the generated HTML file.
    """
    metrics_path = Path(metrics_path)
    if output_path is None:
        output_path = metrics_path.parent / "dashboard.html"
    else:
        output_path = Path(output_path)

    html = generate_dashboard_html(
        metrics_path,
        eval_results_path=eval_results_path,
        calibration_path=calibration_path,
        optimization_path=optimization_path,
    )
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Dashboard saved to {output_path}")
    return output_path
