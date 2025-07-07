"""Generate an HTML or PDF report summarizing evaluation results."""

import argparse
import json
import os
from datetime import datetime
from jinja2 import Template

try:
    from weasyprint import HTML  # optional PDF generation
except ImportError:  # pragma: no cover - optional dependency
    HTML = None


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { text-align: center; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
        img { max-width: 100%; height: auto; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Model Evaluation Report</h1>
    <p>Generated: {{ timestamp }}</p>

    <h2>Classification Report</h2>
    <table>
        <tr>
            <th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th>
        </tr>
        {% for label, scores in report.items() if label not in ['accuracy', 'macro avg', 'weighted avg'] %}
        <tr>
            <td>{{ label }}</td>
            <td>{{ '%.2f'|format(scores['precision']) }}</td>
            <td>{{ '%.2f'|format(scores['recall']) }}</td>
            <td>{{ '%.2f'|format(scores['f1-score']) }}</td>
            <td>{{ scores['support'] }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Confusion Matrix</h2>
    <img src="{{ confusion_img }}" alt="Confusion Matrix">

    <h2>ROC Curves</h2>
    <img src="{{ roc_img }}" alt="ROC Curves">

    <h2>Grad-CAM Samples</h2>
    {% for img in gradcam_images %}
        <img src="{{ img }}" alt="Grad-CAM">
    {% endfor %}

    {% if notes %}
    <h2>Notes</h2>
    <pre>{{ notes }}</pre>
    {% endif %}
</body>
</html>
"""


def generate_report(metrics_path, gradcam_dir, output_html, notes_path=None, output_pdf=None):
    with open(metrics_path) as f:
        metrics = json.load(f)

    report = metrics.get("classification_report", {})

    confusion_img = os.path.join(os.path.dirname(metrics_path), "confusion_matrix.png")
    roc_img = os.path.join(os.path.dirname(metrics_path), "roc_curves.png")
    gradcam_images = []
    if os.path.isdir(gradcam_dir):
        for fname in sorted(os.listdir(gradcam_dir)):
            if fname.lower().endswith((".png", ".jpg")):
                gradcam_images.append(os.path.join(gradcam_dir, fname))

    notes = None
    if notes_path and os.path.isfile(notes_path):
        with open(notes_path) as f:
            notes = f.read()

    template = Template(HTML_TEMPLATE)
    html = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        report=report,
        confusion_img=confusion_img,
        roc_img=roc_img,
        gradcam_images=gradcam_images,
        notes=notes,
    )

    with open(output_html, "w") as f:
        f.write(html)

    if output_pdf:
        if HTML is None:
            raise RuntimeError("weasyprint is required for PDF output")
        HTML(output_html).write_pdf(output_pdf)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("metrics", help="Path to metrics.json produced by eval_model")
    parser.add_argument("gradcam_dir", help="Directory containing Grad-CAM images")
    parser.add_argument("--notes", help="Optional notes text file")
    parser.add_argument("--html", default="report.html", help="Output HTML file")
    parser.add_argument("--pdf", help="Optional PDF output file")
    args = parser.parse_args()

    generate_report(args.metrics, args.gradcam_dir, args.html, args.notes, args.pdf)


if __name__ == "__main__":
    main()

