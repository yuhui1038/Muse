#!/usr/bin/env python3
"""
Summarize evaluation results and generate tables
Usage: python summarize.py --results_dir <results_directory> --output <output_file.md>
Features:
  - Summarize all metrics
  - Separate Chinese and English + merged statistics
  - Bold highest scores (bold lowest PER)
  - Append to history records
  - Generate visualization table images
"""
import argparse, json, os
from collections import defaultdict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Metric definitions
METRICS = {
    'songeval': ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness'],
    'audiobox': ['CE', 'CU', 'PC', 'PQ', 'Score'],
    'mulan_t': ['Mulan-T'],
    'per': ['PER']
}

ALL_METRICS = METRICS['audiobox'] + METRICS['songeval'] + METRICS['mulan_t'] + METRICS['per']

def load_results(results_dir):
    """Load all results"""
    data = defaultdict(dict)
    
    for metric_type in METRICS:
        metric_dir = os.path.join(results_dir, metric_type)
        if not os.path.exists(metric_dir): continue
        
        for f in os.listdir(metric_dir):
            if not f.endswith('.json') or '_details' in f: continue
            path = os.path.join(metric_dir, f)
            try:
                with open(path) as fp:
                    rec = json.load(fp)
                model = rec.get('model', f.replace('.json', ''))
                metrics = rec.get('metrics', {})
                for k, v in metrics.items():
                    data[model][k] = v
            except: pass
    
    return data

def merge_cn_en(data):
    """Merge Chinese and English results"""
    merged = {}
    base_models = set()
    
    for model in data:
        if model.endswith('_cn') or model.endswith('_en'):
            base_models.add(model[:-3])
    
    for base in base_models:
        cn, en = data.get(f"{base}_cn", {}), data.get(f"{base}_en", {})
        if not cn and not en: continue
        
        merged[base] = {}
        all_keys = set(cn.keys()) | set(en.keys())
        for k in all_keys:
            vals = [v for v in [cn.get(k), en.get(k)] if v is not None]
            if vals:
                merged[base][k] = sum(vals) / len(vals)
    
    return merged

def find_best(data, metric):
    """Find best value"""
    vals = [d.get(metric) for d in data.values() if d.get(metric) is not None]
    if not vals: return None
    return min(vals) if metric == 'PER' else max(vals)

def generate_markdown_table(data, title="Results"):
    """Generate Markdown table"""
    if not data: return ""
    
    # Find best values
    best = {m: find_best(data, m) for m in ALL_METRICS}
    
    # Table header
    lines = [f"## {title}", ""]
    header = "| Model | " + " | ".join(ALL_METRICS) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(ALL_METRICS)) + " |"
    lines.extend([header, sep])
    
    # Data rows
    for model in sorted(data.keys()):
        row = [model]
        for m in ALL_METRICS:
            val = data[model].get(m)
            if val is None:
                row.append("-")
            else:
                s = f"{val:.4f}"
                if best[m] is not None and abs(val - best[m]) < 1e-9:
                    s = f"**{s}**"
                row.append(s)
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)

def save_table_image(data, output_path):
    """Generate table image"""
    if not data: return
    
    # Prepare DataFrame
    rows = []
    for model in sorted(data.keys()):
        row = {'Model': model}
        for m in ALL_METRICS:
            row[m] = data[model].get(m)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Find best values
    best_indices = set()
    for col_idx, col in enumerate(ALL_METRICS):
        if col not in df.columns: continue
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        if numeric_series.isnull().all(): continue
        
        best_val = numeric_series.min() if col == 'PER' else numeric_series.max()
        for row_idx, val in enumerate(numeric_series):
            if pd.notna(val) and abs(val - best_val) < 1e-9:
                best_indices.add((row_idx, col_idx + 1))
    
    # Draw table
    num_rows, num_cols = len(df), len(df.columns)
    fig, ax = plt.subplots(figsize=(max(15, num_cols * 1.5), max(4, num_rows * 0.5 + 2)))
    ax.axis('off')
    
    # Prepare cell text
    cell_text = []
    for _, row in df.iterrows():
        row_text = [str(row['Model'])]
        for m in ALL_METRICS:
            val = row.get(m)
            row_text.append(f"{val:.4f}" if val is not None and pd.notna(val) else "-")
        cell_text.append(row_text)
    
    col_labels = ['Model'] + ALL_METRICS
    
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    
    # Styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        else:
            if (row - 1, col) in best_indices:
                cell.set_text_props(weight='bold', color='#d62728')
            cell.set_facecolor('#f2f2f2' if (row - 1) % 2 == 0 else 'white')
    
    plt.title("Evaluation Summary", fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    
    img_path = output_path.replace('.md', '.png')
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Table image: {img_path}")

def append_history(results_dir, data):
    """Append history records"""
    history_file = os.path.join(results_dir, "history.jsonl")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    record = {
        "timestamp": timestamp,
        "models": {}
    }
    
    for model, metrics in data.items():
        record["models"][model] = metrics
    
    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"History appended: {history_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default="summary.md")
    args = parser.parse_args()
    
    data = load_results(args.results_dir)
    merged = merge_cn_en(data)
    
    # Merge into data
    all_data = dict(data)
    all_data.update(merged)
    
    # Append history records
    append_history(args.results_dir, all_data)
    
    # Generate Markdown table
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = []
    output.append(f"# Baseline Evaluation Results Summary")
    output.append(f"\n**Update Time**: {timestamp}\n")
    output.append(generate_markdown_table(all_data, "All Results"))
    output.append("")
    
    # Write to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("\n".join(output))
    
    print("\n" + "\n".join(output))
    print(f"\nSaved: {args.output}")
    
    # Generate table image
    save_table_image(all_data, args.output)

if __name__ == "__main__":
    main()
