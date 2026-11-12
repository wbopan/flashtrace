# %%
# Token-level Heatmap Visualization for Multi-hop Attribution
# HTML rendering with background color intensity
# Generates multi-row visualization for paper figures

import numpy as np
from pathlib import Path
import argparse


# %%
# === Hardcoded best samples (with tok + hop fields) ===
BEST_SAMPLES = [
    # (path, display_name)
    ('math_ifr_multi_hop_both/ex_000017.npz', 'Math #17'),
    ('math_ifr_multi_hop_both/ex_000054.npz', 'Math #54'),
    ('math_ifr_multi_hop_both/ex_000000.npz', 'Math #00'),
]


# %%
# === Sample Finder: Find best visualization samples ===
def compute_balance_score(lengths):
    """
    Compute balance score for input/cot/output lengths.
    Lower score = more balanced (using coefficient of variation).
    """
    lengths = np.array(lengths, dtype=float)
    if np.mean(lengths) == 0:
        return float('inf')
    # Coefficient of variation: std / mean
    return np.std(lengths) / np.mean(lengths)


def find_best_samples(search_dirs, top_k=10, mas_threshold=None, balance_threshold=None,
                      min_length=100, max_length=2000):
    """
    Find best visualization samples across directories.

    Args:
        search_dirs: List of directories to search (can be glob patterns)
        top_k: Number of top samples to return
        mas_threshold: Maximum MAS value (lower is better)
        balance_threshold: Maximum balance score (lower is more balanced)
        min_length: Minimum total sequence length
        max_length: Maximum total sequence length

    Returns:
        List of (path, mas, balance_score, lengths) tuples, sorted by combined score
    """
    from pathlib import Path
    import glob

    candidates = []

    # Expand directories
    all_dirs = []
    for d in search_dirs:
        if '*' in d:
            all_dirs.extend(glob.glob(d))
        else:
            all_dirs.append(d)

    for search_dir in all_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            print(f"Warning: {search_dir} does not exist, skipping...")
            continue

        npz_files = list(search_path.rglob('*.npz'))
        print(f"Scanning {search_dir}: found {len(npz_files)} npz files")

        for npz_path in npz_files:
            try:
                data = np.load(npz_path, allow_pickle=True)

                # Extract MAS
                if 'mas' in data:
                    mas = float(data['mas'])
                else:
                    # Compute from attr if not present
                    attr = data['attr']
                    mas = np.mean(np.abs(attr))

                # Extract lengths
                if 'length' in data:
                    lengths = data['length']
                else:
                    span_in = data['span_in']
                    span_cot = data['span_cot']
                    span_out = data['span_out']
                    lengths = [
                        span_in[1] - span_in[0] + 1,
                        span_cot[1] - span_cot[0] + 1,
                        span_out[1] - span_out[0] + 1
                    ]

                total_length = sum(lengths)
                balance_score = compute_balance_score(lengths)

                # Apply filters
                if min_length and total_length < min_length:
                    continue
                if max_length and total_length > max_length:
                    continue
                if mas_threshold and mas > mas_threshold:
                    continue
                if balance_threshold and balance_score > balance_threshold:
                    continue

                candidates.append({
                    'path': str(npz_path),
                    'mas': mas,
                    'balance': balance_score,
                    'lengths': list(lengths),
                    'total_length': total_length
                })

            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                continue

    if not candidates:
        print("No samples found matching criteria!")
        return []

    # Normalize MAS and balance scores for combined ranking
    mas_values = np.array([c['mas'] for c in candidates])
    balance_values = np.array([c['balance'] for c in candidates])

    mas_min, mas_max = mas_values.min(), mas_values.max()
    bal_min, bal_max = balance_values.min(), balance_values.max()

    for c in candidates:
        # Normalize to [0, 1], lower is better
        mas_norm = (c['mas'] - mas_min) / (mas_max - mas_min + 1e-8)
        bal_norm = (c['balance'] - bal_min) / (bal_max - bal_min + 1e-8)
        # Combined score (equal weight)
        c['score'] = 0.5 * mas_norm + 0.5 * bal_norm

    # Sort by combined score (lower is better)
    candidates.sort(key=lambda x: x['score'])

    return candidates[:top_k]


def print_sample_report(samples):
    """Print a formatted report of found samples."""
    print("\n" + "=" * 80)
    print("Best Visualization Samples (sorted by combined MAS + Balance score)")
    print("=" * 80)
    print(f"{'Rank':<5} {'MAS':<8} {'Balance':<8} {'Lengths (in/cot/out)':<25} {'Path'}")
    print("-" * 80)

    for i, s in enumerate(samples, 1):
        lengths_str = f"{s['lengths'][0]}/{s['lengths'][1]}/{s['lengths'][2]}"
        # Truncate path for display
        path_display = s['path']
        if len(path_display) > 50:
            path_display = "..." + path_display[-47:]
        print(f"{i:<5} {s['mas']:<8.4f} {s['balance']:<8.4f} {lengths_str:<25} {path_display}")

    print("=" * 80)


# %%
# Color utilities - using Starry blue palette
def value_to_blue(val):
    """Map 0-1 value to white-blue color. Returns rgb string."""
    # More vivid blue: #0052cc = rgb(0, 82, 204)
    r = int(255 - val * (255 - 0))
    g = int(255 - val * (255 - 82))
    b = int(255 - val * (255 - 204))
    return f"rgb({r}, {g}, {b})"


def value_to_diverging(val):
    """Map -1 to 1 value to red-white-green. Returns rgb string."""
    # More vivid colors - Red: #dc2626, Green: #059669
    if val < 0:
        # Red side: -1 = full red, 0 = white
        intensity = -val
        r = int(255 - intensity * (255 - 220))
        g = int(255 - intensity * (255 - 38))
        b = int(255 - intensity * (255 - 38))
    else:
        # Green side: 0 = white, 1 = full green
        intensity = val
        r = int(255 - intensity * (255 - 5))
        g = int(255 - intensity * (255 - 150))
        b = int(255 - intensity * (255 - 105))
    return f"rgb({r}, {g}, {b})"


def get_text_color(val, diverging=False):
    """Return black or white text based on background luminance."""
    if diverging:
        if abs(val) < 0.4:
            return "#333333"
        elif val < 0:
            return "#ffffff" if abs(val) > 0.6 else "#333333"
        else:
            return "#ffffff" if val > 0.6 else "#333333"
    else:
        return "#ffffff" if val > 0.5 else "#333333"


# %%
def render_tokens_html(tokens, values, span_in, span_cot, span_out,
                       diverging=False, highlight_region=None):
    """Render tokens as HTML spans with background colors."""
    html_parts = []

    # Add Input divider at the start
    html_parts.append('<div class="region-divider"><span class="region-label">Input</span></div>')

    for i, token in enumerate(tokens):
        val = values[i]

        # Determine region
        if i <= span_in[1]:
            region = 'input'
        elif i <= span_cot[1]:
            region = 'cot'
        else:
            region = 'output'

        # Get colors
        if diverging:
            bg_color = value_to_diverging(val)
        else:
            bg_color = value_to_blue(val)

        text_color = get_text_color(val, diverging)

        # Escape HTML special characters and remove newlines
        token_str = str(token)
        token_str = token_str.replace('\n', '').replace('\r', '')
        token_str = token_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        token_str = token_str.replace('-', '&#8209;')  # Non-breaking hyphen
        if token_str.startswith(' '):
            token_str = '&nbsp;' + token_str[1:]

        # Add horizontal divider and highlight box for regions
        if i == span_cot[0]:
            html_parts.append('<div class="region-divider"><span class="region-label">CoT</span></div>')
            if highlight_region == 'cot':
                html_parts.append('<div class="highlight-box">')
        elif i == span_out[0]:
            if highlight_region == 'cot':
                html_parts.append('</div>')
            html_parts.append('<div class="region-divider"><span class="region-label">Output</span></div>')
            if highlight_region == 'output':
                html_parts.append('<div class="highlight-box">')

        html_parts.append(
            f'<span class="token {region}" style="background-color: {bg_color}; '
            f'color: {text_color};">{token_str}</span>'
        )

    # Close highlight box at the end if output was highlighted
    if highlight_region == 'output':
        html_parts.append('</div>')

    return ''.join(html_parts)


# %%
def generate_row_html(npz_path, row_label):
    """Generate HTML for a single sample row (3 panels: Hop1, Hop2, Diff)."""
    data = np.load(npz_path, allow_pickle=True)

    tok = data['tok']
    hop = data['hop']
    span_in = data['span_in']
    span_cot = data['span_cot']
    span_out = data['span_out']

    hop1 = hop[0]
    hop2 = hop[1]
    diff = hop2 - hop1

    # Normalize
    hop1_norm = hop1 / np.max(hop1) if np.max(hop1) > 0 else hop1
    hop2_norm = hop2 / np.max(hop2) if np.max(hop2) > 0 else hop2
    max_abs_diff = np.max(np.abs(diff))
    diff_norm = diff / max_abs_diff if max_abs_diff > 0 else diff

    # Generate panels
    hop1_html = render_tokens_html(tok, hop1_norm, span_in, span_cot, span_out, highlight_region='output')
    hop2_html = render_tokens_html(tok, hop2_norm, span_in, span_cot, span_out, highlight_region='cot')
    diff_html = render_tokens_html(tok, diff_norm, span_in, span_cot, span_out, diverging=True)

    return f'''
        <div class="sample-row">
            <div class="row-label">{row_label}</div>
            <div class="panels">
                <div class="panel">
                    <div class="panel-title">Hop 1</div>
                    <div class="token-container">{hop1_html}</div>
                </div>
                <div class="panel">
                    <div class="panel-title">Hop 2</div>
                    <div class="token-container">{hop2_html}</div>
                </div>
                <div class="panel">
                    <div class="panel-title">Δ (Hop2 − Hop1)</div>
                    <div class="token-container">{diff_html}</div>
                </div>
            </div>
        </div>
    '''


# %%
def generate_multi_sample_html(samples):
    """Generate full HTML with multiple sample rows."""
    rows_html = ""
    for path, label in samples:
        try:
            rows_html += generate_row_html(path, label)
            print(f"Generated: {label} from {path}")
        except Exception as e:
            print(f"Error generating {path}: {e}")

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-hop Token Attribution Heatmap</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Libertinus Sans', 'Helvetica Neue', Arial, sans-serif;
            background: #ffffff;
            padding: 20px;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .sample-row {{
            display: flex;
            flex-direction: row;
            gap: 10px;
            background: #fcfcfc;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 12px;
        }}
        .row-label {{
            writing-mode: vertical-rl;
            text-orientation: mixed;
            transform: rotate(180deg);
            font-size: 12px;
            font-weight: bold;
            color: #666;
            padding: 8px 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 30px;
        }}
        .panels {{
            display: flex;
            flex-direction: row;
            gap: 12px;
            flex: 1;
        }}
        .panel {{
            flex: 1;
            background: transparent;
            padding: 8px;
            max-width: 400px;
        }}
        .panel-title {{
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #333;
            text-align: center;
        }}
        .token-container {{
            line-height: 1.5;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            font-size: 9px;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }}
        .token {{
            display: inline;
            padding: 1px 0px;
            border-radius: 2px;
        }}
        .region-divider {{
            display: flex;
            align-items: center;
            margin: 4px 0;
            width: 100%;
        }}
        .region-divider::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: #ccc;
            margin-left: 4px;
        }}
        .region-label {{
            font-size: 7px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .highlight-box {{
            outline: 2px solid #f59e0b;
            outline-offset: 2px;
            border-radius: 4px;
            background: rgba(245, 158, 11, 0.03);
        }}
        .legend-row {{
            display: flex;
            justify-content: center;
            gap: 40px;
            padding: 15px;
            background: #f8f8f8;
            border-radius: 4px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            color: #666;
        }}
        .colorbar {{
            height: 16px;
            width: 80px;
            border-radius: 2px;
            border: 1px solid #ccc;
        }}
        .colorbar-heat {{
            background: linear-gradient(to right, rgb(255, 255, 255), rgb(0, 82, 204));
        }}
        .colorbar-diff {{
            background: linear-gradient(to right, rgb(220, 38, 38), rgb(255, 255, 255), rgb(5, 150, 105));
        }}
        .target-box {{
            width: 14px;
            height: 14px;
            border: 2px solid #f59e0b;
            border-radius: 2px;
            background: rgba(245, 158, 11, 0.05);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="legend-row">
            <div class="legend-item">
                <span>Attribution:</span>
                <div class="colorbar colorbar-heat"></div>
                <span>0 → max</span>
            </div>
            <div class="legend-item">
                <span>Δ Change:</span>
                <div class="colorbar colorbar-diff"></div>
                <span>−max → +max</span>
            </div>
            <div class="legend-item">
                <div class="target-box"></div>
                <span>Attribution Target</span>
            </div>
        </div>
        {rows_html}
    </div>
</body>
</html>
'''
    return html_content


# %%
# Command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heatmap visualization')
    parser.add_argument('--find', action='store_true', help='Run sample finder mode')
    parser.add_argument('--dirs', nargs='+', default=['data/*'],
                        help='Directories to search (supports glob patterns)')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top samples')
    parser.add_argument('--mas-threshold', type=float, default=None,
                        help='Maximum MAS value')
    parser.add_argument('--balance-threshold', type=float, default=None,
                        help='Maximum balance score')
    parser.add_argument('--min-length', type=int, default=100,
                        help='Minimum total sequence length')
    parser.add_argument('--max-length', type=int, default=2000,
                        help='Maximum total sequence length')
    parser.add_argument('--generate', action='store_true',
                        help='Generate multi-sample HTML from hardcoded samples')

    args = parser.parse_args()

    if args.find:
        samples = find_best_samples(
            search_dirs=args.dirs,
            top_k=args.top_k,
            mas_threshold=args.mas_threshold,
            balance_threshold=args.balance_threshold,
            min_length=args.min_length,
            max_length=args.max_length
        )
        print_sample_report(samples)
    elif args.generate:
        # Generate HTML from hardcoded samples
        html_content = generate_multi_sample_html(BEST_SAMPLES)
        output_path = Path('token_heatmap.html')
        output_path.write_text(html_content)
        print(f"Saved to {output_path}")

        # Open in browser
        import subprocess
        subprocess.run(['open', str(output_path)])
    else:
        # Default: generate HTML
        html_content = generate_multi_sample_html(BEST_SAMPLES)
        output_path = Path('token_heatmap.html')
        output_path.write_text(html_content)
        print(f"Saved to {output_path}")

        import subprocess
        subprocess.run(['open', str(output_path)])

# %%
