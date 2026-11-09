"""Figure 7 (athena_comparison.png) with the model-sensitivity runs.

Row 1: unchanged, the original Figure 7 row A (code copied verbatim from Plots/Regret_analysis.ipynb).
Row 2: reward of the sensitivity runs, 5 closed-team runs (black) and 5 open-team runs (red) per case.
Row 3: cost of the sensitivity runs, one metric per panel, cases on the x-axis, open vs closed bars
       (bar = mean over the runs with data, dots = individual runs):
       (left) iterations to acceptance, (center) LLM output tokens per run (thousands; totals printed to stdout), (right) wall-clock time per run.
       A run that was never accepted is drawn as a hollow dot and excluded from the bar mean
       (open RT run 4: cancelled at iteration 41, best score 95).
       Tokens and time come from usage/*_summary.json in Results/Sensitivity/<team>/<case>/runN/, or,
       for the early runs with the old layout (open IB runs 1-5, open Helmholtz runs 1-2), in
       run_logs/sensitivity/<team>/<case>/runN/usage/. Open RT run 4: gate_cut_iteration13.json (whole run).
SI figure (athena_sensitivity_regret_SI.png): cumulative regret vs T (no fits), same runs as row 2.
Failed runs (closed Helmholtz run 4: Strategist/Critic deadlock, no score; open RT run 2: Debugger
misread an import error as an environment failure) are drawn in rows 2-3 with an X and excluded from the
bar means; the SI regret averages the completed runs only.
Closed Helmholtz run 5: its score-100 trial (iteration 3) was printed before the xAI 429 crash but
not written to the report table, so it is appended here.
Run from Plots/:  python fig7_sensitivity.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import re
import pandas as pd
from pathlib import Path

def parse_trial_history(file_path):
    """
    Parse the Full Trial History table from an Experiment_Report.md file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the Experiment_Report.md file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Run, Model Description, Error, Integrity, 
        Accuracy, Details, Optimality, Total
        
    Examples
    --------
    >>> Files = {
    ...     'Burgers': '../Results/PINN_codes/Burgers_Periodic_SinPiX/2025_12_05_1414/Experiment_Report.md'
    ... }
    >>> df = parse_trial_history(Files['Burgers'])
    >>> # Plot error vs run
    >>> import matplotlib.pyplot as plt
    >>> plt.semilogy(df['Run'], df['Error'])
    >>> plt.xlabel('Run')
    >>> plt.ylabel('Error')
    >>> # Extract scores
    >>> print(df[['Run', 'Total']])
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the Full Trial History section
    history_pattern = r'## Full Trial History\s*\n(.*?)(?=\n##|\n###|$)'
    match = re.search(history_pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError("Could not find 'Full Trial History' section in file")
    
    history_section = match.group(1)
    
    # Split into lines and find the table
    lines = history_section.strip().split('\n')
    
    # Find header and separator lines
    header_idx = None
    sep_idx = None
    for i, line in enumerate(lines):
        if '| Run |' in line or '| :--- |' in line:
            if header_idx is None and '| Run |' in line:
                header_idx = i
            elif sep_idx is None and '| :--- |' in line:
                sep_idx = i
                break
    
    if header_idx is None or sep_idx is None:
        raise ValueError("Could not find table header in Full Trial History")
    
    # Parse header
    header_line = lines[header_idx]
    headers = [h.strip().replace('**', '') for h in header_line.split('|')[1:-1]]
    
    # Parse data rows (everything after separator)
    data_rows = []
    for line in lines[sep_idx + 1:]:
        line = line.strip()
        if not line or not line.startswith('|'):
            break
        
        # Split by | and clean up, removing markdown bold formatting
        cells = [cell.strip().replace('**', '') for cell in line.split('|')[1:-1]]
        if len(cells) == len(headers):
            data_rows.append(cells)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=headers)
    
    # Convert numeric columns
    df['Run'] = df['Run'].astype(int) + 1  # Add 1 to all run numbers since files start at 0
    
    # Convert Error column (handles scientific notation like 1.00e-06, percentages like 0.02%, and N/A)
    def parse_error(x):
        x = x.strip()
        if not x or x.upper() == 'N/A' or x.lower() == 'not applicable':
            return 0.0
        # Handle percentage values
        if '%' in x:
            return float(x.replace('%', '')) / 100.0
        # Handle scientific notation and regular floats (new reports may hold free text: not used here)
        try:
            return float(x)
        except ValueError:
            return float('nan')
    
    df['Error'] = df['Error'].apply(parse_error)
    
    # Convert score columns
    score_cols = ['Integrity', 'Accuracy', 'Details', 'Optimality', 'Total']
    for col in score_cols:
        df[col] = df[col].astype(int)
    
    return df


def extract_metrics(df, metric_col):
    """
    Extract a specific metric column for easy plotting.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by parse_trial_history
    metric_col : str
        Column name to extract (e.g., 'Error', 'Total', 'Accuracy')
        
    Returns
    -------
    runs : array
        Run numbers
    values : array
        Metric values
    """
    return df['Run'].values, df[metric_col].values


def plot_error_progression(df, ax=None):
    """
    Quick plot of error vs run number.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from parse_trial_history
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure.
        
    Returns
    -------
    ax : matplotlib axis
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(df['Run'], df['Error'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Run', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('Error Progression Across Runs', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_score_breakdown(df, ax=None):
    """
    Stacked bar chart showing score breakdown by component.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from parse_trial_history
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure.
        
    Returns
    -------
    ax : matplotlib axis
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    score_cols = ['Integrity', 'Accuracy', 'Details', 'Optimality']
    
    bottom = None
    for col in score_cols:
        if bottom is None:
            ax.bar(df['Run'], df[col], label=col)
            bottom = df[col].values
        else:
            ax.bar(df['Run'], df[col], bottom=bottom, label=col)
            bottom = bottom + df[col].values
    
    ax.set_xlabel('Run', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score Breakdown Across Runs', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


# Example usage
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import stats
import pandas as pd

# ============================================================================
# FONTSIZE CONFIGURATION
# ============================================================================
BASE_FONTSIZE = 16
TITLE_FONTSIZE = BASE_FONTSIZE + 2
LABEL_FONTSIZE = BASE_FONTSIZE
LEGEND_FONTSIZE = BASE_FONTSIZE - 4
TICK_FONTSIZE = BASE_FONTSIZE - 4

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = BASE_FONTSIZE
mpl.rcParams['figure.dpi'] = 100

# ============================================================================
# DATASETS
# ============================================================================

Files_SciML_analytic = {
    'BS1D':'../Results/NM_codes/BlackScholes_Dirichlet_CallPayoff/2026_01_11_0059/Experiment_Report.md',
    'VB1D': '../Results/NM_codes/Burgers_Periodic_SinPi/2025_12_17_1605/Experiment_Report.md',
    'DE2D':'../Results/NM_codes/AnisotropicDiffusion_Periodic_SinPlusCos/2026_01_11_0116/Experiment_Report.md',
    'IB2D': '../Results/NM_codes/Burgers_Periodic_SinSum/2025_12_15_2248/Experiment_Report.md',
    'W3D':'../Results/NM_codes/Wave_Neumann_GaussianSin/2026_01_11_1252/Experiment_Report.md',
}

Files_SciML = {
    'A': '../Results/PINN_codes/AllenCahn_Periodic_X2CosPiX/2025_12_05_1208/Experiment_Report.md',
    'VB': '../Results/PINN_codes/Burgers_Periodic_SinPiX/2025_12_05_1414/Experiment_Report.md',
    'P':'../Results/PINN_codes/Poisson_Dirichlet_Sin4pixSin4piy/2026_01_10_2319/Experiment_Report.md',
    'H': '../Results/PINN_codes/Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_05_1421/Experiment_Report.md',
    'KdV': '../Results/PINN_codes/KdV_Mixed_DerivedIC/2026_01_10_2026/Experiment_Report.md',
    'IB': '../Results/PINN_codes/Burgers_Periodic_SinPiX/2025_12_04_1621/Experiment_Report.md',
    'INS': '../Results/PINN_codes/Inverse_NavierStokes_Find_Re_HiddenField_v/2025_11_27_0131/Experiment_Report.md',
}

Files_Numerical = {
    'VB': '../Results/NM_codes/Burgers_Periodic_SinPi/2025_12_17_1554/Experiment_Report.md',
    'CH': '../Results/NM_codes/CahnHilliard_Periodic_Noise/2026_01_17_1918/Experiment_Report.md',
    'KH': '../Results/NM_codes/Euler_Periodic_SlipWalls_ShearLayer_ICs/2025_12_06_2249/Experiment_Report.md',
    'RT': '../Results/NM_codes/CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_2223/Experiment_Report.md',
    'cNS': '../Results/NM_codes/CompressibleNavierStokes_Periodic_NoSlipAdiabatic_SmoothedInterface/2025_12_08_0914/Experiment_Report.md',
    'INS': '../Results/NM_codes/Inverse_NavierStokes_Find_Re_HiddenField_v/2025_11_27_2250/Experiment_Report.md'
}


Analysis_IB2D = {
    'IB1': '../Results/NM_codes/Burgers_Periodic_SinSum/2025_12_15_2248/Experiment_Report.md',
    'IB2': '../Results/NM_codes/Burgers_Periodic_SinHalfSum/2025_12_15_2303/Experiment_Report.md',
    'IB3': '../Results/NM_codes/Burgers_Periodic_SinSumHalf/2025_12_23_1053/Experiment_Report.md',
    'IB4': '../Results/NM_codes/Burgers_Periodic_SinSumHalf/2025_12_23_1233/Experiment_Report.md',
    'IB5': '../Results/NM_codes/Burgers_Periodic_SinHalfSum/2025_11_21_0222/Experiment_Report.md',
}

Analysis_RT = {
    'RT1': '../Results/NM_codes/CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_0937/Experiment_Report.md',
    'RT2': '../Results/NM_codes/CompressibleEuler_Periodic_Reflective_SmoothedInterface/2025_12_05_2322/Experiment_Report.md',
    'RT3': '../Results/NM_codes/CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_1059/Experiment_Report.md',
    'RT4': '../Results/NM_codes/CompressibleEuler_PeriodicReflective_SmoothedInterface/2025_12_09_2125/Experiment_Report.md',
    'RT5': '../Results/NM_codes/CompressibleEuler_Periodic_Reflective_SmoothedInterface_ICs/2025_12_06_2223/Experiment_Report.md',
}

Analysis_HM = {
    'HM1': '../Results/PINN_codes/Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_05_1421/Experiment_Report.md',
    'HM2': '../Results/PINN_codes/Helmholtz_Periodic_SinPiA1A2/2025_12_07_1232/Experiment_Report.md',
    'HM3': '../Results/PINN_codes/Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_07_1331/Experiment_Report.md',
    'HM4': '../Results/PINN_codes/Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_07_1656/Experiment_Report.md',
    'HM5': '../Results/PINN_codes/Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_07_1709/Experiment_Report.md',
}

# ============================================================================
# PARSING & PROCESSING
# ============================================================================

dfs_analytic = {name: parse_trial_history(path) for name, path in Files_SciML_analytic.items()}
dfs_numerical = {name: parse_trial_history(path) for name, path in Files_Numerical.items()}
dfs_sciml = {name: parse_trial_history(path) for name, path in Files_SciML.items()}
dfs_ib2d = {name: parse_trial_history(path) for name, path in Analysis_IB2D.items()}
dfs_rt = {name: parse_trial_history(path) for name, path in Analysis_RT.items()}
dfs_hm = {name: parse_trial_history(path) for name, path in Analysis_HM.items()}

fig, axes = plt.subplots(3, 3, figsize=(12, 10.5))
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
colors = ['black', 'red', 'blue', 'green', 'orange', 'magenta']

# --- ROBUST REGRET FUNCTION ---
def compute_regret(dfs):
    all_scores = []
    
    for name, df in dfs.items():
        if df is None or df.empty or 'Total' not in df.columns:
            continue
        scores = df['Total'].tolist()
        if not scores: continue
        all_scores.append(scores)
    
    if not all_scores:
        return np.array([]), np.array([])
    
    max_len = max(len(s) for s in all_scores)
    padded_scores = []
    for scores in all_scores:
        if len(scores) < max_len:
            padded = scores + [scores[-1]] * (max_len - len(scores))
        else:
            padded = scores
        padded_scores.append(padded)
    
    padded_scores = np.array(padded_scores, dtype=float)
    
    # Use nanmean to safely average even if some runs failed (NaN)
    avg_scores = np.nanmean(padded_scores, axis=0)
    
    if np.isnan(avg_scores).any():
        df_avg = pd.Series(avg_scores)
        avg_scores = df_avg.ffill().fillna(0).values

    T = np.arange(1, max_len + 1)
    regret = np.cumsum(100 - avg_scores)
    
    return (T**(1/2)), regret

# ============================================================================
# FIXED PLOTTING FUNCTION (Handles Single-Point Runs)
# ============================================================================

def plot_panel(ax, dfs, title, colors, ncols=1, y_label='Reward', x_label='Total Iterations', show_label=False, show_legend=True, plot_type='reward'):
    print(f"Plotting {title}...")
    
    if plot_type == 'reward':
        for idx, (name, df) in enumerate(dfs.items()):
            if df is None or df.empty: continue
            runs = [0] + df['Run'].tolist()
            scores = [0] + df['Total'].tolist()
            ax.plot(runs, scores, marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    linewidth=1.5, markersize=5, label=name, alpha=0.8)
        
        if show_label:
            ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)

    else:
        # Calculate Regret
        T_sqrt, regret = compute_regret(dfs)
        
        if len(T_sqrt) > 0:
            # --- FIX 1: Add marker='o' ---
            # This ensures that if you only have 1 point (T=1), it still appears as a dot.
            ax.plot(T_sqrt, regret, color=colors[0], linewidth=2, 
                    marker='o', markersize=6, # <--- Added Marker
                    label=r'$\mathcal{{R}}(T)$', alpha=0.8)
            
            # --- Perfect Solver / Single Point Check ---
            if len(T_sqrt) == 1:
                print(f"  -> Single data point detected for {title}. Skipping regression.")
                # No regression possible with 1 point.
                # Just adjust axes so it looks nice
                ax.set_xlim(0, 2) 
            
            elif np.std(regret) < 1e-9: 
                # Perfect solver (Regret is all zeros)
                intercept = regret[0]
                fit_line = np.full_like(T_sqrt, intercept)
                ax.plot(T_sqrt, fit_line, color='gray', linestyle='--', linewidth=2,
                        label=r'$\mathcal{{R}} \approx 0$', alpha=0.7)
            else:
                # Normal regression
                mask = np.isfinite(T_sqrt) & np.isfinite(regret)
                if np.sum(mask) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(T_sqrt[mask], regret[mask])
                    fit_line = slope * T_sqrt + intercept
                    ax.plot(T_sqrt, fit_line, color='gray', linestyle='--', linewidth=2,
                            label=f'$\mathcal{{R}}={slope:.2f}\sqrt{{T}}{intercept:+.2f}$', alpha=0.7)
        
        if show_label:
            ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
    
    ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    
    # Legend location fix
    if show_legend:
        # If there is very little data (like the single point), move legend to avoid covering it
        loc = 'upper right' if plot_type == 'regret' and len(T_sqrt) < 3 else 'lower right'
        ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.95, ncols=ncols, loc=loc)
        
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if plot_type == 'reward':
        ax.set_ylim([0, 105])
    
    ax.tick_params(labelsize=TICK_FONTSIZE)


# ============================================================================
# SENSITIVITY RUNS (closed vs open team)
# ============================================================================
SENS = Path(__file__).resolve().parent.parent / 'Results' / 'Sensitivity'
RUN_LOGS = Path(__file__).resolve().parent.parent / 'run_logs' / 'sensitivity'
# Failed runs -> iteration at which they stopped. None left: closed Helmholtz run 4 and open RT run 2 are the
# approved reruns (failed originals in <team>/<case>_failed_attempts/). Open RT run 2 rerun: 75, 96, 98, 84.
FAILED = {}
EXTRA = {('closed', 'Helmholtz', 5): (4, 100)}   # (Run after the parser's +1, Total)
# Open RT run 4 ran before the score gate existed and was cancelled at iteration 41. With the gate
# (first score >= 90 at iteration 9, stop 5 iterations later) it stops after iteration 13 with best 95,
# i.e. accepted; Run index after the parser's +1 -> keep Run <= 14. Usage up to the cut: gate_cut_iteration13.json.
GATE_CUT = {('open', 'RT', 4): 14}
TEAM = {'open': ('Open', 'red', 'O', 's'), 'closed': ('Closed', 'black', 'C', 'o')}   # open drawn first, closed on top
LS = {'closed': '--', 'open': '-'}          # closed-team curves dashed
FIT_LS = {'closed': ':', 'open': '--'}      # sqrt(T) fits: dotted for closed so they stay distinct


def sens_trial_history(file_path):
    """Run and Total of the Full Trial History table. Unlike parse_trial_history (kept for row 1),
    a row whose description contains '|' is kept: Run is the first cell, Total the last."""
    m = re.search(r'## Full Trial History\s*\n(.*?)(?=\n##|\n###|$)', Path(file_path).read_text(), re.DOTALL)
    rows = []
    for line in m.group(1).strip().split('\n'):
        cells = [c.strip().replace('**', '') for c in line.strip().split('|')[1:-1]]
        if cells and cells[0].isdigit():
            rows.append({'Run': int(cells[0]) + 1, 'Total': int(cells[-1])})
    return pd.DataFrame(rows)


def sens_dfs(team, case, include_failed=False):
    out = {}
    for r in range(1, 6):
        failed = (team, case, r) in FAILED
        if failed and not include_failed:
            continue
        d = SENS / team / case / f'run{r}'
        reps = [p for p in d.rglob('Experiment_Report.md') if '/box/' not in str(p)]
        if not reps:
            if failed:   # failed before any evaluated trial (no report)
                out[r] = pd.DataFrame({'Run': pd.Series(dtype=int), 'Total': pd.Series(dtype=int)})
            continue
        df = max(((sens_trial_history(p), p.stat().st_mtime) for p in reps),
                 key=lambda x: (len(x[0]), x[1]))[0]
        if (team, case, r) in EXTRA:
            run, tot = EXTRA[(team, case, r)]
            df = pd.concat([df, pd.DataFrame([{'Run': run, 'Total': tot}])], ignore_index=True)
        if (team, case, r) in GATE_CUT:   # score gate applied: trials after the cut are dropped
            df = df[df['Run'] <= GATE_CUT[(team, case, r)]].reset_index(drop=True)
        out[r] = df
    return out


sens = {t: {c: sens_dfs(t, c) for c in ('IB', 'RT', 'Helmholtz')} for t in TEAM}            # completed runs (SI regret)
sens_all = {t: {c: sens_dfs(t, c, True) for c in ('IB', 'RT', 'Helmholtz')} for t in TEAM}  # + failed runs (rows 2-3)


def plot_sens_reward(ax, case, title, show_label=False):
    for team, (lab, col, tag, _) in TEAM.items():
        for r, df in sens_all[team][case].items():
            runs = [0] + df['Run'].tolist()
            scores = [0] + df['Total'].tolist()
            failed = (team, case, r) in FAILED
            ax.plot(runs, scores, marker=markers[(r - 1) % len(markers)], color=col,
                    linewidth=1.5, markersize=5, label=f'{tag}{r}' + (' (failed)' if failed else ''),
                    alpha=0.8, linestyle=LS[team])
            if failed:   # mark where the run stopped
                ax.plot(runs[-1], scores[-1], marker='X', ms=11, color=col, mec='white', mew=0.8, zorder=5)
    if show_label:
        ax.set_ylabel('Reward', fontsize=LABEL_FONTSIZE)
    ax.set_xlabel('Total Iterations', fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    if show_label:   # one legend (first column) for the whole row
        ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.95, ncols=2, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    ax.tick_params(labelsize=TICK_FONTSIZE)


# Runs that were never accepted: (team, case, run) -> iteration at which the run was stopped.
NOT_ACCEPTED = {}   # open RT run 4 is counted with the score gate (GATE_CUT)


def sens_usage(team, case, r):
    """(total LLM tokens, output LLM tokens, wall-clock seconds) of one run, or None if not tracked."""
    d = SENS / team / case / f'run{r}'
    summ = sorted((d / 'usage').glob('*_summary.json'))
    if not summ:   # early runs (old layout): summaries were written to run_logs/sensitivity/
        summ = sorted((RUN_LOGS / team / case / f'run{r}' / 'usage').glob('*_summary.json'))
    if summ:
        j = json.loads(summ[-1].read_text())
        c = j['by_kind']['chat']
        return c['total_tokens'], c['output_tokens'], j['wall_clock_s']
    gate = d / 'gate_cut_iteration13.json'
    if gate.exists():   # open RT run 4: usage up to the score-gate cut (end of iteration 13, 1h54m31s)
        w = json.loads(gate.read_text())['up_to_iteration_13']
        return w['total_tokens'], w['output_tokens'], 1 * 3600 + 54 * 60 + 31.0
    return None


# List API price per 1M OUTPUT tokens (USD; standard tier, prompts <= 200k), looked up 2026-11-06.
# Cost per run counts output tokens only (reasoning tokens are part of output_tokens).
PRICE_OUT = {
    'gemini-3.1-pro-preview': 12.0,    # Google
    'gpt-5.4': 15.0,                   # OpenAI
    'grok-4-1-fast-reasoning': 0.50,   # xAI
    'claude-sonnet-4-6': 15.0,         # Anthropic
    'claude-opus-5-5': 20.0,           # Anthropic (single-agent baseline)
    'glm-5.3': 4.40,                   # Z.ai
    'kimi-k3.1-pro': 10.0,             # Moonshot
    'deepseek-v4.1-flash:1017': 1.20,  # DeepSeek (standard rate, no off-peak discount)
    'nemotron-3-ultra': 2.20,          # NVIDIA API / OpenRouter
    'models/gemini-embedding-001': 0.0,   # embeddings produce no output tokens
}


def sens_out_by_model(team, case, r):
    """{model: output tokens} of one run: from the usage summary, else from the raw usage log."""
    d = SENS / team / case / f'run{r}'
    summ = sorted((d / 'usage').glob('*_summary.json')) or \
        sorted((RUN_LOGS / team / case / f'run{r}' / 'usage').glob('*_summary.json'))
    if summ:
        return {m: v.get('output_tokens', 0) or 0 for m, v in json.loads(summ[-1].read_text())['by_model'].items()}
    if not (d / 'gate_cut_iteration13.json').exists():   # run without a summary (e.g. still running)
        return None
    # open RT run 4: output tokens per model up to the score-gate cut
    rows = json.loads((d / 'gate_cut_iteration13.json').read_text())['by_model_up_to_iteration_13']
    return {x['model']: x['output_tokens'] for x in rows}


def sens_cost(team, case, r):
    out = sens_out_by_model(team, case, r)
    return None if out is None else sum(n * PRICE_OUT[m] for m, n in out.items()) / 1e6


# Baseline: Claude Code + Opus 5.5, up to 5 runs per case (Results/Sensitivity/bare_claude/<case>/run<N>);
# runs still in progress (no exit= in job_info.txt) are skipped
OPUS = ('Opus 5.5', 'green', 'A', '^')


# IB2D: the runs whose prompt is the User Request only (bare_claude/IB had the pipeline task label)
OPUS_DIR = {'IB': 'IB_request_only'}


def opus_metric(case, metric, r=1):
    d = SENS / 'bare_claude' / OPUS_DIR.get(case, case) / f'run{r}'
    if not (d / 'job_info.txt').exists():
        return None
    info = (d / 'job_info.txt').read_text()
    if 'exit=' not in info:
        return None
    res = [json.loads(l) for l in (d / 'events.jsonl').read_text().splitlines() if l.startswith('{')]
    out = sum(e['usage'].get('output_tokens', 0) for e in res if e.get('type') == 'result')
    t0, t1 = (pd.Timestamp(re.search(k + r'=(\S+)', info).group(1)) for k in ('start', 'end'))
    return {'cost': out * PRICE_OUT['claude-opus-5-5'] / 1e6, 'out_tokens': out / 1e3,
            'time': (t1 - t0).total_seconds() / 60.0}[metric]


CASES = [('IB', 'SciC (IB2D)'), ('RT', 'SciC (RT)'), ('Helmholtz', 'SciML (H)')]


def sens_metric(team, case, metric):
    """[(value, status)] over all runs of one team and case; status is 'ok', 'stopped' or 'failed'."""
    out = []
    for r, df in sens_all[team][case].items():
        key = (team, case, r)
        acc = 'failed' if key in FAILED else 'stopped' if key in NOT_ACCEPTED else 'ok'
        if metric == 'iters':
            v = {'ok': lambda: df['Run'].iloc[-1], 'stopped': lambda: NOT_ACCEPTED[key],
                 'failed': lambda: FAILED[key]}[acc]()
        elif metric == 'cost':
            v = sens_cost(team, case, r)
            if v is None:
                continue
        else:
            u = sens_usage(team, case, r)
            if u is None:
                continue
            v = {'tokens': u[0] / 1e6, 'out_tokens': u[1] / 1e3, 'time': u[2] / 60.0}[metric]
        out.append((float(v), acc))
    return out


def plot_sens_bars(ax, metric, ylabel, show_legend=False, show_rate=False, log=True, opus=True):
    rng = np.random.default_rng(0)
    width = 0.27 if opus else 0.36
    for k, (case, _) in enumerate(CASES):
        if opus:   # Opus runs per case, drawn right of the two teams (bar = mean, markers = runs)
            x0 = k + width
            vs = [v for v in (opus_metric(case, metric, r) for r in range(1, 6)) if v is not None]
            if vs:
                ax.bar(x0, np.mean(vs), width=width * 0.9, color=OPUS[1], alpha=0.25, edgecolor=OPUS[1], linewidth=1.2)
                for v in vs:
                    jx = x0 + (rng.uniform(-width * 0.25, width * 0.25) if len(vs) > 1 else 0.0)
                    ax.plot(jx, v, marker=OPUS[3], ms=7, ls='none', color=OPUS[1], alpha=0.9)
            print(f'{metric:6s} {case:9s} opus   n={len(vs)} values={[round(v, 2) for v in vs]}')
        for team, (lab, col, tag, mk) in TEAM.items():
            x0 = k + ((-width if team == 'open' else 0.0) if opus else (-width / 2 if team == 'open' else width / 2))
            vals = [(v, a) for v, a in sens_metric(team, case, metric) if a == 'ok']   # accepted runs only
            if not vals:
                ax.annotate('n/a', (x0, 0.04), xycoords=('data', 'axes fraction'), ha='center',
                            fontsize=LEGEND_FONTSIZE, color=col)
                continue
            acc = [v for v, a in vals if a == 'ok']
            if acc:
                ax.bar(x0, np.mean(acc), width=width * 0.9, color=col, alpha=0.25, edgecolor=col, linewidth=1.2)
            for v, a in vals:
                jx = x0 + rng.uniform(-width * 0.25, width * 0.25)
                if a == 'failed':
                    ax.plot(jx, v, marker='X', ms=9, ls='none', color=col, mec='white', mew=0.6, zorder=5)
                else:
                    ax.plot(jx, v, marker=mk, ms=6, ls='none', color=col,
                            markerfacecolor=col if a == 'ok' else 'white', markeredgewidth=1.3, alpha=0.9)
            if show_rate:   # accepted / total runs
                ax.annotate(f'{len(acc)}/{len(vals)}', (x0, max(v for v, _ in vals)), xytext=(0, 6),
                            textcoords='offset points', ha='center', fontsize=LEGEND_FONTSIZE - 1, color=col)
            print(f'{metric:6s} {case:9s} {team:6s} n={len(vals)} accepted={len(acc)} '
                  f'mean={np.mean(acc) if acc else float("nan"):.2f} values={[round(v, 2) for v, _ in vals]}')
    if log:
        ax.set_yscale('log')
    else:
        ax.set_ylim(-1.5, 52)
    ax.set_xticks(range(len(CASES)))
    ax.set_xticklabels(['IB2D', 'RT', 'H'], fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.tick_params(labelsize=TICK_FONTSIZE)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    if show_legend:
        from matplotlib.lines import Line2D
        h = [Line2D([], [], color=TEAM[t][1], marker=TEAM[t][3], ls='none', ms=6, label=TEAM[t][0]) for t in TEAM]
        if opus:
            h.append(Line2D([], [], color=OPUS[1], marker=OPUS[3], ls='none', ms=7, label=OPUS[0]))
        if log:   # headroom above the data so the legend covers no marker
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 4)
        ax.legend(handles=h, fontsize=LEGEND_FONTSIZE, framealpha=0.95, loc='upper center', ncols=3,
                  columnspacing=0.4, handletextpad=0.1, handlelength=1.0, borderpad=0.3, borderaxespad=0.3)


def plot_sens_regret(ax, case, title, show_label=False, show_legend=False):
    """SI: cumulative regret (mean over the runs of each team) against T, without fitted curves."""
    for team, (lab, col, tag, mk) in TEAM.items():
        dfs = sens[team][case]   # completed runs only
        T_sqrt, regret = compute_regret(dfs)
        T = np.rint(T_sqrt ** 2).astype(int)
        ax.plot(T, regret, color=col, linewidth=2, marker=mk, markersize=6,
                label=f'{lab} (mean of {len(dfs)})', alpha=0.8, linestyle=LS[team])
        print(f'{case:9s} {team:6s} runs={sorted(dfs)} finals={[int(d["Total"].iloc[-1]) for d in dfs.values()]} regret_end={regret[-1]:.1f}')
    if show_label:
        ax.set_ylabel(r'Regret ($\mathcal{R}$)', fontsize=LABEL_FONTSIZE)
    ax.set_xlabel(r'$T$', fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ax.get_xlim()[1] - ax.get_xlim()[0] < 1:   # single point (IB2D): show T = 0..2
        ax.set_xlim(0, 2)
    if show_legend:
        ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.95, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=TICK_FONTSIZE)


# --- EXECUTION ---
# Row 1 (original, unchanged)
plot_panel(axes[0, 0], dfs_analytic, 'SciC (Analytical)', colors, x_label='Total Iterations', y_label='Reward', show_label=True, plot_type='reward')
plot_panel(axes[0, 1], dfs_numerical, 'SciC (Numerical)', colors, x_label='Total Iterations', y_label='Reward', show_label=False, plot_type='reward')
plot_panel(axes[0, 2], dfs_sciml, 'SciML', colors, x_label='Total Iterations', y_label='Reward', show_label=False, ncols=2, plot_type='reward')

# Row 2: reward of the sensitivity runs
plot_sens_reward(axes[1, 0], 'IB', 'SciC (IB2D) ', show_label=True)
plot_sens_reward(axes[1, 1], 'RT', 'SciC (RT)')
plot_sens_reward(axes[1, 2], 'Helmholtz', 'SciML (H) ')

# Row 3: cost of the sensitivity runs (one metric per panel)
plot_sens_bars(axes[2, 0], 'cost', 'Cost per run (USD)', show_legend=True)
plot_sens_bars(axes[2, 1], 'out_tokens', 'Output tokens per run (k)')
plot_sens_bars(axes[2, 2], 'time', 'Wall-clock time (min)')

for case, _ in CASES:   # totals (input + output) for the caption / SI table
    for team in TEAM:
        v = [x for x, a in sens_metric(team, case, 'tokens') if a]
        print(f'total tokens (M, accepted runs) {case:9s} {team:6s} mean={np.mean(v):.2f}')

plt.tight_layout()
plt.savefig('athena_comparison_sensitivity.png', dpi=300, bbox_inches='tight')
print('saved athena_comparison_sensitivity.png')

# SI figure: cumulative regret against T, no fits
fig_si, axes_si = plt.subplots(1, 3, figsize=(12, 3.6))
for k, (case, title) in enumerate(CASES):
    plot_sens_regret(axes_si[k], case, title, show_label=(k == 0), show_legend=(k == 1))
fig_si.tight_layout()
fig_si.savefig('athena_sensitivity_regret_SI.png', dpi=300, bbox_inches='tight')
print('saved athena_sensitivity_regret_SI.png')
