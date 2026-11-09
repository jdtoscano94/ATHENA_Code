"""SI figure: accepted solution vs first acceptable solution (first trial with score >= 90).
Same runs, panels, accounting (output tokens only), and style as Figure 8(C) (Plots/fig7_sensitivity.py, row 3):
(left) cost per run, (center) output tokens per run, (right) wall-clock time per run.
Filled markers / solid bars: accepted solution. Hollow markers / hatched bars: cut at the first trial with score >= 90.
Row 2 (Helmholtz PINN): relative L2 error at the best checkpoint of each iteration against the cumulative
LLM cost when that iteration is evaluated (same pricing as row 1) (the definition used in the
paper), read from the 'Minimum achieved' / 'Best-ever' RL2 line of each iteration's .py.out; (left) closed runs,
(center) open runs, (right) accepted vs first score >= 90, with the Opus 5.5 range of Supplementary Table B5.
Row 1 (PIML benchmarks of Figure 5, closed team): relative L2 error of each scored iteration of the reported run,
with the Opus 5.5 errors of Figure 5 as green horizontal lines (red solid: first case, black dashed: second case).
Run from Plots/:  python fig_first90_SI.py"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

BASE_FONTSIZE = 16                      # as in fig7_sensitivity.py
LABEL_FONTSIZE = BASE_FONTSIZE
LEGEND_FONTSIZE = BASE_FONTSIZE - 4
TITLE_FONTSIZE = BASE_FONTSIZE + 2
TICK_FONTSIZE = BASE_FONTSIZE - 4
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = BASE_FONTSIZE
mpl.rcParams['figure.dpi'] = 100

ROOT = Path(__file__).resolve().parent.parent
SENS = ROOT/'Results'/'Sensitivity'; RUN_LOGS = ROOT/'run_logs'/'sensitivity'
PRICE = {'gemini-3.1-pro-preview': 12.0, 'gpt-5.4': 15.0, 'grok-4-1-fast-reasoning': 0.50, 'claude-sonnet-4-6': 15.0,
         'claude-opus-5-5': 20.0, 'glm-5.3': 4.40, 'kimi-k3.1-pro': 10.0, 'deepseek-v4.1-flash:1017': 1.20,
         'nemotron-3-ultra': 2.20, 'models/gemini-embedding-001': 0.0}
EXTRA = {('closed', 'Helmholtz', 5): (4, 100)}   # score-100 trial printed before the xAI 429 crash
GATE_CUT = {('open', 'RT', 4): 14}               # score gate applied retroactively, as in Figure 8
TEAM = {'open': ('Open', 'red', 's'), 'closed': ('Closed', 'black', 's')}   # same marker, teams differ by color
LS = {'closed': '--', 'open': '-'}   # as in Figure 8(B)
OPUS = ('Opus 5.5', 'green', '^')
OPUS_DIR = {'IB': 'IB_request_only'}
CASES = [('IB', 'IB2D'), ('RT', 'RT'), ('Helmholtz', 'H')]


def hist(p):
    m = re.search(r'## Full Trial History\s*\n(.*?)(?=\n##|\n###|$)', p.read_text(), re.DOTALL); rows = []
    for line in m.group(1).strip().split('\n'):
        c = [x.strip().replace('**', '') for x in line.strip().split('|')[1:-1]]
        if c and c[0].isdigit():
            rows.append((int(c[0]) + 1, int(c[-1])))
    return rows


def run_metrics(team, case, r):
    d = SENS/team/case/f'run{r}'
    h = max((hist(p) for p in d.rglob('Experiment_Report.md') if '/box/' not in str(p)), key=len)
    if (team, case, r) in EXTRA: h.append(EXTRA[(team, case, r)])
    if (team, case, r) in GATE_CUT: h = [x for x in h if x[0] <= GATE_CUT[(team, case, r)]]
    udir = d/'usage' if list((d/'usage').glob('*[0-9].jsonl')) else RUN_LOGS/team/case/f'run{r}'/'usage'
    ev = [json.loads(l) for f in sorted(udir.glob('*[0-9].jsonl')) for l in f.read_text().splitlines() if l.startswith('{')]
    chats = [e for e in ev if e.get('kind') == 'chat']
    evals = [e for e in chats if e.get('role') == 'register_agent_parser'] or [e for e in chats if e.get('role') == 'advisor_agent']

    def upto(t):
        sel = [e for e in chats if t is None or e['t_rel_s'] <= t]
        return (sum((e.get('output_tokens') or 0) * PRICE[e['model']] for e in sel) / 1e6,
                sum(e.get('output_tokens') or 0 for e in sel) / 1e3)

    k = next(i for i, x in enumerate(h) if x[1] >= 90)
    t90 = evals[k]['t_rel_s']
    c90, o90 = upto(t90)
    summ = sorted(udir.glob('*_summary.json'))
    if summ:
        j = json.loads(summ[-1].read_text())
        cacc = sum((v.get('output_tokens', 0) or 0) * PRICE[m] for m, v in j['by_model'].items()) / 1e6
        oacc, tacc = j['by_kind']['chat']['output_tokens'] / 1e3, j['wall_clock_s']
    else:   # open RT run 4: usage up to the score-gate cut (end of iteration 13, 1h54m31s), as in Figure 8
        w = json.loads((d/'gate_cut_iteration13.json').read_text())
        cacc = sum(x['output_tokens'] * PRICE[x['model']] for x in w['by_model_up_to_iteration_13']) / 1e6
        oacc, tacc = w['up_to_iteration_13']['output_tokens'] / 1e3, 1 * 3600 + 54 * 60 + 31.0
    return {'accepted': {'cost': cacc, 'out_tokens': oacc, 'time': tacc / 60},
            'first90': {'cost': c90, 'out_tokens': o90, 'time': t90 / 60}}


def opus(case, r):   # same accounting as opus_metric in fig7_sensitivity.py
    d = SENS/'bare_claude'/OPUS_DIR.get(case, case)/f'run{r}'
    if not (d/'job_info.txt').exists(): return None
    info = (d/'job_info.txt').read_text()
    if 'exit=' not in info: return None
    res = [json.loads(l) for l in (d/'events.jsonl').read_text().splitlines() if l.startswith('{')]
    out = sum(e['usage'].get('output_tokens', 0) for e in res if e.get('type') == 'result')
    t0, t1 = (pd.Timestamp(re.search(k + r'=(\S+)', info).group(1)) for k in ('start', 'end'))
    return {'cost': out * PRICE['claude-opus-5-5'] / 1e6, 'out_tokens': out / 1e3, 'time': (t1 - t0).total_seconds() / 60}


M = {(t, c, r): run_metrics(t, c, r) for t in TEAM for c, _ in CASES for r in range(1, 6)}
OP = {c: [v for v in (opus(c, r) for r in range(1, 6)) if v] for c, _ in CASES}
for c, _ in CASES:
    for metric in ('cost', 'out_tokens', 'time'):
        for t in TEAM:
            a = [M[(t, c, r)]['accepted'][metric] for r in range(1, 6)]
            f = [M[(t, c, r)]['first90'][metric] for r in range(1, 6)]
            print(f'{metric:10s} {c:9s} {t:6s} accepted {min(a):.2f}-{max(a):.2f} (mean {np.mean(a):.2f}) | first>=90 {min(f):.2f}-{max(f):.2f} (mean {np.mean(f):.2f})')
        o = [d[metric] for d in OP[c]]
        print(f'{metric:10s} {c:9s} opus   {min(o):.2f}-{max(o):.2f} (mean {np.mean(o):.2f})')


def plot_bars(ax, metric, ylabel):
    rng = np.random.default_rng(0)
    W = 0.16
    order = [('open', 'accepted'), ('open', 'first90'), ('closed', 'accepted'), ('closed', 'first90')]
    for k, (case, _) in enumerate(CASES):
        for j, (team, state) in enumerate(order):
            x0 = k + (j - 2) * W
            lab, col, mk = TEAM[team]
            vs = [M[(team, case, r)][state][metric] for r in range(1, 6)]
            ax.bar(x0, np.mean(vs), width=W * 0.9, color=col, alpha=0.25 if state == 'accepted' else 0.08,
                   edgecolor=col, linewidth=1.2, hatch=None if state == 'accepted' else '///')
            for v in vs:
                ax.plot(x0 + rng.uniform(-W * 0.25, W * 0.25), v, marker=mk, ms=6, ls='none', color=col,
                        markerfacecolor=col if state == 'accepted' else 'white', markeredgewidth=1.3, alpha=0.9)
        x0 = k + 2 * W
        vs = [d[metric] for d in OP[case]]
        ax.bar(x0, np.mean(vs), width=W * 0.9, color=OPUS[1], alpha=0.25, edgecolor=OPUS[1], linewidth=1.2)
        for v in vs:
            ax.plot(x0 + rng.uniform(-W * 0.25, W * 0.25), v, marker=OPUS[2], ms=7, ls='none', color=OPUS[1], alpha=0.9)
    ax.set_yscale('log')
    ax.set_xticks(range(len(CASES)))
    ax.set_xticklabels([c for _, c in CASES], fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 4)   # headroom for the legend, as in Figure 8


# --- Row 2: Helmholtz relative L2 error per iteration ------------------------------------------
ACCEPTED_RUN = {('closed', 3): 2}   # Advisor: "Exploration complete. Best model was Run 2. Stopping."
OPUS_H_RANGE = (2.8e-5, 1.1e-4)     # Claude Code (Opus 5.5), Supplementary Table B5
ERR_RE = re.compile(r'(?:Minimum achieved\s+RL2 error|Best-ever minimum RL2 error[^:]*|Best-ever RL2 error):\s*([0-9.eE+-]+)')


def helm_errors(team, r):
    """{iteration (Run + 1): (best-checkpoint RL2, score)} of one Helmholtz run."""
    d = SENS/team/'Helmholtz'/f'run{r}'
    h = dict(max((hist(p) for p in d.rglob('Experiment_Report.md') if '/box/' not in str(p)), key=len))
    if (team, 'Helmholtz', r) in EXTRA: h[EXTRA[(team, 'Helmholtz', r)][0]] = EXTRA[(team, 'Helmholtz', r)][1]
    outs = sorted((p for p in d.rglob('*.py.out') if '/box/' not in str(p)
                   and re.fullmatch(r'\d{4}_\d\d_\d\d_\d{4}', p.parent.name)), key=lambda p: p.parent.name)
    res = {}
    for o in outs:
        m = ERR_RE.findall(o.read_text(errors='ignore'))
        rep = o.parent/'Experiment_Report.md'
        rows = hist(rep) if rep.exists() else []
        it = rows[-1][0] if rows else (max(h) if (team, 'Helmholtz', r) in EXTRA and not rows and m else None)
        if m and it is not None:
            res[it] = (float(m[-1]), h.get(it))
    return res


def iter_costs(team, case, r):
    """{iteration (Run + 1): cumulative LLM cost (USD) when the Advisor evaluation of that iteration ends}."""
    d = SENS/team/case/f'run{r}'
    h = max((hist(p) for p in d.rglob('Experiment_Report.md') if '/box/' not in str(p)), key=len)
    udir = d/'usage' if list((d/'usage').glob('*[0-9].jsonl')) else RUN_LOGS/team/case/f'run{r}'/'usage'
    ev = [json.loads(l) for f in sorted(udir.glob('*[0-9].jsonl')) for l in f.read_text().splitlines() if l.startswith('{')]
    chats = [e for e in ev if e.get('kind') == 'chat']
    evals = [e for e in chats if e.get('role') == 'register_agent_parser'] or [e for e in chats if e.get('role') == 'advisor_agent']
    cost = lambda t: sum((e.get('output_tokens') or 0) * PRICE[e['model']] for e in chats if t is None or e['t_rel_s'] <= t) / 1e6
    out = {it: cost(evals[j]['t_rel_s']) for j, (it, _) in enumerate(h) if j < len(evals)}
    if (team, case, r) in EXTRA:   # trial scored before the crash: whole-run cost
        out[EXTRA[(team, case, r)][0]] = cost(None)
    return out


HE = {(t, r): helm_errors(t, r) for t in TEAM for r in range(1, 6)}
HC = {(t, r): iter_costs(t, 'Helmholtz', r) for t in TEAM for r in range(1, 6)}
SUMM = {}
for (t, r), e in HE.items():
    its = sorted(e)
    first = next(i for i in its if (e[i][1] or 0) >= 90)
    acc = ACCEPTED_RUN.get((t, r), max(its) - 1) + 1
    SUMM[(t, r)] = {'first90': e[first][0], 'accepted': e[acc][0], 'it_first': first, 'it_acc': acc}
    print(f'H {t:6s} r{r} ' + ' '.join(f'{i}:{e[i][0]:.1e}({e[i][1]})' for i in its) + f' | first>=90 it{first} {e[first][0]:.2e} | accepted it{acc} {e[acc][0]:.2e}')
for t in TEAM:
    a = [SUMM[(t, r)]['accepted'] for r in range(1, 6)]; f = [SUMM[(t, r)]['first90'] for r in range(1, 6)]
    print(f'H {t:6s} accepted {min(a):.2e}-{max(a):.2e} | first>=90 {min(f):.2e}-{max(f):.2e}')


def plot_helm_curves(ax, team, title, show_label=False):
    lab, col, mk = TEAM[team]
    for r in range(1, 6):
        e = HE[(team, r)]; c = HC[(team, r)]; its = sorted(e); s = SUMM[(team, r)]
        ax.plot([c[i] for i in its], [e[i][0] for i in its], marker=mk, ms=5, color=col, alpha=0.6, ls=LS[team], lw=1.2)
        ax.plot(c[s['it_first']], s['first90'], marker=mk, ms=9, ls='none', color=col, markerfacecolor='white', mew=1.5, zorder=5)
        ax.plot(c[s['it_acc']], s['accepted'], marker='X', ms=11, ls='none', color=col, mec='white', mew=0.8, zorder=6)
    ax.axhspan(*OPUS_H_RANGE, color=OPUS[1], alpha=0.2, lw=0)
    ax.set_yscale('log')
    ax.set_xlabel('Cumulative cost (USD)', fontsize=LABEL_FONTSIZE)
    if show_label: ax.set_ylabel(r'Relative $L^2$ error', fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--'); ax.tick_params(labelsize=TICK_FONTSIZE)


def plot_helm_bars(ax):
    rng = np.random.default_rng(0); W = 0.2
    for j, (team, state) in enumerate([('open', 'accepted'), ('open', 'first90'), ('closed', 'accepted'), ('closed', 'first90')]):
        x0 = (j - 1.5) * W
        lab, col, mk = TEAM[team]
        vs = [SUMM[(team, r)][state] for r in range(1, 6)]
        ax.bar(x0, np.exp(np.mean(np.log(vs))), width=W * 0.9, color=col, alpha=0.25 if state == 'accepted' else 0.08,
               edgecolor=col, linewidth=1.2, hatch=None if state == 'accepted' else '///')
        for v in vs:
            ax.plot(x0 + rng.uniform(-W * 0.25, W * 0.25), v, marker=mk, ms=6, ls='none', color=col,
                    markerfacecolor=col if state == 'accepted' else 'white', markeredgewidth=1.3, alpha=0.9)
    ax.axhspan(*OPUS_H_RANGE, color=OPUS[1], alpha=0.2, lw=0)
    ax.set_yscale('log'); ax.set_xticks([-W, W]); ax.set_xticklabels(['Open', 'Closed'], fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r'Relative $L^2$ error', fontsize=LABEL_FONTSIZE)
    ax.set_title('SciML (H)', fontsize=TITLE_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y'); ax.tick_params(labelsize=TICK_FONTSIZE)


def score_curve(team, case, r):
    """[(iteration, score, cumulative cost)] and the accepted iteration of one sensitivity run."""
    d = SENS/team/case/f'run{r}'
    h = max((hist(p) for p in d.rglob('Experiment_Report.md') if '/box/' not in str(p)), key=len)
    if (team, case, r) in EXTRA: h.append(EXTRA[(team, case, r)])
    if (team, case, r) in GATE_CUT: h = [x for x in h if x[0] <= GATE_CUT[(team, case, r)]]
    c = iter_costs(team, case, r)
    pts = [(i, s, c[i]) for i, s in h if i in c]
    logs = [p for p in (d/'stdout.txt', RUN_LOGS/team/case/f'run{r}'/'stdout.txt') if p.exists()]
    gate = re.findall(r'STOP \(score gate:.*?best \d+ at iteration (\d+)\)', logs[0].read_text(errors='ignore')) if logs else []
    if (team, case, r) in GATE_CUT:   # score gate applied retroactively: best trial up to the cut (ties: latest)
        acc = max(pts, key=lambda x: (x[1], x[0]))[0]
    elif gate:                        # score gate live in the run: it returns the best trial
        acc = int(gate[-1]) + 1
    elif case == 'Helmholtz' and (team, r) in ACCEPTED_RUN:
        acc = ACCEPTED_RUN[(team, r)] + 1
    else:                             # Advisor approval: the last trial
        acc = pts[-1][0]
    return pts, acc


def plot_score_cost(ax, case, title, show_label=False):
    for team in ('closed', 'open'):
        lab, col, mk = TEAM[team]
        for r in range(1, 6):
            pts, acc = score_curve(team, case, r)
            ax.plot([x[2] for x in pts], [x[1] for x in pts], marker=mk, ms=5, color=col, alpha=0.6, ls=LS[team], lw=1.2)
            f = next(x for x in pts if x[1] >= 90); a = next(x for x in pts if x[0] == acc)
            ax.plot(f[2], f[1], marker=mk, ms=9, ls='none', color=col, markerfacecolor='white', mew=1.5, zorder=5)
            ax.plot(a[2], a[1], marker='X', ms=11, ls='none', color=col, mec='white', mew=0.8, zorder=6)
            print(f'score {case:9s} {team:6s} r{r} ' + ' '.join(f'{i}:{s}@{cc:.2f}' for i, s, cc in pts) + f' | first90 {f[0]}@{f[2]:.2f} acc {a[0]}@{a[2]:.2f}')
    oc = [d['cost'] for d in OP[case]]   # Opus 5.5 cost per session (row 2)
    ax.axvspan(min(oc), max(oc), color=OPUS[1], alpha=0.2, lw=0)
    print(f'opus cost {case}: {min(oc):.2f}-{max(oc):.2f}')
    ax.axhline(90, color='gray', ls=':', lw=1.2)
    ax.set_xlabel('Cumulative cost (USD)', fontsize=LABEL_FONTSIZE)
    if show_label: ax.set_ylabel('Score', fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--'); ax.tick_params(labelsize=TICK_FONTSIZE)


# ---------------- Row 3: error against cumulative cost (IB2D and H; RT has no exact solution) ----------------
IB_RL2 = json.loads((Path(__file__).resolve().parent/'ib2d_rl2.json').read_text())   # from ib2d_rl2.py (t = 2, exact entropy solution)
# Opus 5.5 relative L2 error per run (run 1..5)
OPUS_RUN_ERR = {'IB': {int(r): v for r, v in IB_RL2['opus'].items()},
                # Helmholtz: final accuracy stated in each run's REPORT.md (bare_claude/Helmholtz/run*/box/final),
                # range 2.7e-5 to 1.1e-4 as in Supplementary Table B5
                'Helmholtz': {1: 5.46e-5, 2: 2.73e-5, 3: 3.24e-5, 4: 3.0e-5, 5: 1.08e-4}}
OPUS_RUN_COST = {c: {r: v['cost'] for r in range(1, 6) for v in [opus(c, r)] if v} for c in ('IB', 'Helmholtz')}


def err_curve(team, case, r):
    """[(iteration, error, cumulative cost)], first >= 90 iteration and accepted iteration of one run."""
    pts, acc = score_curve(team, case, r)
    first = next(i for i, s, _ in pts if s >= 90)
    if case == 'IB':
        return [(i, IB_RL2['athena'][f'{team}/{r}/{i}'], c) for i, s, c in pts], first, acc
    e = HE[(team, r)]; c = HC[(team, r)]; its = sorted(e); sm = SUMM[(team, r)]
    return [(i, e[i][0], c[i]) for i in its], sm['it_first'], sm['it_acc']


def plot_err_cost(ax, case, title, ylabel):
    for r, e in OPUS_RUN_ERR[case].items():   # one marker per Opus 5.5 session
        ax.plot(OPUS_RUN_COST[case][r], e, marker=OPUS[2], ms=8, color=OPUS[1], ls='none', alpha=0.9, zorder=5)
    for team in ('closed', 'open'):
        lab, col, mk = TEAM[team]
        for r in range(1, 6):
            pts, fi, ai = err_curve(team, case, r)
            ax.plot([x[2] for x in pts], [x[1] for x in pts], marker=mk, ms=5, color=col, alpha=0.6, ls=LS[team], lw=1.2)
            f = next(x for x in pts if x[0] == fi); a = next(x for x in pts if x[0] == ai)
            ax.plot(f[2], f[1], marker=mk, ms=9, ls='none', color=col, markerfacecolor='white', mew=1.5, zorder=5)
            ax.plot(a[2], a[1], marker='X', ms=11, ls='none', color=col, mec='white', mew=0.8, zorder=6)
    ax.set_yscale('log')
    ax.set_xlabel('Cumulative cost (USD)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--'); ax.tick_params(labelsize=TICK_FONTSIZE)


def plot_err_bars(ax):
    rng = np.random.default_rng(0); W = 0.16
    order = [('open', 'accepted'), ('open', 'first90'), ('closed', 'accepted'), ('closed', 'first90')]
    for k, case in enumerate(['IB', 'Helmholtz']):
        for j, (team, state) in enumerate(order):
            x0 = k + (j - 2) * W
            lab, col, mk = TEAM[team]
            vs = []
            for r in range(1, 6):
                pts, fi, ai = err_curve(team, case, r)
                vs.append(next(x[1] for x in pts if x[0] == (ai if state == 'accepted' else fi)))
            ax.bar(x0, np.exp(np.mean(np.log(vs))), width=W * 0.9, color=col, alpha=0.25 if state == 'accepted' else 0.08,
                   edgecolor=col, linewidth=1.2, hatch=None if state == 'accepted' else '///')
            for v in vs:
                ax.plot(x0 + rng.uniform(-W * 0.25, W * 0.25), v, marker=mk, ms=6, ls='none', color=col,
                        markerfacecolor=col if state == 'accepted' else 'white', markeredgewidth=1.3, alpha=0.9)
        vs = list(OPUS_RUN_ERR[case].values()); x0 = k + 2 * W
        ax.bar(x0, np.exp(np.mean(np.log(vs))), width=W * 0.9, color=OPUS[1], alpha=0.25, edgecolor=OPUS[1], linewidth=1.2)
        for v in vs:
            ax.plot(x0 + rng.uniform(-W * 0.25, W * 0.25), v, marker=OPUS[2], ms=7, ls='none', color=OPUS[1], alpha=0.9)
    ax.set_yscale('log'); ax.set_xticks([0, 1]); ax.set_xticklabels(['IB2D', 'H'], fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r'Relative $L^2$ error', fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y'); ax.tick_params(labelsize=TICK_FONTSIZE)


# ---------------- Row 3: PIML benchmarks of Figure 5, relative L2 error per iteration ----------------
# Error and score of every scored trial, read from the Full Trial History of the final Experiment_Report.md
# of the run reported in Figure 5 / Table 1 (closed team, Supplementary Table A1). Iteration = Run + 1, as in row 2.
PINN = ROOT/'Results'/'PINN_codes'
PIML_RUNS = {'AC': 'AllenCahn_Periodic_X2CosPiX/2025_12_05_1208', 'VB': 'Burgers_Periodic_SinPi/2026_06_17_1749',
             'H': 'Helmholtz_Periodic_SinPiA1SinPiA2/2025_12_05_1421', 'P': 'Poisson_Dirichlet_Sin4pixSin4piy/2026_01_10_2319',
             'KdV': 'KdV_Mixed_DerivedIC/2026_01_10_2026', 'IB': 'Burgers_Periodic_SinPiX/2025_12_04_1621'}
# Figure 5 (fig_pinn_bars.py): Opus 5.5 and Human SOTA relative L2 errors
PIML_REF = {'opus':  {'AC': 2.24e-4, 'VB': 3.50e-5, 'H': 2.80e-5, 'P': 1.77e-4, 'KdV': 1.07e-4, 'IB': 2.07e-3}}
PAIRS = [('AC', 'VB'), ('H', 'P'), ('KdV', 'IB')]
CASE_LS = ['-', '--']
CASE_COL = ['red', 'black']   # first / second case of each panel


def piml_history(case):
    """[(iteration, relative L2 error, score)] of the reported run."""
    txt = (PINN/PIML_RUNS[case]/'Experiment_Report.md').read_text(errors='ignore')
    m = re.search(r'## Full Trial History\s*\n(.*?)(?=\n##|\n###|$)', txt, re.DOTALL); out = []
    for line in m.group(1).split('\n'):
        c = [x.strip().replace('**', '') for x in line.split('|')[1:-1]]
        if c and c[0].isdigit():
            e = c[2]; e = float(e.rstrip('%')) / 100 if e.endswith('%') else float(e)
            out.append((int(c[0]) + 1, e, int(c[-1])))
    return out


PH = {c: piml_history(c) for c in PIML_RUNS}
for c, h in PH.items():
    f = next(x for x in h if x[2] >= 90)
    print(f'{c:4s} ' + ' '.join(f'{i}:{e:.1e}({s})' for i, e, s in h) + f' | first>=90 it{f[0]} {f[1]:.2e} | accepted it{h[-1][0]} {h[-1][1]:.2e}'
          + f' | Opus {PIML_REF["opus"][c]:.1e}')


def plot_piml(ax, pair, show_label=False, loc='upper right'):
    mk = TEAM['closed'][2]
    for case, ls, col in zip(pair, CASE_LS, CASE_COL):
        h = PH[case]; its = [x[0] for x in h]; es = [x[1] for x in h]
        f = next(x for x in h if x[2] >= 90)
        ax.plot(its, es, marker=mk, ms=5, color=col, alpha=0.6, ls=ls, lw=1.2)
        ax.plot(f[0], f[1], marker=mk, ms=9, ls='none', color=col, markerfacecolor='white', mew=1.5, zorder=5)
        ax.plot(h[-1][0], h[-1][1], marker='X', ms=11, ls='none', color=col, mec='white', mew=0.8, zorder=6)
        ax.axhline(PIML_REF['opus'][case], color=OPUS[1], ls=ls, lw=1.5, alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=LABEL_FONTSIZE)
    if show_label: ax.set_ylabel(r'Relative $L^2$ error', fontsize=LABEL_FONTSIZE)
    ax.set_title(f'SciML ({pair[0]} and {pair[1]})', fontsize=TITLE_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--'); ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    hs = [Line2D([], [], color=col, ls=ls, lw=1.5, marker=mk, ms=6, label=c) for c, ls, col in zip(pair, CASE_LS, CASE_COL)]
    hs += [Line2D([], [], color=OPUS[1], ls=ls, lw=1.5, label=f'Opus 5.5 ({c})') for c, ls in zip(pair, CASE_LS)]
    ax.legend(handles=hs, ncols=1 if loc == 'lower right' else 2, fontsize=LEGEND_FONTSIZE, framealpha=0.95, loc=loc, handlelength=2.0,
              borderpad=0.3, columnspacing=0.8, handletextpad=0.4)


fig, axes = plt.subplots(3, 3, figsize=(12, 10.44))
plot_bars(axes[1, 0], 'cost', 'Cost per run (USD)')
plot_bars(axes[1, 1], 'out_tokens', 'Output tokens per run (k)')
plot_bars(axes[1, 2], 'time', 'Wall-clock time (min)')
leg = dict(fontsize=LEGEND_FONTSIZE, framealpha=0.95, loc='upper center', columnspacing=0.4, handletextpad=0.1,
           handlelength=1.0, borderpad=0.3, borderaxespad=0.3)
h = [Line2D([], [], color=TEAM[t][1], marker=TEAM[t][2], ls='none', ms=6, label=TEAM[t][0]) for t in TEAM]
h.append(Line2D([], [], color=OPUS[1], marker=OPUS[2], ls='none', ms=7, label=OPUS[0]))
axes[1, 0].legend(handles=h, ncols=3, **leg)
h2 = [Patch(facecolor='gray', alpha=0.25, edgecolor='gray', label='Accepted'),
      Patch(facecolor='white', edgecolor='gray', hatch='///', label=r'First score $\geq 90$')]
axes[1, 1].legend(handles=h2, ncols=2, **{**leg, 'handlelength': 1.6, 'handletextpad': 0.4})
plot_score_cost(axes[2, 0], 'IB', 'SciC (IB2D)', show_label=True)
plot_score_cost(axes[2, 1], 'RT', 'SciC (RT)')
plot_score_cost(axes[2, 2], 'Helmholtz', 'SciML (H)')
h3 = [Line2D([], [], color=TEAM[t][1], marker=TEAM[t][2], ls=LS[t], ms=6, lw=1.2, label=TEAM[t][0]) for t in TEAM]
h3.append(Patch(facecolor=OPUS[1], alpha=0.2, label='Opus 5.5'))
axes[2, 0].legend(handles=h3, ncols=1, **{**leg, 'loc': 'lower right', 'handlelength': 2.0, 'handletextpad': 0.4})
for j, (pair, loc) in enumerate(zip(PAIRS, ['upper right', 'upper right', 'lower right'])):
    plot_piml(axes[0, j], pair, show_label=(j == 0), loc=loc)
h4 = [Line2D([], [], color='gray', marker='s', ls='none', ms=8, markerfacecolor='white', mew=1.5, label=r'First score $\geq 90$'),
      Line2D([], [], color='gray', marker='X', ls='none', ms=10, mec='white', label='Accepted')]
axes[0, 1].add_artist(axes[0, 1].get_legend())
axes[0, 1].legend(handles=h4, fontsize=LEGEND_FONTSIZE, framealpha=0.95, loc='center right', handlelength=1.2, borderpad=0.3, handletextpad=0.4)
plt.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else 'athena_first90_SI.png'
plt.savefig(out, dpi=300, bbox_inches='tight'); print('saved', out)
