#!/usr/bin/env python3
"""Drive an ATHENA request JSON through the HENA loop.

Reads a Conceptualization-team request JSON and dispatches each pipeline stage
to `run_hena(stage.path, stage_request)`. Each run auto-mirrors its full
transcript to crunch_llm/code/HENA_logs/<FAMILY_LABEL>/.

Usage:
    python run_athena.py Conceptualization_team/ATHENA_Request_Rayleigh_Taylor_2D_Euler.json
    python run_athena.py <request.json> --runs 5            # repeat (regret analysis)
    python run_athena.py <request.json> --max-iters 5       # cap iterations for a quick test
    python run_athena.py <request.json> --stage 2           # run only one pipeline stage
    python run_athena.py <request.json> --team open         # open-weight team (Ollama Cloud)
    python run_athena.py <request.json> --resume <dir>      # continue from the last checkpoint in <dir>

Checkpoints: every scored iteration writes Experiment_Report.md (strategy, advisor outcome,
full trial history) next to its code. --resume reloads the newest one under <dir> and the loop
continues at the next iteration. When the Debugger stops a run as STOP_UNRECOVERABLE
(environment, package or input error), UNRECOVERABLE_ERROR.md in the run folder says what
failed and gives the exact command to resume once it is fixed.

Every run also writes a token/time usage log (one JSONL record per LLM call,
embedding call and code execution, plus a summary JSON) to --usage-dir.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path

import crunch_llm.code.HENA as HENA
from crunch_llm.code.HENA import run_hena, build_agents, get_config
from crunch_llm.code.usage_tracker import UsageLog, instrument_agents, instrument_run_code

REPO = Path(__file__).resolve().parent


def split_stage_requests(user_request, n_stages):
    """Return one request string per stage. If the request uses '### STAGE N'
    markers and the count matches, split on them; otherwise give every stage
    the full request."""
    parts = re.split(r'(?im)^\s*#+\s*STAGE\s*\d+.*$', user_request)
    sections = [p.strip() for p in parts if p.strip()]
    if len(sections) == n_stages:
        return sections
    return [user_request] * n_stages


DB_KEYS = ("db_path", "db_functions", "db_functions_ollama")


def isolate_dbs(cfg, dest):
    """Point every vector-DB path in cfg at a private copy under dest."""
    cfg = dict(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    for key in DB_KEYS:
        src = cfg.get(key)
        if not src:
            continue
        src = Path(src).resolve()
        dst = dest / src.name
        if not dst.exists():
            shutil.copytree(src, dst)
        cfg[key] = str(dst)
        print(f"[db] {key}: {src} -> {dst}")
    return cfg


def find_checkpoint(path):
    """The folder holding the newest Experiment_Report.md at or under `path` (or None)."""
    path = Path(path).resolve()
    if (path / "Experiment_Report.md").is_file():
        return path
    reports = sorted(path.rglob("Experiment_Report.md"), key=lambda p: p.stat().st_mtime)
    return reports[-1].parent if reports else None


# Options that athena_box_run.sh sets itself; dropped from the resume command inside the box.
BOX_OPTS = ("--team", "--results-dir", "--log-base", "--usage-dir", "--db-copy-dir")


def resume_command(argv, checkpoint, stage, multi_stage):
    """This run's own command line with --resume (and --stage) pointing at `checkpoint`.

    Inside the box (ATHENA_RESUME_PREFIX, set by athena_box_run.sh) the command is the
    athena_box_run.sh launch plus this run's other options."""
    box_prefix = os.environ.get("ATHENA_RESUME_PREFIX", "")
    drop = ("--resume", "--stage") + (BOX_OPTS if box_prefix else ())
    out, skip = [], False
    for a in argv[2:] if box_prefix else argv:      # box: argv[0:2] = run_athena.py <request>
        if skip:
            skip = False
            continue
        if a in drop:
            skip = True
            continue
        if a.startswith(tuple(o + "=" for o in drop)):
            continue
        out.append(a)
    if checkpoint:
        out += ["--resume", str(checkpoint)]
    if multi_stage:
        out += ["--stage", str(stage)]
    if box_prefix:
        return " ".join([box_prefix, *(shlex.quote(a) for a in out)])
    return " ".join(shlex.quote(a) for a in [Path(sys.executable).name, *out])


def write_help_request(result, argv, stage, multi_stage):
    """Stopped as STOP_UNRECOVERABLE: tell the human what to fix and how to continue."""
    ckpt = result.get("last_checkpoint_dir") or ""
    cmd = resume_command(argv, ckpt, stage, multi_stage)
    where = (f"the last checkpoint, `{ckpt}` (iteration {result.get('refinement_iterations')} "
             "restarts from there)" if ckpt else
             "the beginning: no iteration was scored before the stop, so there is no checkpoint yet")
    text = ("\n## What to do\n\n"
            "The agents were told not to change the software environment, so the run stopped and "
            "asks for help. Fix the problem described above (environment, package or input), then "
            f"resume from {where}:\n\n```bash\n{cmd}\n```\n")
    print("\n🆘 Help needed. After fixing the problem above, resume with:\n   " + cmd + "\n")
    pdir = result.get("project_dir") or ""
    if pdir and Path(pdir).is_dir():
        with open(Path(pdir) / "UNRECOVERABLE_ERROR.md", "a") as f:
            f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request_json", help="path to ATHENA_Request_*.json")
    ap.add_argument("--runs", type=int, default=1,
                    help="number of repeated runs (regret analysis)")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="override max_iterations_all (quick test)")
    ap.add_argument("--stage", type=int, default=None,
                    help="run only this pipeline stage number")
    ap.add_argument("--team", default="original",
                    help="agent team: 'original' or a HENA.TEAM_BUILDERS key (e.g. 'open')")
    ap.add_argument("--results-dir", default=None,
                    help="override base_results_dir (keeps sensitivity runs separate)")
    ap.add_argument("--log-base", default=None,
                    help="write transcripts as <log-base>/<FAMILY>/hena_*_<timestamp>.txt "
                         "(timestamped, never overwrites) instead of Examples_<family>/")
    ap.add_argument("--usage-dir", default=str(REPO / "run_logs" / "usage"),
                    help="directory for the per-run token/time usage logs")
    ap.add_argument("--db-copy-dir", default=None,
                    help="copy the family's vector DBs into this directory and use the "
                         "copies (isolates concurrent runs; the originals are never opened)")
    ap.add_argument("--resume", default=None,
                    help="continue from the newest Experiment_Report.md at or under this folder "
                         "(needs --stage when the request has several stages)")
    args = ap.parse_args()

    req = json.load(open(args.request_json))
    # {REPO} in a request stands for the repository root (input files under Documents/codes/Data)
    req["user_request"] = req["user_request"].replace("{REPO}", str(REPO))
    problem = req["problem_name"]
    pipeline = req["pipeline"]
    stage_reqs = split_stage_requests(req["user_request"], len(pipeline))
    multi_stage = len(pipeline) > 1
    checkpoint = ""
    if args.resume:
        if multi_stage and args.stage is None:
            ap.error("--resume needs --stage: this request has several stages")
        if args.runs != 1:
            ap.error("--resume continues one run; drop --runs")
        found = find_checkpoint(args.resume)
        if found is None:
            ap.error(f"no Experiment_Report.md at or under {args.resume}")
        checkpoint = str(found)
        print(f"[resume] checkpoint: {checkpoint}")

    overrides = {}
    if args.max_iters is not None:
        overrides["max_iterations_all"] = args.max_iters
    if args.results_dir is not None:
        overrides["base_results_dir"] = str(Path(args.results_dir).resolve()) + "/"

    for run_idx in range(1, args.runs + 1):
        run_tag = problem if args.runs == 1 else f"{problem}_run{run_idx}"
        print(f"\n{'='*70}\nATHENA :: {run_tag}\n{'='*70}")
        for stage, sreq in zip(pipeline, stage_reqs):
            if args.stage is not None and stage["stage"] != args.stage:
                continue
            path = stage["path"]
            print(f"\n--- Stage {stage['stage']} (family {path}): {stage['task'][:60]} ---")
            cfg = get_config(path, overrides) if overrides else get_config(path)
            if args.db_copy_dir:
                cfg = isolate_dbs(cfg, Path(args.db_copy_dir))
            agents = build_agents(path, cfg, team=args.team)

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            usage = UsageLog(
                Path(args.usage_dir) / f"usage_{args.team}_{run_tag}_stage{stage['stage']}_{stamp}.jsonl",
                meta=dict(request=str(Path(args.request_json).resolve()), problem=problem,
                          run=run_tag, stage=stage["stage"], family=path, team=args.team,
                          max_iterations_all=cfg.get("max_iterations_all"),
                          resumed_from=checkpoint or None,
                          base_results_dir=cfg.get("base_results_dir")))
            instrument_agents(agents, usage)
            instrument_run_code(HENA, usage)

            result, err = None, None
            try:
                result = run_hena(
                    path, sreq,
                    agents=agents, cfg=cfg,
                    run_name=f"{run_tag}_stage{stage['stage']}",
                    checkpoint_project_dir=checkpoint,
                    log_base=args.log_base,
                )
            except BaseException as e:
                err = f"{type(e).__name__}: {e}"
                raise
            finally:
                keep = ("is_approved", "project_dir", "filename", "refinement_iterations", "log_path",
                        "stop_reason", "gate_stop", "last_checkpoint_dir")
                summ = usage.write_summary(extra=dict(
                    {k: result.get(k) for k in keep} if isinstance(result, dict) else {},
                    error=err))
                print(f"[usage] summary: {summ}")
            print(f"approved={result['is_approved']}  "
                  f"project_dir={result['project_dir']}  "
                  f"log={result.get('log_path')}")
            if result.get("stop_reason"):
                write_help_request(result, sys.argv, stage["stage"], multi_stage)
                sys.exit(3)       # stopped for a human: later stages need this one


if __name__ == "__main__":
    main()
