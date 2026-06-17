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
"""
import argparse
import json
import re
from pathlib import Path

from crunch_llm.code.HENA import run_hena, build_agents, get_config

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request_json", help="path to ATHENA_Request_*.json")
    ap.add_argument("--runs", type=int, default=1,
                    help="number of repeated runs (regret analysis)")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="override max_iterations_all (quick test)")
    ap.add_argument("--stage", type=int, default=None,
                    help="run only this pipeline stage number")
    args = ap.parse_args()

    req = json.load(open(args.request_json))
    problem = req["problem_name"]
    pipeline = req["pipeline"]
    stage_reqs = split_stage_requests(req["user_request"], len(pipeline))

    overrides = {}
    if args.max_iters is not None:
        overrides["max_iterations_all"] = args.max_iters

    for run_idx in range(1, args.runs + 1):
        run_tag = problem if args.runs == 1 else f"{problem}_run{run_idx}"
        print(f"\n{'='*70}\nATHENA :: {run_tag}\n{'='*70}")
        for stage, sreq in zip(pipeline, stage_reqs):
            if args.stage is not None and stage["stage"] != args.stage:
                continue
            path = stage["path"]
            print(f"\n--- Stage {stage['stage']} (family {path}): {stage['task'][:60]} ---")
            cfg = get_config(path, overrides) if overrides else get_config(path)
            agents = build_agents(path, cfg)
            result = run_hena(
                path, sreq,
                agents=agents, cfg=cfg,
                run_name=f"{run_tag}_stage{stage['stage']}",
            )
            print(f"approved={result['is_approved']}  "
                  f"project_dir={result['project_dir']}  "
                  f"log={result.get('log_path')}")


if __name__ == "__main__":
    main()
