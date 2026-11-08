"""
usage_tracker.py — per-run token and wall-clock accounting for ATHENA.

Attaches to an already-built agent dict (from HENA.build_agents) without
changing any agent's behaviour:

  * every chat-model call  -> one record: role, model, tokens in/out (with the
    provider's cache / reasoning breakdown when reported), latency, timestamp;
  * every embedding call   -> one record: role, model, number of texts, chars
    (embedding APIs do not return token counts);
  * every code execution   -> one record: wall-clock seconds of run_code().

Records are appended to a JSONL file as they happen (survives a crash), and a
summary JSON (totals per role, per model, per kind) is written at the end.

Usage:
    from crunch_llm.code.usage_tracker import UsageLog, instrument_agents, instrument_run_code
    log = UsageLog("path/to/usage_<run>.jsonl", meta={...})
    instrument_agents(agents, log)
    instrument_run_code(HENA_module, log)
    ... run ...
    log.write_summary()
"""

import json
import os
import threading
import time
from collections import defaultdict
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _model_name(llm):
    for attr in ("model", "model_name"):
        val = getattr(llm, attr, None)
        if isinstance(val, str) and val:
            return val
    return type(llm).__name__


class UsageLog:
    """Thread-safe append-only usage log for one run."""

    def __init__(self, jsonl_path, meta=None):
        self.jsonl_path = os.path.abspath(jsonl_path)
        os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
        self.meta = dict(meta or {})
        self.meta.setdefault("started", _now())
        self._t0 = time.time()
        self._lock = threading.Lock()
        self.records = []
        self.add({"kind": "run_start", **self.meta})

    def add(self, rec):
        rec = {"time": _now(), "t_rel_s": round(time.time() - self._t0, 3), **rec}
        with self._lock:
            self.records.append(rec)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------------ summary
    def summary(self):
        def blank():
            return defaultdict(float)
        by_role, by_model, by_kind = defaultdict(blank), defaultdict(blank), defaultdict(blank)
        for r in self.records:
            kind = r.get("kind")
            if kind not in ("chat", "chat_error", "embed", "exec"):
                continue
            keys = [(by_kind, kind)]
            if kind in ("chat", "chat_error", "embed"):
                keys += [(by_role, r.get("role", "?")), (by_model, r.get("model", "?"))]
            for table, k in keys:
                t = table[k]
                t["calls"] += 1
                t["seconds"] += r.get("seconds", 0) or 0
                for f in ("input_tokens", "output_tokens", "total_tokens",
                          "cache_read_tokens", "reasoning_tokens", "chars", "n_texts"):
                    t[f] += r.get(f, 0) or 0
                if kind == "chat_error":
                    t["errors"] += 1
        tidy = lambda d: {k: {f: (round(v, 3) if isinstance(v, float) and not v.is_integer() else int(v))
                              for f, v in sorted(t.items())} for k, t in sorted(d.items())}
        return {
            "meta": self.meta,
            "finished": _now(),
            "wall_clock_s": round(time.time() - self._t0, 1),
            "by_kind": tidy(by_kind),
            "by_role": tidy(by_role),
            "by_model": tidy(by_model),
        }

    def write_summary(self, path=None, extra=None):
        path = path or self.jsonl_path.replace(".jsonl", "_summary.json")
        s = self.summary()
        if extra:
            s["result"] = extra
        with open(path, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, default=str)
        return path


# ---------------------------------------------------------------- chat models
class RoleUsageCallback(BaseCallbackHandler):
    """LangChain callback that records usage of one agent role."""

    def __init__(self, log, role, model, provider):
        self.log, self.role, self.model, self.provider = log, role, model, provider
        self._start = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start[run_id] = time.time()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start[run_id] = time.time()

    def on_llm_end(self, response, *, run_id, **kwargs):
        t0 = self._start.pop(run_id, None)
        rec = {"kind": "chat", "role": self.role, "model": self.model, "provider": self.provider,
               "seconds": round(time.time() - t0, 3) if t0 else None}
        usage = None
        try:
            msg = response.generations[0][0].message
            usage = getattr(msg, "usage_metadata", None)
            served = (getattr(msg, "response_metadata", None) or {}).get("model") \
                or (getattr(msg, "response_metadata", None) or {}).get("model_name")
            if served:
                rec["served_model"] = served
        except Exception:
            pass
        if not usage:
            lo = response.llm_output or {}
            usage = lo.get("usage_metadata") or lo.get("token_usage") or lo.get("usage")
            if usage and "prompt_tokens" in usage:
                usage = {"input_tokens": usage.get("prompt_tokens"),
                         "output_tokens": usage.get("completion_tokens"),
                         "total_tokens": usage.get("total_tokens")}
        if usage:
            rec["input_tokens"] = usage.get("input_tokens", 0) or 0
            rec["output_tokens"] = usage.get("output_tokens", 0) or 0
            rec["total_tokens"] = usage.get("total_tokens") or rec["input_tokens"] + rec["output_tokens"]
            itd = usage.get("input_token_details") or {}
            otd = usage.get("output_token_details") or {}
            if itd.get("cache_read"):
                rec["cache_read_tokens"] = itd["cache_read"]
            if otd.get("reasoning"):
                rec["reasoning_tokens"] = otd["reasoning"]
        else:
            rec["usage_missing"] = True
        self.log.add(rec)

    def on_llm_error(self, error, *, run_id, **kwargs):
        t0 = self._start.pop(run_id, None)
        self.log.add({"kind": "chat_error", "role": self.role, "model": self.model,
                      "provider": self.provider,
                      "seconds": round(time.time() - t0, 3) if t0 else None,
                      "error": f"{type(error).__name__}: {str(error)[:300]}"})


# ----------------------------------------------------------------- embeddings
class CountingEmbeddings(Embeddings):
    """Transparent wrapper that logs every embedding call."""

    def __init__(self, inner, log, role):
        self.inner, self.log, self.role = inner, log, role
        self.model = _model_name(inner)

    def _rec(self, texts, t0):
        self.log.add({"kind": "embed", "role": self.role, "model": self.model,
                      "n_texts": len(texts), "chars": sum(len(t) for t in texts),
                      "seconds": round(time.time() - t0, 3)})

    def embed_query(self, text):
        t0 = time.time()
        out = self.inner.embed_query(text)
        self._rec([text], t0)
        return out

    def embed_documents(self, texts):
        t0 = time.time()
        out = self.inner.embed_documents(texts)
        self._rec(list(texts), t0)
        return out


# --------------------------------------------------------------- attach hooks
def instrument_agents(agents, log, quiet=False):
    """Attach usage callbacks to every agent's chat model and vector store.

    Returns {role: model} for the roles that were instrumented. Raises if a
    role exposes no chat model, so no call can go uncounted silently.
    """
    team = {}
    for role, agent in agents.items():
        llm = getattr(agent, "llm", None)
        if llm is None:
            raise RuntimeError(f"usage_tracker: agent '{role}' ({type(agent).__name__}) "
                               "exposes no .llm; its calls would not be counted.")
        model = _model_name(llm)
        cb = RoleUsageCallback(log, role, model, type(llm).__name__)
        existing = llm.callbacks if isinstance(llm.callbacks, list) else []
        llm.callbacks = [c for c in existing if not isinstance(c, RoleUsageCallback)] + [cb]
        team[role] = model
        vs = getattr(agent, "vector_store", None)
        emb = getattr(vs, "_embedding_function", None) if vs is not None else None
        if emb is not None and not isinstance(emb, CountingEmbeddings):
            vs._embedding_function = CountingEmbeddings(emb, log, role)
            team[role + ".embeddings"] = _model_name(emb)
    log.add({"kind": "team", "team": team})
    if not quiet:
        print("[usage] tracking roles:", ", ".join(f"{r}={m}" for r, m in team.items()))
    return team


def instrument_run_code(module, log):
    """Wrap module.run_code (the execution bridge) to log its wall-clock time."""
    orig = getattr(module, "run_code")
    if getattr(orig, "_usage_wrapped", False):
        return

    def run_code(project_dir, final_filename, final_filepath):
        t0 = time.time()
        ok = None
        try:
            out = orig(project_dir, final_filename, final_filepath)
            ok = bool(out[0]) if isinstance(out, tuple) else None
            return out
        finally:
            log.add({"kind": "exec", "file": final_filename, "success": ok,
                     "seconds": round(time.time() - t0, 3)})

    run_code._usage_wrapped = True
    setattr(module, "run_code", run_code)
