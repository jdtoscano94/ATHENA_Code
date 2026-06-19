"""
HENA.py — family-aware HENA loops + agent factories for ATHENA.

The three example-notebook families map one-to-one onto the Gatekeeper path
codes emitted by the Conceptualization Team:

    A = SciC      (numerical methods)        -> teams_SciC          (Examples_NM, Examples_analytical_exact)
    B = PINN      (physics-informed ML)      -> teams_SciML_PIML    (Examples_PINNs)
    C = Operator  (DeepONet / operators)     -> teams_SciML_Operators (Examples_DeepONet)

Each family differs in THREE ways, so this is a dispatcher over three
*preserved* loops, not a single merged loop:
  1. vector-DB / collection config,
  2. the set of agents that get built,
  3. the HENA loop body itself (A and B share a shape but differ in retrieval
     + librarian steps and call signatures; C fuses the front half into
     `code_creation_team_ON` and uses a different advisor).

The same team-function NAMES (`strategy_team`, `implementation_team`,
`debugging_team_PINN`, `advisor_team`, ...) are defined in MULTIPLE teams_*
modules with DIFFERENT signatures, so the modules are imported as namespaces
(TA / TB / TC) and each loop calls its own family's version. Do NOT switch
these to `from teams_* import *` — that would clobber the shared names.

Usage (from a notebook or run_athena.py):

    from crunch_llm.code.HENA import run_hena
    result = run_hena("B", user_question)            # builds agents + runs the B loop
    # or reuse agents across calls:
    from crunch_llm.code.HENA import build_agents, get_config
    cfg = get_config("A"); agents = build_agents("A", cfg)
    result = run_hena("A", user_question, agents=agents, cfg=cfg)

The loop bodies are lifted verbatim from the example notebooks so behavior is
identical; only variable plumbing (checkpoint load, agent unpacking, config)
is factored out.
"""

import os
import sys
from datetime import datetime

# Agent classes + shared utilities (these names do NOT collide across modules).
from crunch_llm.code.agents import *
from crunch_llm.code.utils import *

# Family-specific teams modules imported as namespaces (signatures differ!).
import crunch_llm.code.teams_SciC as TA
import crunch_llm.code.teams_SciML_PIML as TB
import crunch_llm.code.teams_SciML_Operators as TC


# ==========================================================================
# Configuration (vector-DB paths are machine-specific — see CLAUDE.md).
# Override any field by passing a partial dict as `cfg` to run_hena/build_agents.
# ==========================================================================
# Repo-relative so the artifact is self-contained (HENA.py lives at
# <repo>/crunch_llm/code/HENA.py). Override per-call via config_overrides if needed.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_BASE = os.path.join(_REPO_ROOT, "Documents")
RESULTS_BASE = os.path.join(_REPO_ROOT, "Results")

DB_CONFIG = {
    # ---- A : SciC / numerical methods ----
    "A": dict(
        db_path=f"{DB_BASE}/database_NM",
        db_functions=f"{DB_BASE}/database_Modules_ollama",
        db_functions_ollama=f"{DB_BASE}/database_Modules_ollama",
        modules_name="Layers",
        codes_database="Numerical_Methods_opt",
        base_results_dir=f"{RESULTS_BASE}/NM_codes/",
        max_iterations_all=50,
    ),
    # ---- B : PINN ----
    "B": dict(
        db_path=f"{DB_BASE}/database_PIML",
        db_functions_ollama=f"{DB_BASE}/database_Modules_ollama",
        modules_name="Layers",
        codes_database="PINNs_Codes",
        base_results_dir=f"{RESULTS_BASE}/PINN_codes/",
        max_iterations_all=25,
        # B preloads its auxiliary modules from a checkpoint dir (skips the
        # retrieval_team_direct call when present). Set to '' to disable.
        functions_checkpoint_dir=f"{DB_BASE}/codes/SciML/Modules/",
    ),
    # ---- C : Operator / DeepONet ----
    "C": dict(
        db_path=f"{DB_BASE}/database_Operators/",
        db_functions=f"{DB_BASE}/database_Modules_ollama",
        db_functions_ollama=f"{DB_BASE}/database_Modules_ollama",
        modules_name="Layers",
        codes_database="Operators",
        base_results_dir=f"{RESULTS_BASE}/DeepONet_codes",
        max_iterations_all=100,
    ),
}


def get_config(family, overrides=None):
    """Return a copy of the family's default config merged with `overrides`."""
    family = family.upper()
    if family not in DB_CONFIG:
        raise ValueError(f"Unknown family {family!r}; expected one of A, B, C.")
    cfg = dict(DB_CONFIG[family])
    if overrides:
        cfg.update(overrides)
    return cfg


# ==========================================================================
# Per-family run logging.
# Every HENA run mirrors its full stdout transcript to a plain-text file under
#     <LOG_BASE>/<family>/hena_<family>_<name>_<timestamp>.txt
# so each run is auditable and grouped by family (A_SciC / B_PINN / C_Operator).
# The file is opened with a timestamp name up front; once the run discovers the
# problem name (from the filing/naming step) the file is renamed to include it.
# ==========================================================================
LOG_BASE = os.path.join(os.path.dirname(__file__), "HENA_logs")
FAMILY_LABEL = {"A": "A_SciC", "B": "B_PINN", "C": "C_Operator"}

# Normal-run convention: mirror each notebook's <ProblemName>_log.txt into its
# Examples_<family> folder. Family A defaults to Examples_NM; if the agents
# determine the problem is analytical-exact, the log may be moved to
# Examples_analytical_exact afterwards (see README).
EXAMPLES_DIR = {"A": "Examples_NM", "B": "Examples_PINNs", "C": "Examples_DeepONet"}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Tee:
    """Write everything sent to stdout to both the terminal and the log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            # Flush each write so the transcript is saved live (survives a kill).
            try:
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            s.flush()


def _slug(text, fallback="run"):
    """Filesystem-safe short slug for log file names."""
    if not text:
        return fallback
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(text))
    keep = keep.strip("_")
    return keep[:60] or fallback


class _RunLogger:
    """Context manager that tees stdout to a per-family log file and renames it
    to include the discovered problem name on exit."""

    def __init__(self, family, log_base=None, run_name=None):
        self.family = family.upper()
        self.run_name = run_name
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if log_base is None:
            # Default: mirror the notebook convention —
            # Examples_<family>/<ProblemName>_log.txt at the repo root.
            self.fam_dir = os.path.join(
                REPO_ROOT,
                EXAMPLES_DIR.get(self.family, FAMILY_LABEL.get(self.family, self.family)))
            self._examples_style = True
        else:
            # Explicit override keeps the <log_base>/<FAMILY_LABEL>/hena_*.txt layout.
            self.fam_dir = os.path.join(log_base, FAMILY_LABEL.get(self.family, self.family))
            self._examples_style = False
        self.path = None
        self._file = None
        self._prev_stdout = None

    def _filename(self, base):
        """Log file name for a given problem/run base name."""
        if self._examples_style:
            return f"{base}_log.txt"
        return f"hena_{self.family}_{base}_{self.stamp}.txt"

    def __enter__(self):
        os.makedirs(self.fam_dir, exist_ok=True)
        init_name = _slug(self.run_name, fallback="run")
        self.path = os.path.join(self.fam_dir, self._filename(init_name))
        self._file = open(self.path, "a", encoding="utf-8")
        self._prev_stdout = sys.stdout
        sys.stdout = _Tee(self._prev_stdout, self._file)
        print(f"[HENA] family={self.family} | logging to: {self.path}")
        return self

    def finalize_name(self, result):
        """Pick the best human name from the run result for the log filename."""
        name = self.run_name
        if isinstance(result, dict):
            pd = result.get("project_dir") or ""
            fn = result.get("filename") or ""
            name = (os.path.basename(os.path.normpath(pd)) if pd else "") \
                or (os.path.splitext(os.path.basename(fn))[0] if fn else "") \
                or name
        self._final_name = name

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._prev_stdout
        try:
            self._file.close()
        except Exception:
            pass
        # Rename to include the discovered problem name, if any.
        final = _slug(getattr(self, "_final_name", None), fallback="")
        if final:
            new_path = os.path.join(self.fam_dir, self._filename(final))
            if new_path != self.path:
                try:
                    os.replace(self.path, new_path)
                    self.path = new_path
                except OSError:
                    pass
        print(f"[HENA] log saved: {self.path}")
        return False  # never suppress exceptions


# ==========================================================================
# Agent factories — one per family (lifted from each notebook's cell 3).
# ==========================================================================
def build_agents_sciC(cfg):
    """Family A agents (teams_SciC)."""
    # Strategist: notebook overrides the gemini choice with gpt-5.4; keep the
    # effective one (swap model/provider here if desired).
    strategist_agent = memoryAgent(
        model='gpt-5.4', verbose=False,
        system_prompt=TA.strategist_system_prompt, model_provider='openai')
    critic_agent = memoryAgent(
        model='gpt-5.4', verbose=False,
        system_prompt=TA.strategist_critic_system_prompt, model_provider='openai')

    retriever_agent = Retriever_Agent(
        model='gpt-oss:120b', verbose=False,
        system_prompt=TA.Retriever_system_prompt,
        embeddings_provider='google', embeddings_model='models/gemini-embedding-001',
        db_path=cfg['db_path'], collection_name=cfg['codes_database'], k=3)
    validator_agent = memoryAgent(
        model='grok-4-1-fast-reasoning', verbose=False,
        system_prompt=TA.Validator_system_prompt, model_provider='grok')

    planner_agent = memoryAgent(
        model='gemini-3.1-pro-preview', verbose=False,
        system_prompt=TA.planner_agent_system_prompt_numerical, model_provider='google')
    planner_parser_agent = StructuredAgent(
        model='gpt-oss:120b', system_prompt=TA.planner_parser_system_prompt, verbose=False)
    patcher_agent = StructuredAgent(
        model='gpt-oss:120b', system_prompt=TA.patcher_agent_system_prompt_PINN, verbose=False)

    inspector_agent = StructuredAgent(
        model='grok-4-1-fast-reasoning', model_provider='grok',
        system_prompt=TA.inspector_system_prompt, verbose=False)
    naming_agent = StructuredAgent(
        model='gpt-oss:120b', verbose=False,
        system_prompt=TA.naming_agent_system_prompt_PINN)
    debugger_agent = StructuredAgent(
        model='claude-sonnet-4-6', verbose=False,
        system_prompt=TA.debugger_system_prompt_numerical, model_provider='claude')

    advisor_agent = aRAGAgent_multimodal(
        model='gemini-3.1-pro-preview', verbose=True,
        system_prompt=TA.advisor_system_prompt_numerical, model_provider='google',
        embeddings_provider='google', embeddings_model='models/gemini-embedding-001',
        db_path=cfg['db_functions'], collection_name=cfg['modules_name'], k=5)
    advisor_parser = StructuredAgent(
        model='grok-4-1-fast-reasoning', model_provider='grok',
        system_prompt=TA.inspector_system_prompt, verbose=False)
    register_agent_parser = StructuredAgent(
        model='grok-4-1-fast-reasoning', model_provider='grok',
        system_prompt=TA.register_agent_system_prompt_numerical, verbose=False)

    return dict(
        strategist_agent=strategist_agent, critic_agent=critic_agent,
        retriever_agent=retriever_agent, validator_agent=validator_agent,
        planner_agent=planner_agent, planner_parser_agent=planner_parser_agent,
        patcher_agent=patcher_agent, inspector_agent=inspector_agent,
        naming_agent=naming_agent, debugger_agent=debugger_agent,
        advisor_agent=advisor_agent, advisor_parser=advisor_parser,
        register_agent_parser=register_agent_parser)


def build_agents_pinn(cfg):
    """Family B agents (teams_SciML_PIML)."""
    strategist_agent = memoryAgent(
        model='gemini-3.1-pro-preview', verbose=False,
        system_prompt=TB.strategist_PINN_system_prompt, model_provider='google')
    # NOTE: the original notebook referenced strategist_critic_numerical_prompt_PIML,
    # which does not exist in this repo; the correct symbol is *_system_prompt_PIML.
    critic_agent = memoryAgent(
        model='gpt-5.4', verbose=False,
        system_prompt=TB.strategist_critic_system_prompt_PIML, model_provider='openai')

    retriever_agent = Retriever_Agent(
        model='gpt-oss:120b', verbose=False,
        system_prompt=TB.Retriever_system_prompt_PIML,
        embeddings_provider='google', embeddings_model='models/gemini-embedding-001',
        db_path=cfg['db_path'], collection_name=cfg['codes_database'], k=3)
    validator_agent = memoryAgent(
        model='grok-4-1-fast-reasoning', verbose=False,
        system_prompt=TB.Validator_system_prompt_PIML, model_provider='grok')
    collector_agent = Retriever_Agent(
        model='gpt-oss:120b', verbose=False,
        system_prompt=TB.collector_system_prompt_PIML,
        embeddings_provider='google', embeddings_model='models/gemini-embedding-001',
        db_path=cfg['db_functions_ollama'], collection_name=cfg['modules_name'], k=1)
    librarian_agent = memoryAgent(
        model='gemini-3.1-pro-preview', verbose=False,
        system_prompt=TB.librarian_system_prompt_PIML, model_provider='google')

    planner_agent = memoryAgent(
        model='gemini-3.1-pro-preview', verbose=False,
        system_prompt=TB.planner_agent_system_prompt_PIML, model_provider='google')
    planner_parser_agent = StructuredAgent(
        model='gpt-oss:120b', system_prompt=TB.planner_parser_system_prompt, verbose=False)
    patcher_agent = StructuredAgent(
        model='gpt-oss:120b', system_prompt=TB.patcher_agent_system_prompt_PINN, verbose=False)

    inspector_agent = StructuredAgent(
        model='grok-4-1-fast-reasoning', model_provider='grok',
        system_prompt=TB.inspector_system_prompt_PIML, verbose=False)
    naming_agent = StructuredAgent(
        model='gpt-oss:120b', verbose=False,
        system_prompt=TB.naming_agent_system_prompt_PINN)
    debugger_agent = StructuredAgent(
        model='claude-sonnet-4-6', verbose=False,
        system_prompt=TB.debugger_system_prompt_numerical, model_provider='claude')

    advisor_agent = aRAGAgent_multimodal(
        model='gemini-3.1-pro-preview', verbose=True,
        system_prompt=TB.advisor_system_prompt_PIML, model_provider='google',
        embeddings_provider='google', embeddings_model='models/gemini-embedding-001',
        db_path=cfg['db_functions_ollama'], collection_name=cfg['modules_name'], k=5)
    advisor_parser = StructuredAgent(
        model='gpt-oss:120b', verbose=False,
        system_prompt=TB.advisor_parser_system_prompt_ON)
    register_agent_parser = StructuredAgent(
        model='grok-4-1-fast-reasoning', model_provider='grok',
        system_prompt=TB.register_agent_system_prompt_numerical, verbose=False)

    return dict(
        strategist_agent=strategist_agent, critic_agent=critic_agent,
        retriever_agent=retriever_agent, validator_agent=validator_agent,
        collector_agent=collector_agent, librarian_agent=librarian_agent,
        planner_agent=planner_agent, planner_parser_agent=planner_parser_agent,
        patcher_agent=patcher_agent, inspector_agent=inspector_agent,
        naming_agent=naming_agent, debugger_agent=debugger_agent,
        advisor_agent=advisor_agent, advisor_parser=advisor_parser,
        register_agent_parser=register_agent_parser)


def build_agents_operator(cfg):
    """Family C agents (teams_SciML_Operators)."""
    naming_agent = StructuredAgent(
        model='gpt-oss:120b', verbose=False,
        system_prompt=TC.naming_agent_system_prompt_PINN)
    retriever_agent = Retriever_Agent(
        model='gpt-oss:120b', verbose=False,
        system_prompt=TC.Retriever_system_prompt_ON,
        embeddings_provider='google', embeddings_model='models/gemini-embedding-001',
        db_path=cfg['db_path'], collection_name=cfg['codes_database'], k=3)
    collector_agent = Retriever_Agent(
        model='gpt-oss:120b', verbose=False,
        system_prompt=TC.collector_system_prompt_ON,
        embeddings_provider='google', embeddings_model='models/gemini-embedding-001',
        db_path=cfg['db_functions_ollama'], collection_name=cfg['modules_name'], k=1)
    librarian_agent = memoryAgent(
        model='gemini-3.1-pro-preview', verbose=False,
        system_prompt=TC.librarian_system_prompt_ON, model_provider='google')
    planner_agent = memoryAgent(
        model='gemini-3-flash-preview', verbose=False,
        system_prompt=TC.planner_agent_system_prompt_ON, model_provider='google')
    patcher_agent = StructuredAgent(
        model='gpt-oss:120b', system_prompt=TC.patcher_agent_system_prompt_PINN, verbose=False)
    debugger_agent = StructuredAgent(
        model='gemini-3-flash-preview', model_provider='google',
        system_prompt=TC.debugger_system_prompt_PINN, verbose=False)

    advisor_agent = aRAGAgent_multimodal(
        model='gemini-3.1-pro-preview', verbose=True,
        system_prompt=TC.advisor_system_prompt_ON, model_provider='google',
        embeddings_provider='google', embeddings_model='models/gemini-embedding-001',
        db_path=cfg['db_functions'], collection_name=cfg['modules_name'], k=5)
    advisor_parser = StructuredAgent(
        model='gpt-oss:120b', verbose=False,
        system_prompt=TC.advisor_parser_system_prompt_ON)
    strategist_agent = aRAGAgent_single_chunk(
        model='gemini-3-flash-preview', verbose=False,
        system_prompt=TC.strategist_agent_system_prompt_ON, model_provider='google',
        embeddings_provider='google', embeddings_model='models/gemini-embedding-001',
        db_path=cfg['db_functions'], collection_name=cfg['modules_name'], k=2)
    register_agent_parser = StructuredAgent(
        model='gpt-oss:120b', verbose=False,
        system_prompt=TC.register_agent_system_prompt_ON)

    return dict(
        naming_agent=naming_agent, retriever_agent=retriever_agent,
        collector_agent=collector_agent, librarian_agent=librarian_agent,
        planner_agent=planner_agent, patcher_agent=patcher_agent,
        debugger_agent=debugger_agent, advisor_agent=advisor_agent,
        advisor_parser=advisor_parser, strategist_agent=strategist_agent,
        register_agent_parser=register_agent_parser)


# ==========================================================================
# HENA loops — one per family (lifted verbatim from the notebooks).
# ==========================================================================
def run_hena_sciC(user_question, agents, cfg,
                  checkpoint_project_dir='', load_code_only=False,
                  specific_requests=''):
    """Family A loop (teams_SciC). Mirrors Examples_NM / Examples_analytical_exact."""
    a = agents
    strategist_agent = a['strategist_agent']
    critic_agent = a['critic_agent']
    retriever_agent = a['retriever_agent']
    validator_agent = a['validator_agent']
    planner_agent = a['planner_agent']
    planner_parser_agent = a['planner_parser_agent']
    patcher_agent = a['patcher_agent']
    inspector_agent = a['inspector_agent']
    naming_agent = a['naming_agent']
    debugger_agent = a['debugger_agent']
    advisor_agent = a['advisor_agent']
    advisor_parser = a['advisor_parser']
    register_agent_parser = a['register_agent_parser']

    (strategist_report, error_report, history_log_string,
     history, initial_code, filename) = load_checkpoint_data(
        checkpoint_project_dir, load_code_only=load_code_only)
    print(error_report)

    max_iterations_all = cfg['max_iterations_all']
    base_results_dir = cfg['base_results_dir']
    refinement_iterations = 0
    is_approved = False
    new_code_details = 'Test'
    requires_name = True
    debug_approval = True
    critic_approval = False
    project_dir = ""

    while (is_approved == False and refinement_iterations < max_iterations_all):
        print(f"\n--- Code Creation Iteration: {refinement_iterations} ---")
        print("\n--- CURRENT TRIAL HISTORY ---")
        print(history_log_string)
        print("---------------------------\n")
        if debug_approval:
            print("--- 1. Calling Strategist Team ---")
            strategist_report, critic_approval = TA.strategy_team(
                user_question + specific_requests,
                strategist_agent, critic_agent,
                history_log_string, error_report, max_iterations=5)
        else:
            print("--- There was an implementation error so we will keep the same strategy for this iteration---")
        if critic_approval:
            if initial_code == '':
                print("--- 2. Calling Collection Team ---")
                initial_code, filename, _, _ = TA.retrieval_team(
                    strategist_report, retriever_agent, validator_agent, max_iterations=5)
                print(f"\n--- Retrieved template: {filename} (First 5 lines) ---")
                print('\n'.join(initial_code.split('\n')[:5]))
                print("---------------------------------------------------\n")
            print("--- 3. Calling Implementation Team ---")
            filename, initial_code = TA.implementation_team(
                user_question, planner_agent, patcher_agent, planner_parser_agent,
                initial_code, filename, error_report, strategist_report,
                remove_comments=False)
            print("--- 4. Calling Debugging Team ---")
            prompt_inspector = (
                f"[SCIENTIFIC_STRATEGY_REPORT]:\n{strategist_report}\n\n"
                f"[FINAL_GENERATED_CODE]:\n{initial_code}\n\n")
            response_inspector = inspector_agent.run(prompt_inspector)
            implementation_details = response_inspector['summary']
            generated_data_details = response_inspector['data_saving']
            implementation_approval = check_approval(response_inspector['status'])
            print('Inspector summary:\n', implementation_details)
            if implementation_approval:
                project_dir, filename, final_filepath, new_code_details = TA.filing_team_PINN(
                    user_question, filename, initial_code, naming_agent,
                    requires_name, new_code_details, base_results_dir=base_results_dir)
                requires_name = False
                success, log_file_name = run_code(project_dir, filename, final_filepath)
                status, debugger_report, debug_approval, full_log_for_agent, log_content = TA.debugging_team_PINN(
                    user_question, final_filepath, log_file_name, debugger_agent)
                if debug_approval:
                    print("--- 5. Calling Advisor Team ---")
                    error_report, is_approved, run_summary = TA.advisor_team(
                        user_question=user_question, history_log_string=history_log_string,
                        project_dir=project_dir, log_content=log_content,
                        strategist_report=strategist_report,
                        implementation_details=implementation_details,
                        advisor_agent=advisor_agent, advisor_parser=advisor_parser,
                        register_agent_parser=register_agent_parser, max_width=800)
                    run_summary['implementation_details'] = implementation_details
                    run_summary['project_dir'] = project_dir
                    run_summary['refinement_iterations'] = refinement_iterations
                    run_summary['data_saving'] = generated_data_details
                    history.append(run_summary)
                    history_log_string = update_and_save_report(
                        history=history, user_question=user_question,
                        strategist_report=strategist_report,
                        error_report=error_report, project_dir=project_dir)
                    if run_summary['change_architecture']:
                        print("--- Looking for a new code template---")
                        initial_code = ''
                else:
                    error_report = f"Debugger Analysis:\n{debugger_report}\n"
            else:
                debug_approval = False
                discrepancies = response_inspector['discrepancies']
                debugger_report = 'There are discrepancies with the base strategy:\n' + discrepancies
                print('Report:', debugger_report)
                error_report = f"Debugger Analysis:\n{debugger_report}\n"
        else:
            print('The agents did not get into an agreement')
            break
        print("--- Clearing agent short-term memories ---")
        strategist_agent.clear_history()
        critic_agent.clear_history()
        retriever_agent.clear_history()
        planner_agent.clear_history()
        validator_agent.clear_history()
        advisor_agent.clear_history()
        refinement_iterations = refinement_iterations + 1

    return dict(
        family="A", is_approved=is_approved, project_dir=project_dir,
        filename=filename, initial_code=initial_code,
        history=history, history_log_string=history_log_string,
        error_report=error_report, refinement_iterations=refinement_iterations)


def run_hena_pinn(user_question, agents, cfg,
                  checkpoint_project_dir='', load_code_only=False,
                  specific_requests=''):
    """Family B loop (teams_SciML_PIML). Mirrors Examples_PINNs."""
    a = agents
    strategist_agent = a['strategist_agent']
    critic_agent = a['critic_agent']
    retriever_agent = a['retriever_agent']
    validator_agent = a['validator_agent']
    collector_agent = a['collector_agent']
    librarian_agent = a['librarian_agent']
    planner_agent = a['planner_agent']
    planner_parser_agent = a['planner_parser_agent']
    patcher_agent = a['patcher_agent']
    inspector_agent = a['inspector_agent']
    naming_agent = a['naming_agent']
    debugger_agent = a['debugger_agent']
    advisor_agent = a['advisor_agent']
    advisor_parser = a['advisor_parser']
    register_agent_parser = a['register_agent_parser']

    (strategist_report, error_report, history_log_string,
     history, initial_code, filename) = load_checkpoint_data(
        checkpoint_project_dir, load_code_only=load_code_only)
    print(error_report)

    # B preloads its auxiliary modules from a separate checkpoint dir (the
    # Examples_PINNs notebooks load my_functions from .../codes/SciML/Modules/).
    functions_ckpt = cfg.get('functions_checkpoint_dir', '')
    if functions_ckpt:
        _, _, _, _, my_functions, _ = load_checkpoint_data(functions_ckpt, load_code_only=True)
    else:
        my_functions = ''

    max_iterations_all = cfg['max_iterations_all']
    base_results_dir = cfg['base_results_dir']
    refinement_iterations = 0
    is_approved = False
    new_code_details = 'Test'
    clean_functions = ''
    requires_name = True
    debug_approval = True
    critic_approval = False
    project_dir = ""

    while (is_approved == False and refinement_iterations < max_iterations_all):
        print(f"\n--- Code Creation Iteration: {refinement_iterations} ---")
        print("\n--- CURRENT TRIAL HISTORY ---")
        print(history_log_string)
        print("---------------------------\n")
        if debug_approval:
            print("--- 1. Calling Strategist Team ---")
            strategist_report, critic_approval = TB.strategy_team(
                user_question + specific_requests,
                strategist_agent, critic_agent,
                history_log_string, error_report, max_iterations=5)
        else:
            print("--- There was an implementation error so we will keep the same strategy for this iteration---")
        if critic_approval:
            if debug_approval:
                if initial_code == '':
                    print("--- 2. Calling Collection Team (Main files) ---")
                    initial_code, filename, _, _ = TB.retrieval_team(
                        strategist_report, retriever_agent, validator_agent, max_iterations=5)
                    print(f"\n--- Retrieved template: {filename} (First 5 lines) ---")
                    print('\n'.join(initial_code.split('\n')[:5]))
                    print("---------------------------------------------------\n")
                if my_functions == '':
                    print("--- 2.5 Retrieving auxiliary functions ---")
                    my_functions, filename_aux, _, _ = TB.retrieval_team_direct(
                        user_question=strategist_report,
                        retriever_agent=collector_agent, max_iterations=5)
                    print(f"\n--- Retrieved functions: {filename_aux} (First 5 lines) ---")
                    print('\n'.join(my_functions.split('\n')[:5]))
                    print("---------------------------------------------------\n")
                prompt_librarian = (
                    f"[STRATEGY_REPORT]:\n{strategist_report}\n\n"
                    f"[CODE_LIBRARY]:\n{my_functions}\n")
                response_librarian = librarian_agent.run(prompt_librarian)
                clean_functions = response_librarian['output']
                print(clean_functions)
            print("--- 3. Calling Implementation Team ---")
            filename, initial_code = TB.implementation_team(
                user_question, planner_agent, patcher_agent, planner_parser_agent,
                initial_code, clean_functions, filename, error_report, strategist_report,
                remove_comments=False)
            print("--- 4. Calling Debugging Team ---")
            prompt_inspector = (
                f"[SCIENTIFIC_STRATEGY_REPORT]:\n{strategist_report}\n\n"
                f"[FINAL_GENERATED_CODE]:\n{initial_code}\n\n")
            response_inspector = inspector_agent.run(prompt_inspector)
            implementation_details = response_inspector['summary']
            generated_data_details = response_inspector['data_saving']
            implementation_approval = check_approval(response_inspector['status'])
            print('Inspector summary:\n', implementation_details)
            if implementation_approval:
                project_dir, filename, final_filepath, new_code_details = TB.filing_team_PINN(
                    user_question, filename, initial_code, naming_agent,
                    requires_name, new_code_details, base_results_dir=base_results_dir)
                requires_name = False
                success, log_file_name = run_code(project_dir, filename, final_filepath)
                status, debugger_report, debug_approval, full_log_for_agent, log_content = TB.debugging_team_PINN(
                    user_question, final_filepath, log_file_name, filename, debugger_agent)
                if debug_approval:
                    print("--- 5. Calling Advisor Team ---")
                    error_report, is_approved, run_summary = TB.advisor_team(
                        user_question=user_question, history_log_string=history_log_string,
                        project_dir=project_dir, log_content=log_content,
                        strategist_report=strategist_report,
                        implementation_details=implementation_details,
                        advisor_agent=advisor_agent, advisor_parser=advisor_parser,
                        register_agent_parser=register_agent_parser, max_width=800)
                    run_summary['implementation_details'] = implementation_details
                    run_summary['project_dir'] = project_dir
                    run_summary['refinement_iterations'] = refinement_iterations
                    run_summary['data_saving'] = generated_data_details
                    history.append(run_summary)
                    history_log_string = update_and_save_report(
                        history=history, user_question=user_question,
                        strategist_report=strategist_report,
                        error_report=error_report, project_dir=project_dir)
                    # NOTE: change_architecture template reset is intentionally
                    # disabled in the PINN notebook; preserved as-is.
                else:
                    error_report = f"Debugger Analysis:\n{debugger_report}\n"
            else:
                debug_approval = False
                discrepancies = response_inspector['discrepancies']
                debugger_report = 'There are discrepancies with the base strategy:\n' + discrepancies
                print('Report:', debugger_report)
                error_report = f"Debugger Analysis:\n{debugger_report}\n"
        else:
            print('The agents did not get into an agreement')
            break
        print("--- Clearing agent short-term memories ---")
        strategist_agent.clear_history()
        critic_agent.clear_history()
        retriever_agent.clear_history()
        planner_agent.clear_history()
        validator_agent.clear_history()
        advisor_agent.clear_history()
        refinement_iterations = refinement_iterations + 1

    return dict(
        family="B", is_approved=is_approved, project_dir=project_dir,
        filename=filename, initial_code=initial_code,
        history=history, history_log_string=history_log_string,
        error_report=error_report, refinement_iterations=refinement_iterations)


def run_hena_operator(user_question, agents, cfg,
                      checkpoint_project_dir='', load_code_only=False,
                      specific_requests=''):
    """Family C loop (teams_SciML_Operators). Mirrors Examples_DeepONet."""
    a = agents
    strategist_agent = a['strategist_agent']
    retriever_agent = a['retriever_agent']
    collector_agent = a['collector_agent']
    librarian_agent = a['librarian_agent']
    planner_agent = a['planner_agent']
    patcher_agent = a['patcher_agent']
    debugger_agent = a['debugger_agent']
    naming_agent = a['naming_agent']
    advisor_agent = a['advisor_agent']
    advisor_parser = a['advisor_parser']
    register_agent_parser = a['register_agent_parser']

    (strategist_report, error_report, history_log_string,
     history, file_content, filename) = load_checkpoint_data(
        checkpoint_project_dir, load_code_only=load_code_only)

    max_iterations_all = cfg['max_iterations_all']
    base_results_dir = cfg['base_results_dir']
    refinement_iterations = 0
    is_approved = False
    new_code_details = 'Test'
    clean_functions = ''
    my_functions = ''
    requires_name = True
    debug_approval = True
    project_dir = ""

    while (is_approved == False and refinement_iterations < max_iterations_all):
        print(f"\n--- Code Creation Iteration: {refinement_iterations} ---")
        print("\n--- CURRENT TRIAL HISTORY ---")
        print(history_log_string)
        print("---------------------------\n")

        filename, file_content, strategist_report, clean_functions, my_functions = TC.code_creation_team_ON(
            user_question=user_question + specific_requests,
            history_log_string=history_log_string,
            initial_code=file_content, filename=filename,
            error_report=error_report, strategist_report=strategist_report,
            strategist_agent=strategist_agent, collector_agent=collector_agent,
            librarian_agent=librarian_agent, retriever_agent=retriever_agent,
            planner_agent=planner_agent, patcher_agent=patcher_agent,
            debug_approval=debug_approval, clean_functions=clean_functions,
            my_functions=my_functions)

        project_dir, filename, final_filepath, new_code_details = TC.filing_team_PINN(
            user_question, filename, file_content, naming_agent,
            requires_name, new_code_details, base_results_dir=base_results_dir)
        requires_name = False
        success, log_file_name = run_code(project_dir, filename, final_filepath)

        status, debugger_report, debug_approval, full_log_for_agent, log_content = TC.debugging_team_PINN(
            user_question, final_filepath, log_file_name, debugger_agent)

        if debug_approval:
            error_report, is_approved, run_summary = TC.advisor_team_ON(
                user_question=user_question, history_log_string=history_log_string,
                project_dir=project_dir, log_content=log_content,
                strategist_report=strategist_report,
                advisor_agent=advisor_agent, advisor_parser=advisor_parser,
                register_agent_parser=register_agent_parser, max_width=500)
            history.append(run_summary)
            history_log_string = update_and_save_report(
                history=history, user_question=user_question,
                strategist_report=strategist_report,
                error_report=error_report, project_dir=project_dir)
            print("--- Clearing agent short-term memories ---")
            strategist_agent.clear_history()
            retriever_agent.clear_history()
            collector_agent.clear_history()
            librarian_agent.clear_history()
            advisor_agent.clear_history()
            if run_summary.get('change_architecture') == True:
                print("--- Gate Check: change_architecture is TRUE. Re-initializing file_content to force new template retrieval. ---")
                file_content = ''
        else:
            error_report = f"Debugger Analysis:\n{debugger_report}\n\n" + f"Log File:\n{full_log_for_agent}\n"

        refinement_iterations = refinement_iterations + 1

    return dict(
        family="C", is_approved=is_approved, project_dir=project_dir,
        filename=filename, initial_code=file_content,
        history=history, history_log_string=history_log_string,
        error_report=error_report, refinement_iterations=refinement_iterations)


# ==========================================================================
# Dispatcher.
# ==========================================================================
BUILDERS = {"A": build_agents_sciC, "B": build_agents_pinn, "C": build_agents_operator}
RUNNERS = {"A": run_hena_sciC, "B": run_hena_pinn, "C": run_hena_operator}


def build_agents(family, cfg=None):
    """Build the agent set for a family ('A'|'B'|'C'). cfg defaults to get_config(family)."""
    family = family.upper()
    if family not in BUILDERS:
        raise ValueError(f"Unknown family {family!r}; expected one of A, B, C.")
    cfg = cfg or get_config(family)
    return BUILDERS[family](cfg)


def run_hena(family, user_question, *, agents=None, cfg=None,
             checkpoint_project_dir='', load_code_only=False, specific_requests='',
             config_overrides=None, log=True, log_base=None, run_name=None):
    """
    Run the HENA loop for a problem family.

    family            : 'A' (SciC), 'B' (PINN), or 'C' (Operator).
    user_question     : the problem statement string (from a .txt or a request JSON).
    agents            : prebuilt agent dict (reuse across calls); built if None.
    cfg               : full config dict; defaults to get_config(family, config_overrides).
    config_overrides  : partial dict merged onto the family default config.
    checkpoint_project_dir / load_code_only : forwarded to load_checkpoint_data.
    specific_requests : extra instruction string appended to the strategist prompt.
    log               : if True, mirror the full transcript to a per-family log file.
    log_base          : root dir for logs (default <module dir>/HENA_logs).
    run_name          : initial log name hint; overridden by the discovered
                        problem name once the run produces one.

    Returns the runner's result dict, with 'log_path' added when logging is on.
    """
    family = family.upper()
    if family not in RUNNERS:
        raise ValueError(f"Unknown family {family!r}; expected one of A, B, C.")
    cfg = cfg or get_config(family, config_overrides)
    if agents is None:
        agents = BUILDERS[family](cfg)

    def _go():
        return RUNNERS[family](
            user_question, agents, cfg,
            checkpoint_project_dir=checkpoint_project_dir,
            load_code_only=load_code_only, specific_requests=specific_requests)

    if not log:
        return _go()

    with _RunLogger(family, log_base=log_base, run_name=run_name) as logger:
        result = _go()
        logger.finalize_name(result)
    if isinstance(result, dict):
        result["log_path"] = logger.path
    return result
