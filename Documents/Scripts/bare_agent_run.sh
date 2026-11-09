#!/bin/bash
# Baseline without ATHENA: one model in a coding-agent harness (Claude Code or OpenCode), in
# the same release box, given only the user request of an ATHENA case and a working folder. No
# ATHENA code, templates, vector DBs, Strategist/Critic/Advisor or rubric; no web tools. It plans,
# writes, runs and iterates by itself.
#
#   <RUN_DIR>/
#     job_info.txt        node, GPU, start/end, model, harness, image, exit code
#     prompt.md           the exact message the agent received
#     events.jsonl        the harness's JSON event stream (every tool call, tokens per step)
#     run.log             harness stderr
#     usage_summary.json  token totals (and cost, turns, duration for Claude Code)
#     network_commands.txt  shell commands that could have reached the network (web tools are denied)
#     box/                the agent's only writable (and only visible) folder, mounted at /box;
#                         box/final/ holds the code and outputs it hands in; box/.box/ harness state
#
# Usage: bare_agent_run.sh <request.json> <harness/model> <RUN_DIR>
#   harness/model: claude/claude-opus-5-5 (Claude Code, subscription token),
#                  google/gemini-3.1-pro-preview (OpenCode, Gemini API key) or
#                  ollama/glm-5.3 (OpenCode, Ollama Cloud: OLLAMA_API in crunch_llm/code/.env,
#                  reasoning effort high and 128k output, as in Open-GRAFT run_agent.sh)
#   Subagents are allowed in every harness (Claude Code Agent tool, OpenCode task); web and cross-session tools are off.
#   Claude Code needs a long-lived token from `claude setup-token`: CLAUDE_CODE_OAUTH_TOKEN in
#   crunch_llm/code/.env (default; ATHENA_ENV_FILE points elsewhere), or CLAUDE_TOKEN_FILE=<one-line token file>.
set -u
REQUEST="$1"; MODEL="$2"; RUN_DIR="$3"
REPO="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/../..")"
SIF="${ATHENA_SIF:-$REPO/container/images/athena_release.sif}"
[ -f "$SIF" ] || { echo "container image not found: $SIF" >&2; exit 1; }
[ -f "$REQUEST" ] || REQUEST="$REPO/$REQUEST"
[ -f "$REQUEST" ] || { echo "request not found: $1" >&2; exit 1; }
HARNESS="${MODEL%%/*}"; MID="${MODEL#*/}"
case "$HARNESS" in claude|google|ollama) ;; *) echo "harness must be claude/<model>, google/<model> or ollama/<model>" >&2; exit 1 ;; esac
mkdir -p "$RUN_DIR"; RUN_DIR="$(readlink -f "$RUN_DIR")"
case "$RUN_DIR/" in "$REPO"/Results/*) ;; *) echo "RUN_DIR must be under $REPO/Results/" >&2; exit 1 ;; esac
[ -e "$RUN_DIR/events.jsonl" ] && { echo "Refusing to overwrite an existing run: $RUN_DIR" >&2; exit 1; }
BOX="$RUN_DIR/box"
mkdir -p "$BOX/final" "$BOX/.box"/{home,julia_depot,xdg_config/opencode,xdg_data,xdg_cache,xdg_state}

NCPU="${SLURM_CPUS_PER_TASK:-$(nproc)}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then HW="$NCPU CPU cores and one NVIDIA GPU"; else HW="$NCPU CPU cores, no GPU"; fi

# --- the prompt: the case's user_request verbatim and what the box contains, nothing else ---
read_req() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$REQUEST" "$1"; }
# {REPO} in a request stands for the repository root (input files under Documents/codes/Data)
USER_REQUEST="$(read_req 'd["user_request"]')"
USER_REQUEST="${USER_REQUEST//\{REPO\}/$REPO}"
cat > "$RUN_DIR/prompt.md" <<EOF
You are working on the computational task below. Your working folder is /box; any input file the task names is available read-only at its path.

Hardware: $HW.

Plan your approach, write the code, run it, check the results, and keep improving until you are satisfied that the result is correct and complete. When you are done, put the final code and every output the task asks for in /box/final/, and write /box/final/REPORT.md: the method, how you verified the result, the final accuracy you obtained, and any remaining issues.

TASK:
$USER_REQUEST
EOF
cp "$RUN_DIR/prompt.md" "$BOX/.box/prompt.md"

# --- input files the request names by absolute path: bound read-only at that same path ---
IN_BINDS=()
for p in $(grep -oE '(^|[ (`"])/[^ )`"]+\.(npz|npy|mat|h5|csv|txt)' <<<"$USER_REQUEST" | sed 's/^[ (`"]//' | sort -u); do
    src="$p"
    [ -f "$src" ] || { echo "input file named in the request not found: $p" >&2; exit 1; }
    IN_BINDS+=(-B "$(readlink -f "$src"):$p:ro")
    echo "input $p <- $(readlink -f "$src")" >> "$RUN_DIR/job_info.txt"
done

# --- worked examples hidden: the examples/, docs/ and test/ folders at the top of every installed
#     Julia package (e.g. Trixi ships elixir_euler_rayleigh_taylor_instability.jl, the RT case
#     itself) are covered by an empty read-only folder. Package source (src/) stays readable. ---
HIDE="$RUN_DIR/.hidden_empty"; mkdir -p "$HIDE"
mapfile -t HIDDEN < <(apptainer exec --containall "$SIF" find /opt/julia_depot/packages -mindepth 3 -maxdepth 3 -type d \
    \( -name examples -o -name example -o -name docs -o -name doc -o -name test -o -name tests -o -name notebooks -o -name tutorials \) 2>/dev/null)
[ "${#HIDDEN[@]}" -gt 100 ] || { echo "could not list the package example folders to hide" >&2; exit 1; }
for d in "${HIDDEN[@]}"; do IN_BINDS+=(-B "$HIDE:$d:ro"); done
echo "hidden: ${#HIDDEN[@]} examples/docs/test folders of the Julia packages" >> "$RUN_DIR/job_info.txt"

# --- harness: credentials into a private env file outside the box, and the command to run ---
ENV_FILE="${ATHENA_ENV_FILE:-$REPO/crunch_llm/code/.env}"
SECRET="$(mktemp "${TMPDIR:-/tmp}/bare_agent_env.XXXXXX")"; chmod 600 "$SECRET"
trap 'rm -f "$SECRET"' EXIT
H_BINDS=()
if [ "$HARNESS" = claude ]; then
    # subscription token (claude setup-token): CLAUDE_CODE_OAUTH_TOKEN in crunch_llm/code/.env
    # (or ATHENA_ENV_FILE), or a one-line token file given as CLAUDE_TOKEN_FILE
    TOKF="${CLAUDE_TOKEN_FILE:-$ENV_FILE}"
    if grep -q '^CLAUDE_CODE_OAUTH_TOKEN=' "$TOKF" 2>/dev/null; then
        TOK="$(grep -E '^CLAUDE_CODE_OAUTH_TOKEN=' "$TOKF" | head -1 | cut -d= -f2- | tr -d "\"' \r")"
    else
        TOK="$(tr -d ' \r\n' < "$TOKF" 2>/dev/null)"
    fi
    [ -n "$TOK" ] || { echo "no Claude token in $TOKF (run: claude setup-token)" >&2; exit 1; }
    CLAUDE_BIN="$(readlink -f "$(command -v claude)")"
    [ -x "$CLAUDE_BIN" ] || { echo "claude binary not found on the host" >&2; exit 1; }
    H_BINDS=(-B "$CLAUDE_BIN:/usr/local/bin/claude:ro")
    printf '%s\n' "CLAUDE_CODE_OAUTH_TOKEN=$TOK" DISABLE_AUTOUPDATER=1 DISABLE_TELEMETRY=1 \
        DISABLE_ERROR_REPORTING=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 > "$SECRET"
    VERSION="claude-code $("$CLAUDE_BIN" --version 2>/dev/null | cut -d' ' -f1)"
    AGENT_CMD='cd /box && exec claude -p "$(cat /box/.box/prompt.md)" --model "$0" --output-format stream-json --verbose --dangerously-skip-permissions --disallowedTools WebFetch WebSearch Workflow SendMessage ListAgents RemoteTrigger CronCreate CronDelete CronList ScheduleWakeup'
else
    # OpenCode with one model; subagents (task) allowed, as the Agent tool is in Claude Code
    if [ "$HARNESS" = google ]; then
        OUTMAX=65536; KEYVAR=GOOGLE_GENERATIVE_AI_API_KEY
        KEY="$(grep -E '^GOOGLE_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d "\"' \r")"
        PROVIDER="\"google\": { \"options\": { \"apiKey\": \"{env:$KEYVAR}\" },
      \"models\": { \"$MID\": { \"name\": \"$MID\", \"tool_call\": true,
                            \"limit\": { \"context\": 1048576, \"output\": $OUTMAX } } } }"
    else
        # Ollama Cloud through its OpenAI-compatible endpoint, same settings as Open-GRAFT run_agent.sh
        OUTMAX=131072; KEYVAR=OLLAMA_API_KEY
        KEY="$(grep -E '^OLLAMA_API=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d "\"' \r")"
        PROVIDER="\"ollama\": { \"npm\": \"@ai-sdk/openai-compatible\", \"name\": \"Ollama Cloud\",
      \"options\": { \"baseURL\": \"https://ollama.com/v1\", \"apiKey\": \"{env:$KEYVAR}\" },
      \"models\": { \"$MID\": { \"name\": \"$MID\", \"tool_call\": true,
                            \"limit\": { \"context\": 1048576, \"output\": $OUTMAX },
                            \"options\": { \"reasoningEffort\": \"high\" } } } }"
    fi
    [ -n "$KEY" ] || { echo "no API key for $HARNESS" >&2; exit 1; }
    cat > "$BOX/.box/xdg_config/opencode/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "autoupdate": false,
  "share": "disabled",
  "model": "$MODEL",
  "small_model": "$MODEL",
  "provider": {
    $PROVIDER
  },
  "permission": {
    "read": "allow", "edit": "allow", "glob": "allow", "grep": "allow", "list": "allow",
    "bash": "allow", "external_directory": "allow", "todowrite": "allow",
    "lsp": "allow", "skill": "allow", "doom_loop": "allow",
    "webfetch": "deny", "websearch": "deny", "question": "deny"
  }
}
JSON
    printf '%s\n' "$KEYVAR=$KEY" OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=$OUTMAX \
        OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_SHARE=1 OPENCODE_DISABLE_CLAUDE_CODE=1 > "$SECRET"
    VERSION="opencode $(apptainer exec --containall "$SIF" opencode --version 2>/dev/null)"
    MID="$MODEL"
    AGENT_CMD='cd /box && exec opencode run --auto --format json -m "$0" "$(cat /box/.box/prompt.md)"'
fi

{
    echo "job=${SLURM_JOB_ID:-none} node=$(hostname) cpus=$NCPU gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
    echo "request=$REQUEST model=$MODEL harness=$VERSION box=$SIF"
    echo "image_sha256=$(cut -d' ' -f1 "$SIF.sha256" 2>/dev/null)"
    echo "git=$(git -C "$REPO" rev-parse --short HEAD) branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
} >> "$RUN_DIR/job_info.txt"

GPU_ENV=(); [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && GPU_ENV=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
apptainer exec --nv --containall --cleanenv \
    -B "$BOX:/box" "${IN_BINDS[@]}" "${H_BINDS[@]}" \
    --home /box/.box/home \
    --pwd /box \
    --env-file "$SECRET" \
    --env XDG_CONFIG_HOME=/box/.box/xdg_config --env XDG_DATA_HOME=/box/.box/xdg_data \
    --env XDG_CACHE_HOME=/box/.box/xdg_cache --env XDG_STATE_HOME=/box/.box/xdg_state \
    --env ATHENA_JULIA_DEPOT=/box/.box/julia_depot \
    --env JULIA_PKG_PRECOMPILE_AUTO=0 --env JULIA_NUM_THREADS="$NCPU" \
    --env PATH=/opt/julia/bin:/opt/envs/nabla_jax/bin:/usr/local/bin:/usr/bin:/bin \
    "${GPU_ENV[@]}" \
    "$SIF" \
    /bin/bash -c "$AGENT_CMD" "$MID" \
    < /dev/null > "$RUN_DIR/events.jsonl" 2> "$RUN_DIR/run.log"
rc=$?
echo "exit=$rc end=$(date -Is)" >> "$RUN_DIR/job_info.txt"

# token totals (per-step numbers stay in events.jsonl), and every shell command that could reach
# the network (the web tools are denied, the shell is not)
if [ "$HARNESS" = claude ]; then
    # one result event per segment: the agent can end a turn while its background jobs run and
    # resume when they finish. turns and duration are summed over segments; cost, usage and
    # modelUsage are the session totals of the last segment.
    jq -s '[.[] | select(.type=="result")] as $r | ($r | last) as $l |
           {turns: ($r | map(.num_turns) | add), duration_ms: ($r | map(.duration_ms) | add),
            segments: ($r | map({num_turns, duration_ms, total_cost_usd})),
            total_cost_usd: $l.total_cost_usd, is_error: $l.is_error, subtype: $l.subtype,
            usage: $l.usage, modelUsage: $l.modelUsage}' \
        "$RUN_DIR/events.jsonl" > "$RUN_DIR/usage_summary.json" 2>/dev/null
    jq -r 'select(.type=="assistant") | .message.content[]? | select(.type=="tool_use" and .name=="Bash") | .input.command' \
        "$RUN_DIR/events.jsonl" 2>/dev/null > "$RUN_DIR/.bash_commands"
else
    jq -s '[.[] | select(.type=="step_finish") | .part.tokens] |
           {steps: length, input: (map(.input) | add), output: (map(.output) | add),
            reasoning: (map(.reasoning) | add), cache_read: (map(.cache.read) | add),
            cache_write: (map(.cache.write) | add), total: (map(.total) | add)}' \
        "$RUN_DIR/events.jsonl" > "$RUN_DIR/usage_summary.json" 2>/dev/null
    jq -r 'select(.type=="tool_use" and .part.tool=="bash") | .part.state.input.command // empty' \
        "$RUN_DIR/events.jsonl" 2>/dev/null > "$RUN_DIR/.bash_commands"
fi
grep -E 'curl|wget|pip install|Pkg\.add|git clone|http' "$RUN_DIR/.bash_commands" > "$RUN_DIR/network_commands.txt"
rm -f "$RUN_DIR/.bash_commands"
exit $rc
