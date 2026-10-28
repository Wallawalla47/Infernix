"""Route traces of a mixed task set for calibrated_order.py (candidates §23 NU-43): starts infernix-serve once per
speculative arm with --route-trace, sends the task set one request at a time over the OpenAI Chat Completions API,
and stops the server (Ctrl+Break on Windows, SIGTERM elsewhere) so the trace is complete. Python 3.11, no other
dependency.

The task set:
- the examples manifest's text fixtures (examples/cli/manifest.json), by domain: code (CUDA, Python, TypeScript,
  a code review), mathematics (three AIME 2026 problems, thinking), reasoning (a logic grid, thinking), prose (three
  stories), translation (three), structured output (JSONL, CSV, SQL) and a chat history;
- with --agent-plan, agent turns with tools: each actor's opening conversation from a bench/agentic_ab plan.json (a
  coding agent's resumed session of tool calls and results, its subagents' and reviewers' first turns) with its
  persona's tools.
Every request samples with the model's default settings and a fixed seed, at most --max-new tokens. Each arm writes
<out>/<arm>.bin (the trace), <arm>.requests.jsonl (each request's name, domain and token counts) and <arm>.log (the
server's output). The server runs with --expert-state off, so no saved expert state is read or overwritten, and one
lane, so the trace's requests are the task set's in order.

Usage (from the repository root, with the GPU free):

    py -3.11 -m tools.expert_cache_replay.collect_traces --serve build-windows/apps/Release/infernix-serve.exe \\
        --artifact <model>-00001-of-00003.infernix --out profiles/calibration/<model> \\
        [--arms plain,mtp,dflash2] [--arm-flags mtp="--spec mtp --draft-tokens 3"] [--serve-flags "..."] \\
        [--agent-plan <plan.json>] [--max-new 2048] [--port 8090]
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY / 'examples' / 'cli' / 'manifest.json'
MODEL_ID = 'calibration'
# The speculative arms calibrated_order.py mixes (NInfer's method drafts MTP 3 and DFlash 7).
ARM_FLAGS = {
    'plain': [],
    'mtp': ['--spec', 'mtp', '--draft-tokens', '3'],
    'dflash2': ['--spec', 'dflash2', '--draft-tokens', '7'],
}
# Every arm: one lane, a context for the agent turns (up to about 81K prompt tokens), no saved expert state.
BASE_FLAGS = ['--max-concurrency', '1', '--max-context', '131072', '--kv-dtype', 'int8', '--expert-state', 'off']
FIXTURES = {
    'code': ('scenario_code_cuda', 'scenario_code_python', 'scenario_code_typescript', 'text_code_review'),
    'math': ('long_decode_aime26_01', 'long_decode_aime26_15', 'long_decode_aime26_30'),
    'reasoning': ('thinking_logic_grid',),
    'prose': ('scenario_story_en_mystery', 'scenario_story_zh_scifi', 'scenario_story_zh_dialogue'),
    'translation': ('scenario_translation_zh_en', 'scenario_translation_en_zh', 'scenario_translation_markdown'),
    'structured': ('scenario_structured_jsonl', 'scenario_structured_csv', 'scenario_structured_sql'),
    'chat': ('text_chat_history',),
}
STARTUP_TIMEOUT_S = 1800.0
REQUEST_TIMEOUT_S = 3600.0
STOP_TIMEOUT_S = 300.0


def fixture_tasks(max_new):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    cases = {case['name']: case for case in manifest['cases']}
    tasks = []
    for domain, names in FIXTURES.items():
        for name in names:
            case = cases[name]
            messages = json.loads((MANIFEST.parent / case['messages']).read_text(encoding='utf-8'))
            body = {'messages': messages, 'max_completion_tokens': min(int(case['max_new']), max_new),
                    'enable_thinking': bool(case['thinking'])}
            tasks.append((name, domain, body))
    return tasks


def agent_tasks(plan_path, max_new):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    tasks = []
    for actor in plan['actors']:
        persona = plan['personas'][actor['persona']]
        messages = actor['initial']
        if not messages or messages[0].get('role') != 'system':
            messages = [{'role': 'system', 'content': persona['system']}] + list(messages)
        tasks.append((f"agent_{actor['name']}", 'agent',
                      {'messages': messages, 'tools': persona['tools'], 'max_completion_tokens': max_new}))
    return tasks


def wait_ready(process, port):
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'infernix-serve exited with status {process.returncode} during startup')
        try:
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            connection.request('GET', '/health')
            if connection.getresponse().status == 200:
                return
        except OSError:
            pass
        time.sleep(2.0)
    raise RuntimeError('infernix-serve did not become ready')


def send(port, body, seed):
    payload = dict(body, model=MODEL_ID, seed=seed, stream=False)
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=REQUEST_TIMEOUT_S)
    connection.request('POST', '/v1/chat/completions', body=data,
                       headers={'Content-Type': 'application/json', 'Content-Length': str(len(data))})
    response = connection.getresponse()
    text = response.read()
    if response.status != 200:
        raise RuntimeError(f'HTTP {response.status}: {text[:400]!r}')
    return json.loads(text)


def stop(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        process.send_signal(signal.CTRL_BREAK_EVENT)  # the serve's immediate stop (Ctrl+Break)
    else:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise RuntimeError('infernix-serve did not stop; the trace may be incomplete')


def run_arm(args, arm, flags, tasks):
    out = Path(args.out)
    trace = out / f'{arm}.bin'
    command = [str(args.serve), str(args.artifact), '--host', '127.0.0.1', '--port', str(args.port),
               '--model-id', MODEL_ID, '--route-trace', str(trace)] + BASE_FLAGS + flags + \
        shlex.split(args.serve_flags, posix=os.name != 'nt')
    print(f'[{arm}] {" ".join(command)}', flush=True)
    creation = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    with open(out / f'{arm}.log', 'wb') as log, open(out / f'{arm}.requests.jsonl', 'w', encoding='utf-8') as rows:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=creation)
        try:
            wait_ready(process, args.port)
            for index, (name, domain, body) in enumerate(tasks):
                began = time.monotonic()
                result = send(args.port, body, args.seed + index)
                usage = result.get('usage', {})
                row = {'index': index, 'name': name, 'domain': domain,
                       'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens'),
                       'finish_reason': (result.get('choices') or [{}])[0].get('finish_reason'),
                       'seconds': round(time.monotonic() - began, 2)}
                rows.write(json.dumps(row) + '\n')
                rows.flush()
                print(f'[{arm}] {index + 1}/{len(tasks)} {name}: {row["prompt_tokens"]} prompt, '
                      f'{row["completion_tokens"]} generated tokens in {row["seconds"]} s', flush=True)
        finally:
            stop(process)
    if not trace.exists() or trace.stat().st_size <= 64:
        raise RuntimeError(f'{trace}: no rounds were traced')
    print(f'[{arm}] wrote {trace} ({trace.stat().st_size >> 20} MiB)', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--serve', required=True, type=Path, help='the infernix-serve executable')
    ap.add_argument('--artifact', required=True, type=Path, help='the artifact (its first file)')
    ap.add_argument('--out', required=True, type=Path, help='directory for the traces, request lists and logs')
    ap.add_argument('--arms', default='plain,mtp,dflash2', help='arms to run, of plain, mtp, dflash2')
    ap.add_argument('--arm-flags', action='append', default=[],
                    help='NAME="FLAGS": an arm\'s speculative flags instead of the default (repeatable)')
    ap.add_argument('--serve-flags', default='', help='more infernix-serve flags for every arm')
    ap.add_argument('--agent-plan', type=Path, default=None,
                    help='a bench/agentic_ab plan.json: its actors\' opening turns join the task set')
    ap.add_argument('--max-new', type=int, default=2048, help='the most tokens a request generates (default 2048)')
    ap.add_argument('--seed', type=int, default=1000, help='request i samples with seed + i')
    ap.add_argument('--port', type=int, default=8090)
    args = ap.parse_args(argv)
    flags = {name: list(value) for name, value in ARM_FLAGS.items()}
    for item in args.arm_flags:
        name, sep, value = item.partition('=')
        if not sep or name not in flags:
            raise SystemExit(f'--arm-flags {item!r}: expected NAME="FLAGS" with NAME one of {", ".join(flags)}')
        flags[name] = shlex.split(value, posix=os.name != 'nt')
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    if not arms or any(a not in flags for a in arms):
        raise SystemExit(f'--arms takes some of {", ".join(flags)}')
    tasks = fixture_tasks(args.max_new) + (agent_tasks(args.agent_plan, args.max_new) if args.agent_plan else [])
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'{len(tasks)} tasks per arm: ' + ', '.join(sorted({d for _, d, _ in tasks})), flush=True)
    for arm in arms:
        run_arm(args, arm, flags[arm], tasks)
    return 0


if __name__ == '__main__':
    sys.exit(main())
