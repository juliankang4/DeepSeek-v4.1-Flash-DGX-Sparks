#!/usr/bin/env python3
"""Parity of this stack's V4.1 encoder against DeepSeek's encoding/encoding.py.

Loads the checkpoint's reference encoder (the only prompt-format spec DeepSeek
ships) and checks thinking-off, tools, and reasoning-effort renders. When SGLang
is importable (in the serving image), also diffs encoding_dsv41.encode_messages
(prompt and image order) after the publisher effort remap and the 1017 rules
(adapter/encoding_1017.py), runs the checkpoint's golden fixtures
(encoding/tests/test_input_N.json -> test_output_N.txt) through it, and checks
that SGLang-only renders (developer role) did not change. --stock skips the
1017 rules to show what they change.

    python3 tests/test_chat_encoding.py
    python3 tests/test_chat_encoding.py --src ~/NewModels/DeepSeek-V4.1-Flash
"""
import argparse
import difflib
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))

from encoding_compat import PUBLISHER_REASONING_EFFORT, install_encoder  # noqa: E402
from encoding_1017 import install as install_1017  # noqa: E402

DEFAULT_SRC = os.environ.get(
    'MODEL_PATH', os.path.expanduser('~/NewModels/DeepSeek-V4.1-Flash'))


def call(cid, name, arguments, **extra):
    return {'type': 'function', 'id': cid, **extra,
            'function': {'name': name, 'arguments': arguments}}


def text_image(text, url):
    return [{'type': 'text', 'text': text}, {'type': 'image_url', 'image_url': {'url': url}}]


AGENT_LOOP = [{'role': 'user', 'content': 'q1'},
              {'role': 'assistant', 'content': 'a1', 'reasoning_content': 'r1'},
              {'role': 'user', 'content': 'q2'},
              {'role': 'assistant', 'content': '', 'reasoning_content': 'r2',
               'tool_calls': [call('c1', 'lookup', '{"q": "x"}')]},
              {'role': 'tool', 'tool_call_id': 'c1', 'content': 'v1'},
              {'role': 'assistant', 'content': '', 'reasoning_content': 'r3',
               'tool_calls': [call('c2', 'lookup', {'q': 'y'})]},
              {'role': 'tool', 'tool_call_id': 'c2', 'content': 'v2'}]

CASES = [
    ('plain user, thinking',
     dict(messages=[{'role': 'user', 'content': 'How many r in strawberry?'}],
          thinking_mode='thinking')),
    ('plain user, chat mode',
     dict(messages=[{'role': 'user', 'content': 'How many r in strawberry?'}],
          thinking_mode='chat')),
    ('system + user, chat mode',
     dict(messages=[{'role': 'system', 'content': 'You are terse.'},
                    {'role': 'user', 'content': 'Hi'}],
          thinking_mode='chat')),
    ('reasoning effort low',
     dict(messages=[{'role': 'user', 'content': 'Hi'}],
          thinking_mode='thinking', reasoning_effort='low')),
    ('reasoning effort high',
     dict(messages=[{'role': 'user', 'content': 'Hi'}],
          thinking_mode='thinking', reasoning_effort='high')),
    ('reasoning effort max',
     dict(messages=[{'role': 'user', 'content': 'Hi'}],
          thinking_mode='thinking', reasoning_effort='max')),
    ('reasoning effort int',
     dict(messages=[{'role': 'user', 'content': 'Hi'}],
          thinking_mode='thinking', reasoning_effort=42)),
    ('multi turn, chat mode',
     dict(messages=[{'role': 'user', 'content': 'First question'},
                    {'role': 'assistant', 'content': 'First answer'},
                    {'role': 'user', 'content': 'Second question'}],
          thinking_mode='chat')),
    ('consecutive user messages merge',
     dict(messages=[{'role': 'user', 'content': 'part one'},
                    {'role': 'user', 'content': 'part two'}],
          thinking_mode='thinking')),
    ('tools in system message, thinking',
     dict(messages=[{'role': 'system', 'content': 'You may use tools.'},
                    {'role': 'user', 'content': 'Weather in Paris?'}],
          tools=[{'type': 'function', 'function': {
              'name': 'get_weather', 'description': 'Get weather',
              'parameters': {'type': 'object',
                             'properties': {'city': {'type': 'string'}},
                             'required': ['city']}}}],
          thinking_mode='thinking')),
    ('chat mode with tools',
     dict(messages=[{'role': 'system', 'content': 'sys'},
                    {'role': 'user', 'content': 'Weather?'}],
          thinking_mode='chat',
          tools=[{'type': 'function',
                  'function': {'name': 'w', 'parameters': {'type': 'object'}}}])),
    ('assistant tool call + tool result',
     dict(messages=[{'role': 'system', 'content': 'You may use tools.'},
                    {'role': 'user', 'content': 'Weather in Paris?'},
                    {'role': 'assistant', 'content': '',
                     'reasoning_content': 'need the tool',
                     'tool_calls': [{'type': 'function', 'id': 'c1', 'function': {
                         'name': 'get_weather',
                         'arguments': {'city': 'Paris', 'days': 3, 'metric': True}}}]},
                    {'role': 'tool', 'tool_call_id': 'c1', 'content': '18C, cloudy'},
                    {'role': 'user', 'content': 'And tomorrow?'}],
          tools=[{'type': 'function', 'function': {
              'name': 'get_weather', 'description': 'Get weather',
              'parameters': {'type': 'object',
                             'properties': {'city': {'type': 'string'}}}}}],
          thinking_mode='thinking')),
    ('two tool results merge into one user turn',
     dict(messages=[{'role': 'user', 'content': 'Compare Paris and Rome'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [
                        {'type': 'function', 'id': 'a',
                         'function': {'name': 'w', 'arguments': {'city': 'Paris'}}},
                        {'type': 'function', 'id': 'b',
                         'function': {'name': 'w', 'arguments': {'city': 'Rome'}}}]},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': '18C'},
                    {'role': 'tool', 'tool_call_id': 'b', 'content': '24C'}],
          thinking_mode='thinking')),
    ('out-of-order tool results',
     dict(messages=[{'role': 'user', 'content': 'both'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [
                        {'type': 'function', 'id': 'a',
                         'function': {'name': 'w', 'arguments': {'c': 'P'}}},
                        {'type': 'function', 'id': 'b',
                         'function': {'name': 'w', 'arguments': {'c': 'R'}}}]},
                    {'role': 'tool', 'tool_call_id': 'b', 'content': 'second'},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': 'first'}],
          thinking_mode='thinking')),
    ('response_format schema',
     dict(messages=[{'role': 'system', 'content': 'sys'},
                    {'role': 'user', 'content': 'Hi'}],
          response_format={'type': 'json_object',
                           'schema': {'type': 'object',
                                      'properties': {'a': {'type': 'string'}}}},
          thinking_mode='thinking')),
    # 1017 rules (adapter/encoding_1017.py)
    ('agent loop without tools keeps reasoning across tool results',
     dict(messages=AGENT_LOOP, thinking_mode='thinking')),
    ('new user turn after tool results drops their reasoning',
     dict(messages=AGENT_LOOP + [{'role': 'user', 'content': 'q3'}], thinking_mode='thinking')),
    ('tool result and user text form one turn',
     dict(messages=AGENT_LOOP[:5] + [{'role': 'user', 'content': 'and now?'}],
          thinking_mode='thinking')),
    ('namespaced tools and calls',
     dict(messages=[{'role': 'system', 'content': 'sys'},
                    {'role': 'user', 'content': 'find it'},
                    {'role': 'assistant', 'content': '', 'reasoning_content': 'both',
                     'tool_calls': [call('a', 'read', {'path': 'x'}, namespace='files'),
                                    call('b', 'search::read', {'q': 'x'})]},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': 'file'},
                    {'role': 'tool', 'tool_call_id': 'b', 'content': 'hit'}],
          tools=[{'type': 'function',
                  'namespace': {'name': 'files', 'description': 'File tools.'},
                  'function': {'name': 'read', 'description': 'Read a file',
                               'parameters': {'type': 'object'}}},
                 {'type': 'function', 'function': {'name': 'read', 'namespace': 'search',
                                                   'parameters': {'type': 'object'}}},
                 {'type': 'function', 'function': {'name': 'read',
                                                   'parameters': {'type': 'object'}}}],
          thinking_mode='thinking')),
    ('tool call arguments null or double-encoded',
     dict(messages=[{'role': 'user', 'content': 'go'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [
                        call('a', 'w', None), call('b', 'w', json.dumps(json.dumps({'c': 'P'})))]},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': 'x'},
                    {'role': 'tool', 'tool_call_id': 'b', 'content': 'y'}],
          thinking_mode='chat')),
    ('tool content list with a video part',
     dict(messages=[{'role': 'user', 'content': 'clip?'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [call('a', 'clip', {})]},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': [
                        {'type': 'text', 'text': 'clip:'},
                        {'type': 'video_url', 'video_url': {'url': 'v.mp4'}}]}],
          thinking_mode='chat')),
    ('out-of-order tool results carrying images',
     dict(messages=[{'role': 'user', 'content': text_image('look', 'history.png')},
                    {'role': 'assistant', 'content': '', 'tool_calls': [
                        call('a', 'snap', {'n': 1}), call('b', 'snap', {'n': 2})]},
                    {'role': 'tool', 'tool_call_id': 'b', 'content': text_image('b', 'b.png')},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': text_image('a', 'a.png')}],
          thinking_mode='thinking')),
]

# SGLang-only renders (the reference has no developer role): must equal the stock encoder's.
STOCK_CASES = [
    ('developer role with tools, thinking',
     dict(messages=[{'role': 'developer', 'content': 'Use tools.'},
                    {'role': 'user', 'content': 'Weather?'}],
          tools=[{'type': 'function', 'function': {'name': 'w', 'parameters': {'type': 'object'}}}],
          thinking_mode='thinking')),
    ('developer after tool results, thinking',
     dict(messages=AGENT_LOOP[2:5] + [{'role': 'developer', 'content': 'Be brief.'}],
          thinking_mode='thinking')),
]


def load_reference(src):
    path = os.path.join(src, 'encoding', 'encoding.py')
    spec = importlib.util.spec_from_file_location('dsv41_encoding', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare_messages(case):
    msgs = json.loads(json.dumps(case['messages']))
    if case.get('tools'):
        msgs[0]['tools'] = case['tools']
    if case.get('response_format'):
        msgs[0]['response_format'] = case['response_format']
    return msgs


def render(enc, case):
    """(prompt, image urls in prompt order) from either encoder."""
    kwargs = {k: case[k] for k in ('reasoning_effort', 'context') if case.get(k) is not None}
    prompt, media = enc.encode_messages(
        prepare_messages(case),
        thinking_mode=case['thinking_mode'],
        return_multi_modal_data=True,
        **kwargs,
    )
    return prompt, [image.get('url') for image in media['images']]


def golden_cases(enc, src):
    """The checkpoint's encoding/tests fixtures as (name, case, expected prompt)."""
    for path in sorted(glob.glob(os.path.join(src, 'encoding', 'tests', 'test_input_*.json'))):
        case = enc.load_cases(path)[0]
        case['thinking_mode'] = case['thinking_mode'] or 'chat'
        with open(path.replace('test_input_', 'test_output_')[:-len('.json')] + '.txt') as f:
            expected = f.read()
        yield f'golden {os.path.basename(path)}', case, expected


def try_sglang_encoder(stock):
    """encoding_dsv41 with the adapters installed, plus its stock STOCK_CASES renders."""
    try:
        from sglang.srt.entrypoints.openai import encoding_dsv41
    except ImportError:
        return None, None
    stock_renders = {name: render(encoding_dsv41, case) for name, case in STOCK_CASES}
    install_encoder(encoding_dsv41)
    if not stock and not install_1017(encoding_dsv41):
        raise SystemExit('encoding_1017 did not install: encoding_dsv41 drifted')
    return encoding_dsv41, stock_renders


def check_reference_shape(name, case, prompt, enc):
    """Structural checks that do not require SGLang: thinking markers, effort, tools."""
    errors = []
    if case['thinking_mode'] == 'thinking':
        if '<think>' not in prompt:
            errors.append('thinking mode missing <think> generation header')
        effort = case.get('reasoning_effort', 'high')
        if isinstance(effort, str):
            budget = enc.REASONING_EFFORT_MAPPINGS[effort]
        else:
            budget = effort
        needle = f'Reasoning Effort: {budget} '
        if needle not in prompt:
            errors.append(f'missing {needle!r}')
    else:
        if 'Reasoning Effort:' in prompt:
            errors.append('chat mode should not render a reasoning-effort prefix')
        # Generation header in chat mode is </think>, not <think>
        if prompt.endswith('<think>'):
            errors.append('chat mode must not open <think> for generation')
        if '<｜Assistant｜></think>' not in prompt and not prompt.endswith('</think>'):
            # last turn should close thinking so the model answers in the clear
            if case['messages'][-1]['role'] == 'user':
                errors.append('chat mode user turn should end with </think> header')
    if case.get('tools'):
        if '## Tools' not in prompt:
            errors.append('tools missing ## Tools section')
        if '｜DSML｜ calls' not in prompt:
            errors.append('tools missing DSML call wrapper')
        for tool in case['tools']:
            fn = tool.get('function') or tool
            if fn.get('name') and fn['name'] not in prompt:
                errors.append(f'tool schema missing name {fn["name"]!r}')
    if case.get('response_format') and '## Response Format:' not in prompt:
        errors.append('response_format missing schema block')
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('--stock', action='store_true', help='compare without the 1017 rules')
    args = ap.parse_args()

    enc_path = os.path.join(args.src, 'encoding', 'encoding.py')
    if not os.path.isfile(enc_path):
        print(f'skip: no reference encoder at {enc_path}', file=sys.stderr)
        sys.exit(0)

    ref = load_reference(args.src)
    assert ref.REASONING_EFFORT_MAPPINGS['low'] == 50
    assert ref.REASONING_EFFORT_MAPPINGS['high'] == 75
    assert ref.REASONING_EFFORT_MAPPINGS['max'] == 100
    assert PUBLISHER_REASONING_EFFORT['low'] == 50
    assert PUBLISHER_REASONING_EFFORT['high'] == 75

    sgl, stock_renders = try_sglang_encoder(args.stock)
    if sgl is None:
        print('sglang encoding_dsv41 not importable; reference-shape checks only')
    else:
        print('comparing sglang encoding_dsv41 to publisher encoding.py '
              f'(effort map {dict(sgl.REASONING_EFFORT_MAPPINGS)}, '
              f'1017 rules {"off" if args.stock else "on"})')

    rows = [(name, case, None) for name, case in CASES]
    rows += list(golden_cases(ref, args.src))
    if sgl is not None:
        rows += [(name, case, stock_renders[name]) for name, case in STOCK_CASES]

    def diff(want, got, a, b):
        lines = difflib.unified_diff(want.splitlines(), got.splitlines(), a, b, lineterm='', n=1)
        return f'{b} prompt differs from {a}:\n' + '\n'.join(lines)[:2000]

    failures = 0
    for name, case, expected in rows:
        errors = []
        if isinstance(expected, str):  # golden fixture
            want = render(ref, case)
            if want[0] != expected:
                errors.append(diff(expected, want[0], 'golden', 'publisher'))
        elif expected is None:
            want = render(ref, case)
            errors = check_reference_shape(name, case, want[0], ref)
        else:  # SGLang-only case: stock render
            want = expected
        label = 'reference' if expected is None or isinstance(expected, str) else 'stock'
        if sgl is not None:
            try:
                got = render(sgl, case)
            except Exception as exc:
                errors.append(f'sglang encode raised {type(exc).__name__}: {exc}')
            else:
                if got[0] != want[0]:
                    errors.append(diff(want[0], got[0], label, 'sglang'))
                if got[1] != want[1]:
                    errors.append(f'sglang image order {got[1]} != {label} {want[1]}')
        if errors:
            failures += 1
            print(f'[FAIL] {name}')
            for err in errors:
                for line in err.splitlines():
                    print('       ' + line)
        else:
            print(f'[OK]   {name}')
            if args.verbose:
                print('       ' + want[0].replace('\n', '\\n')[:220])

    print(f'\n{len(rows) - failures}/{len(rows)} cases passed')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
