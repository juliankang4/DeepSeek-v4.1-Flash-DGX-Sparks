"""DeepSeek-V4.1-Flash-1017 prompt rules on SGLang's V4.1 encoder.

1017 (deepseek-ai/DeepSeek-V4.1-Flash-1017 @1bae0e10, ``encoding/encoding.py``) changed the
reference encoder after SGLang e00930c5 ported ``encoding_dsv41``. This replaces only the
stock functions that render differently; SGLang's own additions (``developer`` role, ``xhigh``,
ValueError on bad input) stay.

* ``find_last_user_index``: a user message whose blocks hold only tool results does not start
  a turn. With thinking on and no tool definitions, the reasoning before those results is
  kept and the generation header stays ``<think>``.
* ``tools_from_openai_format`` / ``tool_calls_from_openai_format``: a ``namespace`` on the tool,
  function or call renders the name as ``namespace::name``; a namespace dict's description
  prefixes the function description. Stock ``render_message`` prints the call's ``name``, so
  the qualified name goes there.
* ``encode_arguments_to_dsml``: double-encoded JSON arguments are decoded; anything else that
  is not an object renders as one ``arguments`` parameter.
* ``encode_messages`` / ``merge_tool_messages``: merge and sort tool results before collecting
  images, so image order follows the placeholders, and a tool message's content list stays a
  list (non-text parts render as ``[Unsupported <type>]`` instead of vanishing). Stock
  ``_encode_messages_text`` merges and sorts again, which is a no-op on that output.

Through Chat Completions, pydantic drops ``namespace`` fields and serving_chat rejects
non-object argument strings first, so the live changes are the tool-result turn, image order,
tool content lists, ``arguments: null`` and ``::`` name validation.

Drift guard: sha256[:16] of every stock function replaced or relied on. On any mismatch the
stock encoder is left in place, loudly; ``tests/test_chat_encoding.py`` then fails.
"""
import copy
import hashlib
import inspect
import json

# SGLang v0.5.21 e00930c5, sglang/srt/entrypoints/openai/encoding_dsv41.py
_EXPECTED = {
    # replaced
    'find_last_user_index': '5b01983d829ecafe',
    'merge_tool_messages': '1a3ba1a87d0599f0',
    'encode_messages': '19c15953ff9c9954',
    'tools_from_openai_format': '3fd4045411ba255f',
    'tool_calls_from_openai_format': '4ff997773218e7bc',
    'encode_arguments_to_dsml': '67ec53e72852e02d',
    # call the replacements through module globals, or are called by them
    '_encode_messages_text': '6d66aa51ce40f5e5',
    'render_message': '90b1b73255363685',
    '_drop_thinking_messages': 'de723e3279f3e447',
    'attach_task_to_last_user_message': '387a3d971d565d3d',
    'sort_tool_results_by_call_order': '2c99f7bf778cd332',
    'process_image_messages': 'be66f08506677ed0',
    '_process_image_blocks': 'fc92eeafe871cd73',
    '_validate_no_image_sp_tokens': '5985d7433c617df9',
}

_enc = None  # sglang.srt.entrypoints.openai.encoding_dsv41, set by install()


def _split_tool_name(name, namespace=None):
    """Split ``namespace::name`` and check it against an explicit namespace."""
    prefix, separator, bare_name = name.partition('::')
    if separator:
        if namespace not in (None, prefix):
            raise ValueError(f'Conflicting tool namespaces: {namespace} != {prefix}')
        namespace, name = prefix, bare_name
    if '::' in name:
        raise ValueError(f"Tool name must not contain '::': {name}")
    if namespace is not None and '::' in namespace:
        raise ValueError(f"Tool namespace must not contain '::': {namespace}")
    return namespace, name


def _tool_name_for_encoding(tool):
    namespace = tool.get('namespace')
    if isinstance(namespace, dict):
        namespace = namespace['name']
    namespace, name = _split_tool_name(tool['name'], namespace)
    return name if namespace is None else f'{namespace}::{name}'


def tools_from_openai_format(tools):
    functions = []
    for tool in tools:
        function = dict(tool['function'])
        if tool.get('namespace') is not None:
            function['namespace'] = tool['namespace']
        function['name'] = _tool_name_for_encoding(function)
        namespace = function.pop('namespace', None)
        if isinstance(namespace, dict) and namespace.get('description'):
            function['description'] = (
                namespace['description'] + '\n' + (function.get('description') or ''))
        functions.append(function)
    return functions


def tool_calls_from_openai_format(tool_calls):
    calls = []
    for tool_call in tool_calls:
        function = tool_call['function']
        namespace = tool_call.get('namespace') or function.get('namespace')
        calls.append({
            'name': _tool_name_for_encoding({'name': function['name'], 'namespace': namespace}),
            'arguments': function['arguments'],
        })
    return calls


def encode_arguments_to_dsml(tool_call):
    p_dsml_template = (
        '<{dsml_token}{tool_parameter_tag_name} name="{key}" string="{is_str}">'
        '{value}</{dsml_token}{tool_parameter_tag_name}>'
    )
    arguments = tool_call['arguments']
    if not isinstance(arguments, dict):
        # Tolerate JSON strings, including double-encoded ones.
        for _ in range(2):
            if not isinstance(arguments, str):
                break
            try:
                arguments = json.loads(arguments)
            except Exception:
                break
        if not isinstance(arguments, dict):
            arguments = {'arguments': tool_call['arguments']}
    return '\n'.join(
        p_dsml_template.format(
            dsml_token=_enc.dsml_token,
            tool_parameter_tag_name=_enc.tool_parameter_tag_name,
            key=k,
            is_str='true' if isinstance(v, str) else 'false',
            value=v if isinstance(v, str) else _enc.to_json(v),
        )
        for k, v in arguments.items())


def find_last_user_index(messages):
    """Stock (developer and mid-conversation system messages count), plus 1017's skip of
    user messages that hold only tool results."""
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get('role')
        if role in ('user', 'developer') or (role == 'system' and idx > 0):
            content_blocks = messages[idx].get('content_blocks')
            if content_blocks and not any(
                    block.get('type') in ('text', 'image', 'image_url')
                    for block in content_blocks):
                continue
            return idx
    return -1


def merge_tool_messages(messages):
    merged = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get('role')
        if role == 'tool':
            content = msg.get('content', '')
            if msg.get('content_blocks') and not isinstance(msg.get('content'), str):
                content = msg['content_blocks']
            tool_block = {
                'type': 'tool_result',
                'tool_use_id': msg.get('tool_call_id', ''),
                'content': content,
            }
            if merged and merged[-1].get('role') == 'user' and 'content_blocks' in merged[-1]:
                merged[-1]['content_blocks'].append(tool_block)
            else:
                merged.append({'role': 'user', 'content_blocks': [tool_block]})
        elif role == 'user':
            content_blocks = msg.get('content_blocks')
            if content_blocks is None:
                content = msg.get('content', '')
                content_blocks = (content if isinstance(content, list)
                                  else [{'type': 'text', 'text': content}])
            if (merged and merged[-1].get('role') == 'user'
                    and 'content_blocks' in merged[-1] and merged[-1].get('task') is None):
                merged[-1]['content_blocks'].extend(content_blocks)
            else:
                msg['content_blocks'] = content_blocks
                merged.append(msg)
        else:
            merged.append(msg)
    return merged


def encode_messages(messages, thinking_mode, context=None, drop_thinking=True,
                    add_default_bos_token=True, reasoning_effort=None,
                    return_multi_modal_data=False):
    # Validate before merging can discard message-level textual fields.
    for msg in (context or []) + messages:
        _enc._validate_no_image_sp_tokens(msg)
    # Collect images only after tool results are in their final prompt order.
    context = _enc.merge_tool_messages(context or [])
    messages = _enc.merge_tool_messages(messages)
    messages = _enc.sort_tool_results_by_call_order(context + messages)[len(context):]
    processed_context, _ = _enc.process_image_messages(context) if context else ([], [])
    processed_messages, images = _enc.process_image_messages(messages)
    prompt = _enc._encode_messages_text(
        processed_messages,
        thinking_mode=thinking_mode,
        context=processed_context if processed_context else None,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=reasoning_effort,
    )
    if return_multi_modal_data:
        return prompt, {'images': images}
    return prompt


_REPLACEMENTS = (find_last_user_index, merge_tool_messages, encode_messages,
                 tools_from_openai_format, tool_calls_from_openai_format, encode_arguments_to_dsml)


def drifted(module):
    """Names whose stock source differs from the audited engine."""
    bad = []
    for name, want in _EXPECTED.items():
        obj = getattr(module, name, None)
        got = hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16] if obj else None
        if got != want:
            bad.append(f'{name} {got} != {want}')
    return bad


def install(module):
    """Replace the stock functions on encoding_dsv41; return False if its source drifted."""
    global _enc
    if _enc is module:
        return True
    bad = drifted(module)
    if bad:
        print('DSV41 encoding_1017: encoding_dsv41 drifted, stock encoder kept: '
              + '; '.join(bad), flush=True)
        return False
    _enc = module
    for fn in _REPLACEMENTS:
        setattr(module, fn.__name__, fn)
    return True
