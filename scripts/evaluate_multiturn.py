# cspell:words multiturn rollouts
"""Actual model rollout with deterministic offline tool execution and viewport scoring."""
from __future__ import annotations

import argparse
import base64
import copy
import json
import mimetypes
from pathlib import Path
import re
import time

from schematic_model.agent import AGENT_TOOLS, private_endpoint, request_json, validate_agent_arguments
from schematic_model.multiturn_data import SYSTEM, current_context, tool_result, wrap
from schematic_model.training import sha256

NAVIGATION = {'show_label', 'show_region', 'show_full_sheet'}


def user_message(question, candidate, viewer, attach=False):
    text = wrap(question, viewer)
    if not attach:
        return {'role': 'user', 'content': text}
    path = Path(candidate['image'])
    if sha256(path) != candidate['image_sha256']:
        raise ValueError('Frozen rollout image changed')
    image = 'data:' + (mimetypes.guess_type(path)[0] or 'image/png') + ';base64,' + base64.b64encode(path.read_bytes()).decode()
    evidence = {'page': candidate['page'], 'bom': [candidate['part']], 'scope': 'selected native component; not a full electrical review'}
    text = 'Extracted native schematic evidence (not an electrical review):\n' + json.dumps(evidence) + '\n\n' + text
    return {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image}}, {'type': 'text', 'text': text}]}


def final_matches(content, turn):
    if not isinstance(content, str) or not content.strip():
        return False
    if any(marker in content for marker in ('<tool_call>', '<tool_response>', '<|im_start|>', '\nUser:', '\nHuman:')):
        return False
    if turn.get('expected_value') and turn['expected_value'].casefold() not in content.casefold():
        return False
    if turn.get('clarify') and not (re.search(r'\b(which|select|specify|provide)\b', content, re.I)
                                    and re.search(r'\b(part|component|reference|designator)\b', content, re.I)):
        return False
    if turn.get('refusal') and not any(word in content.lower() for word in ('not', 'cannot', "can't", 'unavailable', 'measurement', 'measure')):
        return False
    return True


def rollout(case, complete, max_rounds=4):
    candidate = case['candidate']
    viewer = copy.deepcopy(case['initial_context'])
    messages = [{'role': 'system', 'content': SYSTEM}]
    records = []
    blocked = False
    for index, turn in enumerate(case['turns']):
        record = {'case_id': case['id'], 'turn': index, 'question': turn['question'], 'allow_navigation': turn['allow_navigation'],
                  'passed': False, 'events': [], 'responses': [], 'viewport_invariant': False, 'terminated': False}
        if blocked:
            record['error'] = 'blocked_by_prior_incomplete_turn'
            records.append(record)
            continue
        if turn.get('page_candidate'):
            candidate = turn['page_candidate']
            viewer = current_context(candidate, False)
        if turn.get('clear_focus'):
            viewer['focused_label'] = None
        before = copy.deepcopy(viewer['viewport'])
        record['viewport_before'] = before
        messages.append(user_message(turn['question'], candidate, viewer, attach=index == 0 or bool(turn.get('page_candidate'))))
        final = None
        started = time.monotonic()
        try:
            for _ in range(max_rounds):
                response = complete(copy.deepcopy(messages))
                record['responses'].append(response)
                choice = response['choices'][0]
                message = choice['message']
                calls = message.get('tool_calls') or []
                if not calls:
                    if choice.get('finish_reason') != 'stop' or not isinstance(message.get('content'), str) or not message['content'].strip():
                        raise ValueError('Incomplete final assistant turn')
                    final = message['content']
                    if any(marker in final for marker in ('<tool_call>', '<tool_response>', '<|im_start|>', '\nUser:', '\nHuman:')):
                        raise ValueError('Invented conversation in final answer')
                    messages.append({'role': 'assistant', 'content': final})
                    record['terminated'] = True
                    break
                if choice.get('finish_reason') != 'tool_calls' or len(calls) != 1:
                    raise ValueError('Expected one complete tool call and no invented continuation')
                if message.get('content') and any(marker in message['content'] for marker in ('<tool_response>', '\nUser:', '<|im_start|>')):
                    raise ValueError('Invented conversation after tool call')
                call = calls[0]
                name = call['function']['name']
                args = validate_agent_arguments(name, json.loads(call['function']['arguments']))
                result = tool_result(name, args, candidate)
                messages.append({'role': 'assistant', 'content': message.get('content'), 'tool_calls': [call]})
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)})
                if name == 'show_label' and result['status'] == 'shown':
                    viewer = current_context(candidate)
                elif name == 'show_region':
                    viewer['viewport'] = {'box': args['box'], 'full_sheet': False}
                    viewer['focused_label'] = None
                elif name == 'show_full_sheet':
                    viewer['viewport'] = {'box': [0, 0, 1000, 1000], 'full_sheet': True}
                    viewer['focused_label'] = None
                record['events'].append({'tool': name, 'arguments': args, 'result': result})
            if final is None:
                raise ValueError('Tool round limit reached without final answer')
            names = [event['tool'] for event in record['events']]
            expected = turn['tool']
            actions_match = names == [expected] if expected else not names
            if expected == 'web_search' and actions_match:
                actions_match = candidate['part']['value'].casefold() in record['events'][0]['arguments']['query'].casefold()
            if expected == 'show_label' and actions_match:
                actions_match = record['events'][0]['arguments'] == turn['arguments']
            record['viewport_invariant'] = viewer['viewport'] == before and not any(name in NAVIGATION for name in names)
            record['passed'] = actions_match and final_matches(final, turn) and (turn['allow_navigation'] or record['viewport_invariant'])
            record['answer'] = final
        except Exception as error:
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
            blocked = True
        record['elapsed_seconds'] = time.monotonic() - started
        record['viewport_after'] = copy.deepcopy(viewer['viewport'])
        records.append(record)
    return records


def run(cases_path, output, endpoint, model, max_tokens=256):
    if output.exists():
        raise ValueError('Output exists; preserve prior real-model results')
    endpoint = private_endpoint(endpoint)
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Cases must be nonempty and unique')
    output.mkdir(parents=True)
    all_records = []
    def complete(messages):
        return request_json(endpoint + '/chat/completions', {'model': model, 'messages': messages,
                            'tools': AGENT_TOOLS, 'max_tokens': max_tokens, 'temperature': 0})
    for case in cases:
        records = rollout(case, complete)
        all_records += records
        with (output / 'turns.jsonl').open('a') as stream:
            stream.write(''.join(json.dumps(record) + '\n' for record in records))
        print(json.dumps({'case': case['id'], 'passed_turns': sum(r['passed'] for r in records), 'turns': len(records)}), flush=True)
    non_navigation = [r for r in all_records if not r['allow_navigation']]
    report = {'cases_sha256': sha256(cases_path), 'endpoint': endpoint, 'model': model, 'max_tokens': max_tokens,
              'cases': len(cases), 'user_turns': len(all_records), 'passed_turns': sum(r['passed'] for r in all_records),
              'terminated_turns': sum(r['terminated'] for r in all_records),
              'non_navigation_turns': len(non_navigation), 'viewport_invariant_turns': sum(r['viewport_invariant'] for r in non_navigation),
              'unsolicited_navigation_calls': sum(e['tool'] in NAVIGATION for r in non_navigation for e in r['events']),
              'passed': all(r['passed'] for r in all_records),
              'scope': 'Actual model generated-history rollout; viewer tool state executed offline; web search returns explicit simulated empty results without network traffic. Not electrical answer qualification.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', default='schematic')
    parser.add_argument('--max-tokens', type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.cases, args.out, args.endpoint, args.model, args.max_tokens), indent=2))


if __name__ == '__main__':
    main()
