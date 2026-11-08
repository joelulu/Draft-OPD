"""Official GSM8K main/test, source order, raw question only as model input."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(rows, count):
    selected, seen = [], set()
    for index, row in enumerate(rows):
        question = row.get('question')
        if question is None and 'turns' in row:
            turns = row['turns']
            if not isinstance(turns, list) or len(turns) != 1:
                raise ValueError(f'Expected a single-turn GSM8K prompt at source row {index}')
            question = turns[0]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f'Invalid GSM8K question at source row {index}')
        if question in seen:
            continue
        seen.add(question)
        selected.append(dict(id=f'gsm8k_test_{index}', source_index=index,
            question=question, answer=row.get('answer'),
            prompt=[dict(role='user', content=question)]))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} unique questions, expected {count}')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--local-test-jsonl', help='Optional user-supplied GSM8K test JSONL with question/answer fields')
    args = p.parse_args()
    if args.samples < 1:
        p.error('samples must be positive')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.local_test_jsonl:
        source = str(Path(args.local_test_jsonl).resolve())
        rows = (json.loads(line) for line in Path(source).read_text().splitlines() if line.strip())
        dataset_fingerprint = None
    else:
        from datasets import load_dataset
        rows = load_dataset('openai/gsm8k', 'main', split='test')
        source = 'openai/gsm8k/main/test'
        dataset_fingerprint = getattr(rows, '_fingerprint', None)
    selected = prepare(rows, args.samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected)
    output.write_text(payload)
    output.with_suffix('.source.json').write_text(json.dumps(dict(
        source=source, dataset_fingerprint=dataset_fingerprint, samples=len(selected),
        selection='first unique questions in source order; no shuffle',
        source_indices=[r['source_index'] for r in selected],
        jsonl_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        input='raw question as user message; answer field never included in prompt'), indent=2))
    print(f'GSM8K test prompts: {output}', flush=True)
    for r in selected:
        print(f"  {r['id']}: {r['question'][:100]}", flush=True)


if __name__ == '__main__':
    main()
