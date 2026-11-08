"""CPU-only audit of same-state K dependence from saved oracle trials.

Capacity is controlled: 'extra width helps' compares a larger K with a smaller
K that can already hold the larger block's accepted prefix. Excludes clipped
tail states and EOS-adjusted trials. Candidate IDs allow separating proposal
changes from acceptance changes with identical candidate prefixes. These are
descriptive, oracle-trajectory-conditioned statistics, not a causal bug proof.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics

KS = [4, 8, 12, 16, 20, 24, 28, 32]


def analyze_state(record):
    trials = sorted(record['trials'], key=lambda t:t['requested_k'])
    if [t['requested_k'] for t in trials] != KS:
        raise ValueError('Expected all eight nominal K values')
    for t in trials:
        k, a, block = t['actual_k'], t['accepted_raw'], t['block_ids']
        if not (1 <= k <= t['requested_k'] and isinstance(a, int) and 0 <= a <= k):
            raise ValueError('Invalid K/acceptance')
        if len(block) != k+1 or not 1 <= t['emitted'] <= a+1:
            raise ValueError('Invalid block shape/emitted count')
    if len({t['block_ids'][0] for t in trials}) != 1:
        raise ValueError('Trials at supposedly same state have different anchors')
    if any(t['actual_k'] != t['requested_k'] for t in trials):
        return None, [], 'tail_clipped'
    if any(t['emitted'] != t['accepted_raw']+1 for t in trials):
        return None, [], 'eos_adjusted'
    accepted = [t['accepted_raw'] for t in trials]
    maximum = max(accepted)
    min_best_k = next(t['actual_k'] for t in trials if t['accepted_raw'] == maximum)
    capacity_k = next(k for k in KS if k >= maximum) if maximum else None
    at_capacity = next(t['accepted_raw'] for t in trials if t['actual_k'] == capacity_k) if maximum else None
    pairs = []
    for i, lo in enumerate(trials):
        for hi in trials[i+1:]:
            small, large = lo['actual_k'], hi['actual_k']
            a_small, a_large = lo['accepted_raw'], hi['accepted_raw']
            c_small, c_large = lo['block_ids'][1:], hi['block_ids'][1:]
            # If draft proposals were merely truncated, shared candidate IDs
            # should match and a_small would equal min(a_large, small).
            equal_prefix = c_small == c_large[:small]
            first_diff = next((j+1 for j, (x, y) in enumerate(zip(c_small, c_large)) if x != y), None)
            eligible = 0 < a_large <= small
            helps = eligible and a_small < a_large
            pairs.append(dict(small_k=small, large_k=large, small_accept=a_small,
                large_accept=a_large, common_prefix_changed=not equal_prefix,
                first_candidate_difference=first_diff,
                truncation_accept_mismatch=a_small != min(a_large, small),
                large_hurts=a_large < a_small,
                capacity_controlled=eligible,
                extra_width_helps=helps,
                extra_width_accept_gain=a_large-a_small if helps else 0,
                changed_within_large_accepted_prefix=(first_diff is not None and first_diff <= a_large) if eligible else None,
                identical_prefix_accept_mismatch=equal_prefix and a_small != min(a_large, small)))
    row = dict(max_accept=maximum, smallest_k_for_max=min_best_k,
        smallest_grid_k_with_capacity=capacity_k, acceptance_at_capacity_k=at_capacity,
        needs_extra_width=maximum > 0 and min_best_k > capacity_k,
        extra_width=min_best_k-capacity_k if maximum else 0,
        max_accept_loss_at_capacity=maximum-at_capacity if maximum else 0,
        all_accept_equal=len(set(accepted)) == 1,
        any_common_prefix_change=any(p['common_prefix_changed'] for p in pairs),
        any_truncation_accept_mismatch=any(p['truncation_accept_mismatch'] for p in pairs),
        any_large_hurts=any(p['large_hurts'] for p in pairs),
        any_capacity_controlled_help=any(p['extra_width_helps'] for p in pairs),
        any_identical_prefix_accept_mismatch=any(p['identical_prefix_accept_mismatch'] for p in pairs))
    row.update({f'accept_k{k}':a for k, a in zip(KS, accepted)})
    return row, pairs, None


def rate(n, d):
    return n/d if d else None


def aggregate(states, pairs):
    n = len(states)
    positive = [r for r in states if r['max_accept'] > 0]
    eligible = [r for r in pairs if r['capacity_controlled']]
    helps = [r for r in eligible if r['extra_width_helps']]
    identical = [r for r in pairs if not r['common_prefix_changed']]
    return dict(states=n, positive_accept_states=len(positive), pairs=len(pairs),
        state_rates={name:rate(sum(bool(r[name]) for r in states), n) for name in (
            'any_common_prefix_change', 'any_truncation_accept_mismatch', 'any_large_hurts',
            'any_capacity_controlled_help', 'any_identical_prefix_accept_mismatch', 'all_accept_equal')},
        needs_extra_width_count=sum(r['needs_extra_width'] for r in positive),
        needs_extra_width_rate_among_positive=rate(sum(r['needs_extra_width'] for r in positive),len(positive)),
        capacity_controlled_pairs=len(eligible), extra_width_help_pairs=len(helps),
        extra_width_help_pair_rate=rate(len(helps),len(eligible)),
        mean_accept_gain_when_extra_width_helps=statistics.mean(r['extra_width_accept_gain'] for r in helps) if helps else None,
        helped_pairs_with_candidate_change_inside_accepted_prefix=sum(bool(r['changed_within_large_accepted_prefix']) for r in helps),
        identical_common_prefix_pairs=len(identical),
        identical_prefix_accept_mismatch_pairs=sum(r['identical_prefix_accept_mismatch'] for r in identical),
        smallest_k_for_max_histogram=dict(Counter(r['smallest_k_for_max'] for r in states)),
        extra_width_histogram=dict(Counter(r['extra_width'] for r in positive)))


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, help='Existing gsm8k_oracle_TIMESTAMP directory')
    p.add_argument('--output')
    p.add_argument('--examples', type=int, default=8, help='Extra-width examples printed per draft')
    args = p.parse_args()
    root = Path(args.root)
    paths = sorted((root/'schedules').glob('*/*_accept.json'))
    if not paths:
        raise FileNotFoundError(f'No acceptance-oracle schedules under {root}')
    out = Path(args.output) if args.output else root/('k_dependence_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    states, pairs, skipped, example_rows = [], [], defaultdict(Counter), []
    for path in paths:
        arm = path.parent.name
        key = path.name.removesuffix('_accept.json')
        schedule = json.loads(path.read_text())
        if schedule['objective'] != 'accept':
            raise ValueError(f'Unexpected objective: {path}')
        for record in schedule['trials']:
            row, compared, skip = analyze_state(record)
            if skip:
                skipped[arm][skip] += 1
                continue
            identity = dict(arm=arm, prompt_key=key, round=record['round']+1,
                            context_length=record['context_length'])
            states.append(dict(identity, **row))
            pairs.extend(dict(identity, **r) for r in compared)
            if row['needs_extra_width']:
                example_rows.append(dict(identity, **row,
                    block_ids_by_k={str(t['requested_k']):t['block_ids'] for t in record['trials']}))
    write_csv(out/'states.csv', states)
    write_csv(out/'pairs.csv', pairs)
    summary = {}
    prompt_summary = []
    for arm in sorted({path.parent.name for path in paths}):
        arm_states = [r for r in states if r['arm'] == arm]
        arm_pairs = [r for r in pairs if r['arm'] == arm]
        result = aggregate(arm_states, arm_pairs)
        result['skipped'] = dict(skipped[arm])
        summary[arm] = result
        print(f'\n===== {arm} =====')
        print('Analyzed non-tail, non-EOS states:', result['states'], 'Skipped:',result['skipped'])
        for name, value in result['state_rates'].items():
            print(f'{name}: {value:.2%}' if value is not None else f'{name}: N/A')
        ratio = result['needs_extra_width_rate_among_positive']
        print('Positive-accept states requiring more than capacity-sized grid K:',
              f'{result["needs_extra_width_count"]}/{result["positive_accept_states"]}',
              f'({ratio:.2%})' if ratio is not None else 'N/A')
        ratio = result['extra_width_help_pair_rate']
        print('Larger K helps when smaller K already has enough capacity:',
              f'{result["extra_width_help_pairs"]}/{result["capacity_controlled_pairs"]}',
              f'({ratio:.2%})' if ratio is not None else 'N/A')
        print('Same candidate prefix but inconsistent truncated acceptance:',
              f'{result["identical_prefix_accept_mismatch_pairs"]}/{result["identical_common_prefix_pairs"]}')
        print('Smallest K achieving maximum acceptance:', result['smallest_k_for_max_histogram'])
        for key in sorted({r['prompt_key'] for r in arm_states}):
            group = [r for r in arm_states if r['prompt_key']==key]
            pos = [r for r in group if r['max_accept']>0]
            prompt_summary.append(dict(arm=arm, prompt_key=key, states=len(group),
                positive_accept_states=len(pos), needs_extra_width=sum(r['needs_extra_width'] for r in pos),
                needs_extra_width_rate=rate(sum(r['needs_extra_width'] for r in pos),len(pos))))
        examples = sorted((r for r in example_rows if r['arm']==arm),
                          key=lambda r:(-r['extra_width'], -r['max_accept_loss_at_capacity']))[:args.examples]
        for r in examples:
            print(f"\n{r['prompt_key'][:12]} round={r['round']} context={r['context_length']} "
                  f"max_accept={r['max_accept']} needs K={r['smallest_k_for_max']}; "
                  f"capacity-only K={r['smallest_grid_k_with_capacity']} accepts {r['acceptance_at_capacity_k']}")
            print('  '+' '.join(f"{r[f'accept_k{k}']}({k})" for k in KS))
    write_csv(out/'per_prompt.csv', prompt_summary)
    (out/'examples.json').write_text(json.dumps(example_rows, indent=2))
    summary['scope'] = ('Same states along each draft own acceptance-oracle trajectory; state and pair denominators differ. '
        'Raw acceptance excludes anchor; clipped tails and EOS-adjusted states excluded. '
        'Smallest grid K with capacity is ceil(acceptance/4)*4, not exact K=acceptance unless in grid. '
        'Changed proposals do not identify masking vs position/numerical causes. Identical-prefix acceptance mismatch '
        'flags target verification/cache/numerical sensitivity for further replay, not a proven bug. '
        'No inference for untested K=1,2,3,5,6,etc. These statistics do not prove large K is always needed.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('\nResults:', out)


if __name__ == '__main__':
    main()
