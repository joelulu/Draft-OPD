"""Draft-only cross-K token selection; target never participates in selection."""
import math
from collections import Counter

METHODS=('vote','mean_probability','contrast16_8','contrast24_16')


def select_tokens(probabilities, top2, original):
    """Per-K probabilities over the union of top2 candidates, not renormalized."""
    candidates=sorted(set(t for ids in top2.values() for t in ids))
    def best(score, allowed=candidates):
        # Prefer unchanged K16 token on exact ties, then ascending token ID.
        return max(allowed,key=lambda t:(score(t),t==original,-t))
    votes=Counter(ids[0] for ids in top2.values())
    result={'vote':best(lambda t:votes[t]),
        'mean_probability':best(lambda t:sum(p[t] for p in probabilities.values())/len(probabilities))}
    for name,strong,weak in [('contrast16_8',16,8),('contrast24_16',24,16)]:
        if strong not in probabilities or weak not in probabilities:
            result[name]=original
        else:
            # alpha=1. Limit to strong distribution's top2 to prevent tiny-probability artifacts.
            result[name]=best(lambda t:2*math.log(max(probabilities[strong][t],1e-30))
                -math.log(max(probabilities[weak][t],1e-30)),top2[strong])
    return result


def summarize_cross(rows, thresholds):
    eligible=[r for r in rows if not r['cross_any_eos']]
    output=[]
    for threshold in thresholds:
        for method in METHODS:
            trials=[r['cross'][str(threshold)][method] for r in eligible]
            n=len(trials);base=sum(r['accepted'] for r in eligible)
            new=sum(t['accepted_repaired'] for t in trials)
            changed=[t for t in trials if t['changed']]
            hits=[t for t,r in zip(trials,eligible) if r['rules'][str(threshold)]['category']=='hit_first_error']
            output.append(dict(threshold=threshold,method=method,blocks=n,eos_excluded=len(rows)-n,
                selected=sum(t['position'] is not None for t in trials),changed=len(changed),
                changed_to_target=sum(t['matches_target_at_position'] for t in changed),
                selected_first_errors=len(hits),first_errors_fixed=sum(t['matches_target_at_position'] for t in hits),
                improved_blocks=sum(t['accepted_repaired']>r['accepted'] for t,r in zip(trials,eligible)),
                harmed_blocks=sum(t['accepted_repaired']<r['accepted'] for t,r in zip(trials,eligible)),
                mean_accepted=base/n if n else None,mean_accepted_repaired=new/n if n else None,
                mean_gain=(new-base)/n if n else None,gain_pct=100*(new-base)/base if base else None,
                oracle_selected_mean=sum(r['rules'][str(threshold)]['accepted_repaired'] for r in eligible)/n if n else None))
    return output
