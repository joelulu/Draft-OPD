import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('workflow_tools',ROOT/'scripts/adaptive_length/workflow_tools.py')
tools=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)


def fake_results(root,arm,keys=('p1','p2')):
    folder=root/'evaluation'/arm
    folder.mkdir(parents=True)
    modes=['fixed_16']
    if arm=='mixed':
        modes += ['classifier_'+context+'_'+objective for context in ('full_kv','window_16') for objective in ('accept','speed')]
    with (folder/'summary_rank000.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['mode','decode_tok_s','mean_accepted','mean_committed','selector_time_fraction'])
        w.writeheader()
        for mode in modes:
            w.writerow(dict(mode=mode,decode_tok_s=200 if arm=='baseline' else 220,mean_accepted=5,
                mean_committed=6,selector_time_fraction=.02 if mode.startswith('classifier') else 0))
    with (folder/'results_rank000.jsonl').open('w') as f:
        for mode in modes:
            for key in keys:
                f.write(json.dumps(dict(mode=mode,prompt_key=key))+'\n')


def test_three_arm_comparison_uses_fixed_trained_baseline(tmp_path):
    fake_results(tmp_path,'baseline')
    fake_results(tmp_path,'mixed')
    tools.summarize(tmp_path)
    with (tmp_path/'final_comparison.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==6
    assert float(rows[0]['speedup_vs_baseline'])==1
    assert all(float(row['speedup_vs_baseline'])==1.1 for row in rows[1:])


def test_summary_rejects_different_prompt_sets(tmp_path):
    fake_results(tmp_path,'baseline')
    fake_results(tmp_path,'mixed',keys=['other'])
    with pytest.raises(ValueError,match='prompt sets'):
        tools.summarize(tmp_path)


def test_live_status_prints_latest_loss_and_eta(tmp_path,capsys):
    folder=tmp_path/'classifiers_ddp'
    folder.mkdir()
    tools.write_json(dict(phase='train',epoch=2,epochs=10,world_size=8,states=100,total_states=1000,
        percent=10,eta_s=3661,updated_at=0,last_train_loss={'full_kv_accept':1.23}),folder/'progress.json')
    tools.status(tmp_path,folder)
    output=capsys.readouterr().out
    assert '01:01:01' in output and '1.230000' in output and 'GPUs 8' in output


def test_workflow_status_shell_needs_no_gpu_and_preserves_files(tmp_path):
    import os
    import subprocess
    import sys
    marker=tmp_path/'preserve.txt'
    marker.write_text('existing experiment')
    result=subprocess.run(['bash',str(ROOT/'scripts/adaptive_length/run_fixed_vs_dynamic.sh'),'status'],
        env={**os.environ,'PYTHON_BIN':sys.executable,'RESULT_ROOT':str(tmp_path)},
        capture_output=True,text=True,check=True)
    assert 'No v2 classifier progress yet' in result.stdout
    assert marker.read_text()=='existing experiment'
    assert not (tmp_path/'workflow.json').exists()
