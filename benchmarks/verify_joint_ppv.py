"""Independent full-result audit without fitting or transforming a model."""
import argparse
import csv
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from statistics import mean,median

import numpy as np

from .benchmark_joint_ppv import EXACT_KEYS,case_order,source_hashes,validate_inventory


def verify(path,manifest_path):
    path=Path(path);manifest=json.loads(Path(manifest_path).read_text())
    meta=json.loads(path.with_suffix('.csv.json').read_text())
    config=meta['configuration_signature'];expected=manifest['expected']
    with path.open() as f:rows=list(csv.DictReader(f))
    inventory=json.loads(Path(config['inventory']).read_text())
    assert meta['status']=='complete'
    assert meta['source_sha256']==manifest['source_sha256']==source_hashes()
    assert meta['runtime_signature']==manifest['runtime_signature']
    assert meta['source_sha256']['code/minirocket.py']=='d72107e43f65884ed337af5a97bd03ccdefbab9d0f26e1d17f1f9bd15fdfea80'
    for p,h in manifest['auxiliary_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    assert meta['archive_inventory_sha256']==hashlib.sha256(Path(config['inventory']).read_bytes()).hexdigest()
    assert meta['discovered_datasets']==inventory['discovered_datasets']==expected['discovered']
    assert len(meta['skipped_datasets'])==len(inventory['skipped_datasets'])==expected['skipped']
    assert {r['name']:r['reason'] for r in meta['skipped_datasets']}=={r['name']:r['reason'] for r in inventory['skipped_datasets']}
    validate_inventory(meta['source_inventory'],inventory['source_inventory'])
    datasets={r['name']:r for r in inventory['datasets']}
    assert len(datasets)==len(meta['datasets'])==expected['datasets']
    models=[v['name'] for v in meta['models']]
    idx={(r['dataset'],r['model'],int(r['seed'])):r for r in rows}
    assert len(rows)==len(idx)==meta['completed_runs']==meta['expected_runs']==expected['rows']
    assert set(idx)==set(product(datasets,models,config['seeds']))
    for k,v in dict(threads=1,batch_size=256,max_dilations_per_kernel=32,classifier='ridge',
                    transform_repetitions=3,order_seed=20261026,budgets=[2500,5000,10000]).items():
        assert config[k]==v,(k,config[k])
    history={}
    history_path=manifest.get('historical_csv')
    if history_path:
        with Path(history_path).open() as f:
            history={(r['dataset'],r['model'],int(r['seed'])):r for r in csv.DictReader(f)}
    hashes=pairs=historical=0;old_accuracy_differences=[];maximum_noise=0.
    for r in rows:
        budget=int(r['requested_features']);width=budget//84*84;seed=int(r['seed'])
        assert r['implementation'] in ('original','joint_ppv')
        assert r['model']==f"{r['implementation']}_{budget}" and int(r['features'])==width
        assert r['split_sha256']==datasets[r['dataset']]['fingerprint']
        for k in ('n_train','n_test','length'):assert int(r[k])==datasets[r['dataset']][k]
        assert int(r['numba_threads'])==1 and int(r['max_dilations_per_kernel'])==32
        assert r['allocation_policy']=='common_legacy_batch_copy' and r['classifier']=='ridge'
        assert int(r['transform_cache_hits'])>0 and int(r['transform_cache_misses'])==int(r['fit_cache_misses'])==0
        for pool in json.loads(r['threadpools_json']):assert pool['num_threads']==1
        assert int(r['transform_repetitions'])==3
        for split in ('train','test'):
            t=json.loads(r[f'{split}_transform_repetition_seconds'])
            h=json.loads(r[f'{split}_feature_repetition_sha256'])
            inner=json.loads(r[f'{split}_model_call_repetition_seconds'])
            assert len(t)==len(h)==len(inner)==3 and h==[r[f'{split}_feature_sha256']]*3
            assert all(math.isfinite(v) and v>0 for v in t+inner)
            assert all(0<a<b for a,b in zip(inner,t))
            assert math.isclose(mean(t),float(r[f'{split}_transform_seconds']),rel_tol=1e-12)
            assert math.isclose(mean(inner),float(r[f'{split}_model_call_seconds']),rel_tol=1e-12)
            assert math.isclose(mean(t)-mean(inner),float(r[f'{split}_delivery_overhead_seconds']),rel_tol=1e-12,abs_tol=1e-12)
            assert int(r[f'{split}_feature_bytes'])==int(r[f'n_{split}'])*width*4
            hashes+=3;maximum_noise=max(maximum_noise,max(t)/median(t))
        for k in ('fit_seconds','load_seconds','warmup_seconds','classifier_fit_seconds','predict_seconds',
                  'worker_wall_seconds','peak_rss_bytes','peak_physical_footprint_bytes','peak_memory_bytes'):
            assert math.isfinite(float(r[k])) and float(r[k])>0
        assert r['peak_memory_metric']=='macos_peak_physical_footprint'
        assert r['peak_memory_bytes']==r['peak_physical_footprint_bytes']
        assert 0<float(r['pre_pipeline_peak_memory_bytes'])<=float(r['after_fit_peak_memory_bytes'])<=float(r['after_train_peak_memory_bytes'])<=float(r['peak_memory_bytes'])
        assert 0<=float(r['accuracy'])<=1
        assert int(r['parameter_array_bytes'])==int(r['extractor_parameter_array_bytes'])>0
        assert int(r['total_parameter_array_bytes'])==sum(int(r[k]) for k in
            ('extractor_parameter_array_bytes','scaler_parameter_array_bytes','classifier_parameter_array_bytes'))
        assert int(r['pooling_metadata_bytes'])==float(r['pooling_prepare_seconds'])==0
        train=sum(float(r[k]) for k in ('fit_seconds','train_transform_seconds','classifier_fit_seconds'))
        test=sum(float(r[k]) for k in ('test_transform_seconds','predict_seconds'))
        for a,b in ((train,float(r['training_seconds'])),(test,float(r['test_seconds'])),
                    (train+test+float(r['load_seconds']),float(r['pipeline_seconds']))):
            assert math.isclose(a,b,rel_tol=1e-12)
        if r['implementation']=='joint_ppv':
            other=idx[r['dataset'],f'original_{budget}',seed]
            for k in EXACT_KEYS:
                assert (float(r[k])==float(other[k]) if k in ('accuracy','classifier_alpha')
                        else r[k]==other[k]),(r['dataset'],r['model'],seed,k)
            pairs+=1
        old_name='original' if budget==10000 else f'original_{budget}'
        old=history.get((r['dataset'],old_name,seed))
        if old:
            for k in ('split_sha256','train_feature_sha256','test_feature_sha256','features',
                      'extractor_parameters_sha256','extractor_parameter_array_bytes'):
                assert r[k]==old[k],('historical',r['dataset'],r['model'],seed,k)
            if float(old['accuracy'])!=float(r['accuracy']):
                old_accuracy_differences.append(dict(dataset=r['dataset'],model=r['model'],seed=seed,
                                                     historical=float(old['accuracy']),current=float(r['accuracy'])))
            historical+=1
    assert pairs==expected['pairs'] and hashes==expected['repetition_hashes']
    if history_path:assert historical==len(rows)
    for d in datasets:
        identity=int.from_bytes(hashlib.sha256(d.encode()).digest()[:8],'little')
        rng=np.random.default_rng(np.random.SeedSequence([config['order_seed'],identity,2]))
        assert list(dict.fromkeys(int(r['seed']) for r in rows if r['dataset']==d))==[config['seeds'][i] for i in rng.permutation(len(config['seeds']))]
        for seed in config['seeds']:
            group=[r for r in rows if r['dataset']==d and int(r['seed'])==seed]
            order=case_order(d,seed,meta['models'],config['order_seed'])
            assert [r['model'] for r in group]==[v['name'] for v in order]
            assert [int(r['case_order_position']) for r in group]==list(range(len(order)))
    return dict(status='verified',rows=len(rows),datasets=len(datasets),models=len(models),
                exact_feature_parameter_scaler_classifier_prediction_pairs=pairs,
                stable_repetition_hashes=hashes,historical_feature_parameter_matches=historical,
                historical_accuracy_differences=old_accuracy_differences,
                distinct_numeric_worker_pids=len({r['worker_pid'] for r in rows}),
                fresh_workers='One synchronous subprocess per case; numeric PIDs may recycle',
                settings_sources_inventory_order_times_bytes_peaks_and_cache_verified=True,
                maximum_repetition_over_median=maximum_noise,
                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path)
    p.add_argument('--manifest',required=True,type=Path);args=p.parse_args()
    result=verify(args.csv,args.manifest)
    output=args.csv.with_suffix('.verification.json')
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
