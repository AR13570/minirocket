"""Budget-matched archive summaries with independent raw/CSV reconciliation."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean,median

from . import analyze_results as native
from .joint_ppv_statistics import holm_adjust,ratio_summary,sign_flip_pvalue


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-12),(a,b)


def save_csv(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with path.open() as f:saved=list(csv.DictReader(f))
    assert len(saved)==len(rows)
    for a,b in zip(saved,rows):
        for k,v in b.items():
            if v is None:assert a[k]==''
            elif isinstance(v,(int,float)):close(a[k],v)
            else:assert a[k]==str(v)


def analyze(path,dest):
    verification=json.loads(path.with_suffix('.verification.json').read_text())
    assert verification['raw_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert not dest.exists(),'Use a new output directory; never overwrite a partial analysis'
    dest.mkdir(parents=True)
    with path.open() as f:raw=list(csv.DictReader(f))
    parsed=native.load_rows(path);meta=json.loads(path.with_suffix('.csv.json').read_text())
    groups=defaultdict(list)
    for row in raw:groups[row['dataset'],row['model']].append(row)
    numeric=list(dict.fromkeys([*native.METRICS,*[k for k in native.OPTIONAL_METRICS if k in raw[0]],
        'train_model_call_seconds','test_model_call_seconds','train_delivery_overhead_seconds',
        'test_delivery_overhead_seconds','scaler_parameter_array_bytes','classifier_parameter_array_bytes']))
    means={k:{m:mean(float(r[m]) for r in rows) for m in numeric} for k,rows in groups.items()}
    datasets=sorted({r['dataset'] for r in raw});budgets=meta['configuration_signature']['budgets']
    audit={};overview=[];comparisons=[];per_dataset=[]
    for budget in budgets:
        subset=[r for r in parsed if r['requested_features']==str(budget)]
        baseline=f'original_{budget}';candidate=f'joint_ppv_{budget}'
        assert {r['model'] for r in subset}=={baseline,candidate}
        for label,group_by in [('all',['classifier']),('groups',['classifier','length_group','train_size_group'])]:
            report=native.analyze(subset,baseline=baseline,group_by=group_by)
            expected_pairs=len(datasets)*len(meta['configuration_signature']['seeds'])
            assert report['coverage']==dict(input_runs=expected_pairs*2,reference_runs=0,
                                            matched_seed_comparisons=expected_pairs,unmatched_candidate_runs=0)
            assert len(report['dataset_means'])==2*len(datasets) and len(report['per_dataset'])==len(datasets)
            expected_groups={native.group_label(rows[0],group_by) for key,rows in groups.items() if key[1]==baseline}
            assert len(report['models'])==2*len(expected_groups) and len(report['comparisons'])==len(expected_groups)
            for r in report['dataset_means']:
                assert r['seeds']==len(meta['configuration_signature']['seeds'])
                for metric in report['metrics']:close(r[metric],means[r['dataset'],r['model']][metric])
            for r in report['models']:
                keys=[key for key,rows in groups.items() if key[1]==r['model'] and native.group_label(rows[0],group_by)==r['group']]
                assert r['datasets']==len(keys)
                close(r['accuracy_percent'],100*mean(means[k]['accuracy'] for k in keys))
                for metric in report['metrics'][1:]:close(r[metric],mean(means[k][metric] for k in keys))
            for r in report['per_dataset']:
                a,b=means[r['dataset'],candidate],means[r['dataset'],baseline]
                close(r['accuracy_gain_pp'],100*(a['accuracy']-b['accuracy']))
                for metric in report['metrics'][1:]:
                    if b[metric]>0:close(r[native.ratio_name(metric)],a[metric]/b[metric])
                    else:assert r[native.ratio_name(metric)] is None
            for r in report['comparisons']:
                values=[p for p in report['per_dataset'] if p['group']==r['group']]
                assert r['datasets']==len(values)
                close(r['accuracy_gain_pp'],mean(p['accuracy_gain_pp'] for p in values))
                for metric in report['metrics'][1:]:
                    key=native.ratio_name(metric);values2=[p[key] for p in values if p[key] is not None]
                    if values2:close(r[key],mean(values2))
                    else:assert r[key] is None
            folder=dest/str(budget)/label;folder.mkdir(parents=True)
            for table in ('models','comparisons','dataset_means','per_dataset'):save_csv(folder/(table+'.csv'),report[table])
            with (folder/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
            audit[f'{budget}/{label}']=dict(coverage=report['coverage'],models=len(report['models']),
                dataset_means=len(report['dataset_means']),paired_dataset_rows=len(report['per_dataset']),
                raw_means_and_saved_csv_values_verified=True)
        for model in (baseline,candidate):
            rr=[r for r in raw if r['model']==model];worst=max(rr,key=lambda r:float(r['peak_memory_bytes']))
            row=dict(model=model,actual_features=budget//84*84,datasets=len(datasets),
                     accuracy_percent=100*mean(means[d,model]['accuracy'] for d in datasets))
            row.update({m:mean(means[d,model][m] for d in datasets) for m in numeric if m!='accuracy'})
            row.update(mean_physical_peak_mib=row['peak_memory_bytes']/2**20,
                       mean_rss_peak_mib=row['peak_rss_bytes']/2**20,
                       worst_physical_peak_mib=float(worst['peak_memory_bytes'])/2**20,
                       worst_peak_dataset=worst['dataset'],worst_peak_seed=int(worst['seed']))
            for split in ('train','test'):
                row[f'mean_{split}_examples_per_second']=mean(
                    int(groups[d,model][0][f'n_{split}'])/means[d,model][f'{split}_transform_seconds'] for d in datasets)
                row[f'mean_{split}_nanoseconds_per_input_element']=mean(
                    means[d,model][f'{split}_transform_seconds']*1e9/
                    (int(groups[d,model][0][f'n_{split}'])*int(groups[d,model][0]['length'])) for d in datasets)
                row[f'median_sensitivity_{split}_seconds']=mean(median(json.loads(r[f'{split}_transform_repetition_seconds'])) for r in rr)
            overview.append(row)
        gains=[100*(means[d,candidate]['accuracy']-means[d,baseline]['accuracy']) for d in datasets]
        item=dict(model=candidate,baseline=baseline,actual_features=budget//84*84,datasets=len(datasets),
                  accuracy_gain_pp=mean(gains),gain_ci95_pp=native.bootstrap_interval(gains,10000),
                  wins=sum(g>1e-6 for g in gains),ties=sum(abs(g)<=1e-6 for g in gains),losses=sum(g< -1e-6 for g in gains))
        for metric in ('fit_seconds','classifier_fit_seconds','predict_seconds','train_transform_seconds',
                       'test_transform_seconds','train_model_call_seconds','test_model_call_seconds',
                       'pipeline_seconds','worker_wall_seconds','peak_memory_bytes','peak_rss_bytes',
                       'parameter_array_bytes','total_parameter_array_bytes'):
            item[metric]=ratio_summary([means[d,candidate][metric] for d in datasets],
                                       [means[d,baseline][metric] for d in datasets])
        item['faster_test_datasets']=sum(means[d,candidate]['test_transform_seconds']<means[d,baseline]['test_transform_seconds'] for d in datasets)
        order_strata=defaultdict(list)
        for d in datasets:
            controls={int(r['seed']):r for r in groups[d,baseline]}
            for r in groups[d,candidate]:
                control=controls[int(r['seed'])]
                label='joint_first' if int(r['case_order_position'])<int(control['case_order_position']) else 'original_first'
                order_strata[label].append(float(r['test_transform_seconds'])/float(control['test_transform_seconds']))
        item['pair_order_sensitivity']={k:dict(pairs=len(v),mean_seed_pair_test_ratio=mean(v),median_seed_pair_test_ratio=median(v)) for k,v in order_strata.items()}
        for split in ('train','test'):
            a=next(r for r in overview if r['model']==candidate)
            b=next(r for r in overview if r['model']==baseline)
            item[f'median_sensitivity_{split}_ratio']=a[f'median_sensitivity_{split}_seconds']/b[f'median_sensitivity_{split}_seconds']
        logs=[math.log(means[d,candidate]['test_transform_seconds']/means[d,baseline]['test_transform_seconds']) for d in datasets]
        item['paired_signflip_mean_log_test_ratio_p']=sign_flip_pvalue(logs)
        t=item['test_transform_seconds'];memory=item['peak_memory_bytes']
        item['practical_screen_pass']=(t['ratio_of_arithmetic_means']<=.9 and t['ratio_of_means_ci95_high']<1 and memory['ratio_of_arithmetic_means']<=1.05)
        comparisons.append(item)
        for d in datasets:
            rr=groups[d,candidate][0];a,b=means[d,candidate],means[d,baseline]
            per_dataset.append(dict(dataset=d,features=budget//84*84,model=candidate,baseline=baseline,
                accuracy_percent=100*a['accuracy'],accuracy_gain_pp=100*(a['accuracy']-b['accuracy']),
                length=rr['length'],n_train=rr['n_train'],n_test=rr['n_test'],
                original_train_seconds=b['train_transform_seconds'],joint_train_seconds=a['train_transform_seconds'],
                original_test_seconds=b['test_transform_seconds'],joint_test_seconds=a['test_transform_seconds'],
                train_time_ratio=a['train_transform_seconds']/b['train_transform_seconds'],
                test_time_ratio=a['test_transform_seconds']/b['test_transform_seconds'],
                pipeline_ratio=a['pipeline_seconds']/b['pipeline_seconds'],
                physical_peak_ratio=a['peak_memory_bytes']/b['peak_memory_bytes'],rss_peak_ratio=a['peak_rss_bytes']/b['peak_rss_bytes']))
    for item,p in zip(comparisons,holm_adjust([r['paired_signflip_mean_log_test_ratio_p'] for r in comparisons])):
        item['holm_adjusted_p']=p
    result=dict(status='analyzed',verification=verification,overview=overview,comparisons=comparisons,
        per_dataset=per_dataset,audit=audit,metadata=meta,
        interpretation=dict(aggregation='Mean seeds within dataset, then equal dataset weights.',
          ratios='Ratio of arithmetic mean seconds/bytes is distinct from arithmetic mean of per-dataset ratios; both reported.',
          confidence='Paired percentile dataset bootstrap95%,10000 draws,seed0; exploratory reused archive. Secondary two-sided mean-log-ratio signflip tests use50000draws and Holm adjustment over3budgets.',
          model_bytes='Total parameter array bytes includes extractor, scaler mean/variance/scale and classifier coefficient/intercept/class arrays; excludes Python objects, hyperparameter metadata and compiled code.',
          memory='Fresh worker lifetime physical footprint preferred; RSS recorded. Includes imports/warmup/input/fit/repeated outputs/classifier/prior hashes. No sum/subtraction as allocations.',
          cache='Disposable synthetic-data JIT prewarm; every measured transform and fit must load disk cache. Worker wall is runtime/startup evidence, not first-ever compilation time.',
          leakage='All learned state uses outer TRAIN only, verified by external-TEST perturbation fixture. Internal RidgeCV is conditional on outer-TRAIN-fitted features/scaling and is not fully nested.',
          scope='Exact univariate implementation optimization at matched budgets, not a paper-protocol reproduction or universal hardware guarantee.'))
    prefix=path.with_suffix('')
    for suffix,value in (('.report.json',result),('.analysis_verification.json',audit)):
        with Path(str(prefix)+suffix).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    save_csv(Path(str(prefix)+'.overview.csv'),overview)
    save_csv(Path(str(prefix)+'.per_dataset.csv'),per_dataset)
    print(json.dumps(dict(comparisons=comparisons,audit=audit),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path)
    p.add_argument('--output-dir',required=True,type=Path);args=p.parse_args()
    analyze(args.csv,args.output_dir)


if __name__=='__main__':main()
