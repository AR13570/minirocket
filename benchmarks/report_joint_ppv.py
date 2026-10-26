"""Render the audited comparison and mark a completed run manifest."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path


def finish(path,manifest_path):
    prefix=Path(path).with_suffix('');manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    result=json.loads(Path(str(prefix)+'.report.json').read_text())
    assert Path(manifest['exit_status_file']).read_text().strip()=='0'
    assert result['verification']['raw_sha256']==hashlib.sha256(Path(path).read_bytes()).hexdigest()
    root=Path(manifest['cwd'])
    for name,sha in manifest['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    for name,sha in manifest['auxiliary_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    rows=result['overview'];comparisons=result['comparisons'];v=result['verification']
    lines=['# Original MiniRocket versus optimized joint PPV','',
        f"Completed {v['rows']} fresh-process cases on {v['datasets']} datasets. Exact features, fitted extractors, scalers, classifiers and predictions match in all {v['exact_feature_parameter_scaler_classifier_prediction_pairs']} matched pairs. All {v['stable_repetition_hashes']} full-matrix repetition hashes are stable.",'',
        'Three fit seeds use the same official archive splits. Seeds are averaged within each dataset, then datasets receive equal weight. Times are seconds for a whole split, not per-series times.','',
        '| Features | Method | Accuracy % | Fit s | Train transform s | Test transform s | Classifier fit s | Predict s | Pipeline s |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        label='Original' if r['model'].startswith('original') else 'Optimized joint PPV'
        lines.append(f"| {r['actual_features']} | {label} | {r['accuracy_percent']:.4f} | {r['fit_seconds']:.5f} | {r['train_transform_seconds']:.5f} | {r['test_transform_seconds']:.5f} | {r['classifier_fit_seconds']:.5f} | {r['predict_seconds']:.5f} | {r['pipeline_seconds']:.5f} |")
    lines+=['','## Matched performance','',
        'Ratios below one favor optimized joint PPV. Intervals are paired dataset bootstrap 95%. The two ratio aggregations are different and are listed separately. The Holm-adjusted secondary test concerns mean log per-dataset test-time ratios across the three budgets.','',
        '| Features | Test time reduction % | Ratio of mean test seconds, 95% CI | Mean per-dataset test ratio | Train time reduction % | Pipeline reduction % | Physical peak change % | Faster datasets | Holm p |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        t=r['test_transform_seconds']
        lines.append(f"| {r['actual_features']} | {100*(1-t['ratio_of_arithmetic_means']):.2f} | {t['ratio_of_arithmetic_means']:.4f} [{t['ratio_of_means_ci95_low']:.4f}, {t['ratio_of_means_ci95_high']:.4f}] | {t['mean_per_dataset_ratio']:.4f} | {100*(1-r['train_transform_seconds']['ratio_of_arithmetic_means']):.2f} | {100*(1-r['pipeline_seconds']['ratio_of_arithmetic_means']):.2f} | {100*(r['peak_memory_bytes']['ratio_of_arithmetic_means']-1):+.2f} | {r['faster_test_datasets']}/{r['datasets']} | {r['holm_adjusted_p']:.6g} |")
    lines+=['','## RAM and prediction-state arrays','',
        '| Features | Method | Mean physical peak MiB | Mean RSS peak MiB | Worst individual physical peak MiB | Worst dataset / seed | Extractor bytes | Scaler bytes | Classifier bytes | Total learned-array bytes |',
        '|---:|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        label='Original' if r['model'].startswith('original') else 'Optimized joint PPV'
        lines.append(f"| {r['actual_features']} | {label} | {r['mean_physical_peak_mib']:.2f} | {r['mean_rss_peak_mib']:.2f} | {r['worst_physical_peak_mib']:.2f} | {r['worst_peak_dataset']} / {r['worst_peak_seed']} | {r['extractor_parameter_array_bytes']:.1f} | {r['scaler_parameter_array_bytes']:.1f} | {r['classifier_parameter_array_bytes']:.1f} | {r['total_parameter_array_bytes']:.1f} |")
    passed=[str(r['actual_features']) for r in comparisons if r['practical_screen_pass']]
    lines+=['','## Assessment','',
        ('The practical screen passes at '+', '.join(passed)+' features.' if passed else 'No budget passes the complete practical performance screen.'),
        'The screen requires at least 10% lower mean test-transform time, upper paired ratio interval below one, no more than 5% higher mean physical peak, and exact features/predictions. Per-dataset regressions, median sensitivity, order sensitivity and all samples remain in the detailed artifacts.','',
        'This is an implementation-speed comparison with unchanged features. It does not establish an accuracy improvement, a reproduction of the paper classifier/protocol, or universal speed on other hardware. The archive has previously been explored; this is broader exploratory evidence, not a newly untouched accuracy benchmark.','',
        '## Fairness and leakage audit','',
        '- Same feature budget, fitting seed, raw split, dtype, batching, requested thread limits, scaler and RidgeClassifierCV for both methods. Same-budget pairs run adjacently, with randomized order based on dataset name and a fixed seed, independent of TEST contents or results.',
        '- Extractor fitting, discarded warmup, feature scaling, ridge selection and classifier fitting use outer TRAIN only. A counterfactual fixture changes external TEST values and labels and confirms all learned-state hashes remain unchanged.',
        '- Internal RidgeCV is conditional on features and scaling fitted to full outer TRAIN. It is not fully nested internal validation; internal CV scores are not presented as unbiased. External TEST remains untouched by training.',
        '- Both fixed dtype signatures are populated in a disposable synthetic-data process before measured workers. Every measured worker confirms transform cache hits and no transform/fit compilation misses. First-ever compilation time is outside this study.',
        '- Input series receive no preprocessing. Classification-feature scaling is part of the shared classifier pipeline. Unsupported archive datasets and exact reasons are disclosed rather than silently repaired.',
        '- Primary transform times include public validation, model execution/output allocation, output checks and copying. Nested model-call times exclude wrapper checks/copying, but include public validation/output allocation. Hashes and discarded warmup are outside timers. Component pipeline uses mean split transforms; worker wall includes startup, warmup, actual repeats and hashing.',
        '- RAM is a fresh worker lifetime physical footprint, with RSS also recorded. It includes imports, warmup, inputs, fitting, repeated outputs, classifier and prior hashes. Controller archive parsing is excluded. Lifetime peaks are never summed or subtracted as exclusive allocations.',
        '- Total learned-array bytes includes extractor arrays plus scaler mean/variance/scale and classifier coefficient/intercept/class arrays. It excludes Python objects, hyperparameter metadata and compiled code. Feature matrices remain rows*features*4 bytes for both methods.',
        '- Numba and supported library thread counts are checked. Accelerate is requested to use one thread through VECLIB_MAXIMUM_THREADS before process startup; threadpoolctl does not expose its actual worker count on this build. CPU features, NumPy/BLAS build and runtime environment are recorded.',
        f"- Compatibility coverage: {manifest['expected']['datasets']} compatible datasets out of {manifest['expected']['discovered']}, with {manifest['expected']['skipped']} disclosed skips. Historical exact-feature/parameter matches: {v['historical_feature_parameter_matches']}; historical accuracy differences: {len(v['historical_accuracy_differences'])}.",'',
        '## Artifacts','']
    for title,suffix in [('Raw cases','.csv'),('Overview','.overview.csv'),('Per-dataset comparisons','.per_dataset.csv'),('Verification','.verification.json'),('Analysis audit','.analysis_verification.json'),('Detailed report','.report.json')]:
        lines.append(f'- [{title}]({prefix}{suffix})')
    report=Path(str(prefix)+'.report.md')
    with report.open('x') as f:f.write('\n'.join(lines)+'\n')
    manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),completed_rows=v['rows'],
        report_markdown=str(report),report_json=str(prefix)+'.report.json',overview=str(prefix)+'.overview.csv',
        verification=str(prefix)+'.verification.json',analysis_verification=str(prefix)+'.analysis_verification.json',
        practical_screen_pass_features=[r['actual_features'] for r in comparisons if r['practical_screen_pass']])
    manifest['artifacts_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in prefix.parent.glob(prefix.name+'.*') if p.is_file() and p.suffix!='.lock'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if manifest.get('plan'):
        p=Path(manifest['plan']);p.write_text(p.read_text().replace('- [ ]','- [x]')+'\nReport: '+str(report)+'\n')
    print(report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path)
    p.add_argument('--manifest',required=True,type=Path);args=p.parse_args()
    finish(args.csv,args.manifest)


if __name__=='__main__':main()
