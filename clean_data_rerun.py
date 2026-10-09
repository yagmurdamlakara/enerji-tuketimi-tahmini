"""Clean-data rerun of the original four models; no manuscript/FPGA steps."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from reference_baselines import load_and_prepare_hourly, RAW_COLUMNS, TARGET

FEATURES = RAW_COLUMNS + ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']
SEEDS = [11, 22, 33, 44, 55]
MODELS = ['MLP', 'LSTM', 'GRU', 'BiLSTM']
CONFIG = dict(lookback=24, features=FEATURES, seeds=SEEDS, models=MODELS,
              train_fraction=0.70, validation_end_fraction=0.85,
              scaler='MinMaxScaler; training rows only', batch_size=256,
              shuffle=False, learning_rate=0.001, max_epochs=150, patience=10,
              optimizer='Adam', loss='MSELoss', std_ddof=1,
              original_commit='6d74b9b93d34a6efdd9fc3efda65468aaa788172',
              baseline_commit='b3f7008c8753ddd107267d9c531cd281bbabca76')


def prepare(hourly):
    hourly = hourly.copy()
    idx = hourly.index
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError('Hourly timestamps must be unique and sorted.')
    hourly['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24)
    hourly['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24)
    hourly['dow_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7)
    hourly['dow_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7)
    hourly['is_weekend'] = (idx.dayofweek >= 5).astype(int)
    n = len(hourly)
    train_end, val_end = int(n * .70), int(n * .85)
    if train_end <= 24 or val_end >= n:
        raise ValueError('Not enough hourly data for chronological splits.')
    fs, ts = MinMaxScaler(), MinMaxScaler()
    fs.fit(hourly.iloc[:train_end][FEATURES])
    ts.fit(hourly.iloc[:train_end][[TARGET]])
    x = fs.transform(hourly[FEATURES]).astype('float32')
    y = ts.transform(hourly[[TARGET]]).astype('float32')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite feature/target values.')
    # Require exact t-24,...,t-1 and target t, not just 24 retained rows.
    positions = np.arange(24, n)
    continuous = np.array([
        np.all(np.diff(idx[i-24:i+1].asi8) == pd.Timedelta(hours=1).value)
        for i in positions
    ])
    accepted = positions[continuous]
    datasets, counts, manifests = {}, [], []
    for name, lo, hi in [('train', 0, train_end), ('validation', train_end, val_end), ('test', val_end, n)]:
        pos = accepted[(accepted >= lo) & (accepted < hi)]
        if len(pos) < 2:
            raise ValueError(f'Too few continuous windows in {name}.')
        datasets[name] = (np.stack([x[i-24:i] for i in pos]), y[pos], idx[pos], hourly[TARGET].iloc[pos].to_numpy())
        candidates = max(0, hi - max(lo, 24))
        counts.append(dict(split=name, hourly_rows=hi-lo, candidate_windows=candidates,
                           excluded_gap_windows=candidates-len(pos), samples=len(pos),
                           first_target=str(idx[pos[0]]), last_target=str(idx[pos[-1]])))
        manifests.append(pd.DataFrame(dict(split=name, input_start=idx[pos-24],
            input_end=idx[pos-1], target_timestamp=idx[pos], actual_kwh=hourly[TARGET].iloc[pos].to_numpy())))
    return datasets, pd.DataFrame(counts), pd.concat(manifests, ignore_index=True), fs, ts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('clean_model_results'))
    parser.add_argument('--prepare-only', action='store_true', help='Audit preprocessing without training.')
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory is not empty. Choose a new run directory.')
    out.mkdir(parents=True, exist_ok=True)
    hourly, audit = load_and_prepare_hourly(args.data)
    datasets, counts, manifest, fs, ts = prepare(hourly)
    print(counts.to_string(index=False), flush=True)
    counts.to_csv(out/'split_counts.csv', index=False)
    manifest.to_csv(out/'sequence_manifest.csv', index=False)
    targets = manifest.loc[manifest.split == 'test', ['target_timestamp','actual_kwh']]
    targets.to_csv(out/'model_test_targets.csv', index=False)
    joblib.dump(dict(feature_scaler=fs, target_scaler=ts), out/'scalers.joblib')
    audit.update(CONFIG)
    audit['python'] = platform.python_version()
    audit['numpy'] = np.__version__
    audit['pandas'] = pd.__version__
    audit['sklearn'] = sklearn.__version__
    audit['raw_sha256'] = hashlib.file_digest(args.data.open('rb'), 'sha256').hexdigest()
    audit['test_targets_sha256'] = hashlib.sha256((out/'model_test_targets.csv').read_bytes()).hexdigest()
    audit['window_policy'] = 'Only exact continuous t-24..t; split by target before window filtering; past split context allowed.'
    audit['split_counts'] = counts.to_dict('records')
    audit['status'] = 'prepared'
    audit_path = out/'run_summary.json'
    audit_path.write_text(json.dumps(audit, indent=2))
    if args.prepare_only:
        return
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import original_model_definitions as original
    audit.update(torch=torch.__version__, cuda=torch.version.cuda, device=str(original.device),
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 cudnn_deterministic=torch.backends.cudnn.deterministic,
                 cudnn_benchmark=torch.backends.cudnn.benchmark,
                 status='training', completed_runs=0)
    audit_path.write_text(json.dumps(audit, indent=2))
    loaders = {k: DataLoader(TensorDataset(torch.from_numpy(v[0]), torch.from_numpy(v[1])),
                            batch_size=256, shuffle=False) for k,v in datasets.items()}
    rows = []
    for name in MODELS:
        for seed in SEEDS:
            print(f'\nMODEL {name} | SEED {seed}', flush=True)
            original.set_seed(seed)
            model = original.create_model(name)
            model, train_losses, val_losses = original.train_model(
                model, loaders['train'], loaders['validation'],
                learning_rate=.001, max_epochs=150, patience=10)
            pred_scaled, true_scaled = original.predict_model(model, loaders['test'])
            pred = ts.inverse_transform(pred_scaled).ravel()
            actual = datasets['test'][3]
            np.testing.assert_allclose(ts.inverse_transform(true_scaled).ravel(), actual, rtol=1e-5, atol=1e-6)
            if not np.isfinite(pred).all():
                raise ValueError('Non-finite test predictions.')
            pd.DataFrame(dict(target_timestamp=datasets['test'][2], actual_kwh=actual,
                              prediction_kwh=pred, model=name, seed=seed)).to_csv(out/f'{name}_seed_{seed}_predictions.csv', index=False)
            torch.save({k:v.cpu() for k,v in model.state_dict().items()}, out/f'{name}_seed_{seed}.pth')
            pd.DataFrame(dict(epoch=np.arange(1,len(val_losses)+1), train_loss=train_losses,
                              validation_loss=val_losses)).to_csv(out/f'{name}_seed_{seed}_history.csv', index=False)
            rows.append(dict(Model=name, Seed=seed, MAE=mean_absolute_error(actual,pred),
                             RMSE=float(np.sqrt(mean_squared_error(actual,pred))), R2=r2_score(actual,pred),
                             Best_epoch=int(np.argmin(val_losses)+1), Epochs=len(val_losses),
                             Parameters=sum(p.numel() for p in model.parameters()),
                             Train=len(datasets['train'][0]), Validation=len(datasets['validation'][0]), Test=len(actual)))
            pd.DataFrame(rows).to_csv(out/'per_seed_metrics.csv',index=False)
            audit['completed_runs'] = len(rows)
            audit_path.write_text(json.dumps(audit, indent=2))
            del model
    results = pd.DataFrame(rows)
    summary, formatted = [], []
    for name in MODELS:
        group = results[results.Model == name]
        assert group.Seed.tolist() == SEEDS
        row = dict(Model=name, Train=int(group.Train.iloc[0]), Validation=int(group.Validation.iloc[0]), Test=int(group.Test.iloc[0]))
        display_row = row.copy()
        for metric in ['MAE','RMSE','R2']:
            row[metric+'_mean'] = group[metric].mean()
            row[metric+'_std'] = group[metric].std(ddof=1)
            display_row[metric] = f'{row[metric+"_mean"]:.6f} ± {row[metric+"_std"]:.6f}'
        summary.append(row)
        formatted.append(display_row)
    pd.DataFrame(summary).to_csv(out/'model_summary_numeric.csv',index=False)
    table = pd.DataFrame(formatted)
    table.to_csv(out/'advisor_table.csv',index=False)
    (out/'advisor_table.txt').write_text('MAE/RMSE: kWh; R2: birimsiz; std: ddof=1, 5 seed\n'+table.to_string(index=False))
    audit['status'] = 'complete'
    audit_path.write_text(json.dumps(audit, indent=2))
    print(table.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
