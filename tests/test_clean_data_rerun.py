"""Data leakage, gap handling, timestamps and architecture preservation checks."""
import ast
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clean_data_rerun import prepare, FEATURES
from reference_baselines import RAW_COLUMNS, TARGET

class RerunChecks(unittest.TestCase):
    def test_gap_and_split_and_scaler(self):
        idx = pd.date_range('2020-01-01', periods=240, freq='h').delete(185)
        hourly = pd.DataFrame({c: np.arange(len(idx),dtype=float)+1 for c in RAW_COLUMNS}, index=idx)
        hourly[TARGET] = hourly[RAW_COLUMNS[0]]
        train_end, val_end = int(len(idx)*.70), int(len(idx)*.85)
        # Extreme values after training must not influence fitted scaling.
        hourly.iloc[train_end:,hourly.columns.get_loc(RAW_COLUMNS[0])] += 10000
        datasets, counts, manifest, fs, ts = prepare(hourly)
        self.assertEqual(fs.data_max_[0], train_end)
        self.assertEqual(ts.data_max_[0], train_end)
        for split,(x,y,t,actual) in datasets.items():
            self.assertEqual(x.shape, (len(t),24,12))
            np.testing.assert_allclose(ts.inverse_transform(y).ravel(),actual,rtol=1e-5)
            for timestamp in t:
                loc=idx.get_loc(timestamp)
                self.assertTrue(np.all(np.diff(idx[loc-24:loc+1].asi8)==pd.Timedelta('1h').value))
            if split=='validation': self.assertEqual(t[0],idx[train_end])
            if split=='test': self.assertTrue((t>=idx[val_end]).all())
        self.assertGreater(counts.excluded_gap_windows.sum(),0)
        self.assertTrue((manifest.input_end < manifest.target_timestamp).all())
        self.assertTrue((manifest.target_timestamp-manifest.input_start==pd.Timedelta('24h')).all())
        self.assertEqual(len(FEATURES),12)

    def test_original_definitions_unchanged(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'makale.ipynb').read_text())
        extracted={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((root/'original_model_definitions.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        for cell,name in [(77,'MLPModel'),(100,'LSTMModel'),(112,'GRUModel'),(136,'BiLSTMModelCorrected'),(79,'train_model'),(86,'predict_model'),(170,'set_seed'),(191,'create_model')]:
            node=next(n for n in ast.parse(''.join(nb['cells'][cell]['source'])).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
            self.assertEqual(extracted[name],ast.dump(node,include_attributes=False))

    def test_notebook_python_cells_compile(self):
        p=Path(__file__).resolve().parents[1]/'clean_data_models_colab.ipynb'
        for c in json.loads(p.read_text())['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),str(p),'exec')

if __name__=='__main__':unittest.main()
