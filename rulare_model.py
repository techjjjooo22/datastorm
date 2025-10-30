import json, joblib
from pathlib import Path
import pandas as pd
from benchmark import engineer_features, split_features_and_target

model = joblib.load('saved_models/logistic_pd_model_calibrated.joblib')

meta = json.loads(Path('saved_models/logistic_pd_model_metadata.json').read_text(encoding='utf-8'))
threshold = float(meta['calibration']['optimal_threshold'])

df = pd.read_excel('moldova_npl.xlsx', sheet_name='NPL_Data')
features = engineer_features(df)
X, y = split_features_and_target(features)

pd_proba = model.predict_proba(X)[:, 1]
pd_flag = (pd_proba >= threshold).astype(int)

out = features.copy()
out['pd_proba'] = pd_proba
out['pd_flag'] = pd_flag
out.to_excel('scoring_output_runtime.xlsx', index=False)
print('scoring complet! prag folosit:', threshold, ' >>> fișier: scoring_output_runtime.xlsx')