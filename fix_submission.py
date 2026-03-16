import pandas as pd
import numpy as np

df = pd.read_csv('outputs/submissions/submission_20260311_2015.csv')

# Force ALL windows empty first
df['suspicious_start'] = ''
df['suspicious_end'] = ''

# Top 2.8% mules
threshold = float(np.quantile(df['is_mule'], 1.0 - 0.028))
mule_mask = df['is_mule'] >= threshold
print(f"Threshold: {threshold:.4f}, Mules: {mule_mask.sum()}")

# Load windows
tw = pd.read_parquet('data/features/time_windows.parquet')
tw = tw[['account_id','suspicious_start','suspicious_end']].copy()
tw['suspicious_start'] = pd.to_datetime(tw['suspicious_start'], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S').fillna('')
tw['suspicious_end']   = pd.to_datetime(tw['suspicious_end'],   errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S').fillna('')
tw = tw.set_index('account_id')

# Apply only to mules
mule_accounts = df.loc[mule_mask, 'account_id'].values
df = df.set_index('account_id')
for acc in mule_accounts:
    if acc in tw.index:
        df.at[acc, 'suspicious_start'] = tw.at[acc, 'suspicious_start']
        df.at[acc, 'suspicious_end']   = tw.at[acc, 'suspicious_end']
df = df.reset_index()

print(f"Rows:         {len(df)}")
print(f"Max is_mule:  {df['is_mule'].max():.6f}")
print(f"With windows: {(df['suspicious_start'] != '').sum()}")
print(f"Mules > 0.06: {(df['is_mule'] > 0.06).sum()}")

df.to_csv('outputs/submissions/submission_fixed.csv', index=False)
print("DONE: outputs/submissions/submission_fixed.csv")