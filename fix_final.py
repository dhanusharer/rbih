import pandas as pd
import numpy as np

df = pd.read_csv('outputs/submissions/submission_20260311_2015.csv')
df['suspicious_start'] = ''
df['suspicious_end'] = ''

threshold = float(np.quantile(df['is_mule'], 1.0 - 0.028))
mule_accounts = set(df.loc[df['is_mule'] >= threshold, 'account_id'].tolist())
print(f"Mules: {len(mule_accounts)}")

tw = pd.read_parquet('data/features/time_windows.parquet')
tw['suspicious_start'] = pd.to_datetime(tw['suspicious_start'], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S').fillna('')
tw['suspicious_end']   = pd.to_datetime(tw['suspicious_end'],   errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S').fillna('')
tw = tw.set_index('account_id')

for i, row in df.iterrows():
    if row['account_id'] in mule_accounts and row['account_id'] in tw.index:
        df.at[i, 'suspicious_start'] = tw.at[row['account_id'], 'suspicious_start']
        df.at[i, 'suspicious_end']   = tw.at[row['account_id'], 'suspicious_end']

# KEY FIX: write with na_rep='' AND read back with keep_default_na=False
df.to_csv('outputs/submissions/submission_final_v2.csv', index=False, na_rep='')

df2 = pd.read_csv('outputs/submissions/submission_final_v2.csv', keep_default_na=False)
print(f"NaN:    {df2['suspicious_start'].isna().sum()}  <- must be 0")
print(f"Filled: {(df2['suspicious_start'] != '').sum()}  <- must be 1794")
print(f"Max:    {df2['is_mule'].max():.6f}")
print("Saved: submission_final_v2.csv")