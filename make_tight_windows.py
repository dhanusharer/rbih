"""
Generate tight suspicious windows based on mule_flag_date.
From data analysis:
- flag_date = end of mule activity
- Typical mule active period = 90-180 days before flag
- Not the full account lifetime (757 days) which is too wide
"""
import pandas as pd
import numpy as np
from glob import glob

# Load best submission (1793 windows, good AUC)
df = pd.read_csv('outputs/submissions/submission_20260309_1921.csv', keep_default_na=False)
print(f"Base: {len(df)} rows, {(df['suspicious_start']!='').sum()} windows")

# Load time windows with first/last txn dates
tw = pd.read_parquet('data/features/time_windows.parquet')
tw['suspicious_start'] = pd.to_datetime(tw['suspicious_start'], errors='coerce')
tw['suspicious_end']   = pd.to_datetime(tw['suspicious_end'],   errors='coerce')
tw = tw.set_index('account_id')

# Load labels for flag dates
labels = pd.read_parquet('data/raw/train_labels.parquet')

# For test accounts: use last 180 days of activity as window
# This matches what submission 10897 used and got IoU 0.267
# Try different lookback values
LOOKBACK_DAYS = 180  # days before last_txn

# Rebuild windows
df['suspicious_start'] = ''
df['suspicious_end'] = ''

threshold = float(np.quantile(df['is_mule'], 1.0 - 0.028))
mule_accounts = set(df.loc[df['is_mule'] >= threshold, 'account_id'].tolist())

for i, row in df.iterrows():
    acc = row['account_id']
    if acc not in mule_accounts:
        continue
    if acc not in tw.index:
        continue
    
    end_ts   = tw.at[acc, 'suspicious_end']
    start_ts = tw.at[acc, 'suspicious_start']
    
    if pd.isna(end_ts):
        continue
    
    # Use last LOOKBACK_DAYS before end as the suspicious window
    tight_start = end_ts - pd.Timedelta(days=LOOKBACK_DAYS)
    # But don't go before actual first transaction
    if not pd.isna(start_ts):
        tight_start = max(tight_start, start_ts)
    
    df.at[i, 'suspicious_start'] = tight_start.strftime('%Y-%m-%dT%H:%M:%S')
    df.at[i, 'suspicious_end']   = end_ts.strftime('%Y-%m-%dT%H:%M:%S')

df.to_csv('outputs/submissions/submission_tight180.csv', index=False, na_rep='')

# Verify
df2 = pd.read_csv('outputs/submissions/submission_tight180.csv', keep_default_na=False)
filled = (df2['suspicious_start'] != '').sum()
span = []
for _, row in df2[df2['suspicious_start'] != ''].iterrows():
    s = pd.to_datetime(row['suspicious_start'])
    e = pd.to_datetime(row['suspicious_end'])
    span.append((e-s).days)

print(f"Windows:      {filled}")
print(f"Median span:  {np.median(span):.0f} days")
print(f"Mean span:    {np.mean(span):.0f} days")
print(f"Max is_mule:  {df2['is_mule'].max():.6f}")
print("Saved: submission_tight180.csv")