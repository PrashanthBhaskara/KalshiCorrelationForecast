import pandas as pd

# Load CME data
df = pd.read_csv('Fed_Analysis/FedMeeting_20260128.csv')

# Check last row
print("Last row date:", df.iloc[-1]['Date'])
print("\nColumns with non-zero values in last row:")
last_row = df.iloc[-1]
for col in df.columns:
    if col != 'Date' and last_row[col] > 0:
        print(f"  {col}: {last_row[col]}")

# Check what (300-325) column looks like
if '(300-325)' in df.columns:
    print(f"\n(300-325) column in last row: {df.iloc[-1]['(300-325)']}")
else:
    print("\n(300-325) column NOT FOUND")
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")