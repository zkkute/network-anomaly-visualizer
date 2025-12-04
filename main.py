import pandas as pd

df = pd.read_csv("data/raw/cic-ids-2017/CSVs/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", low_memory=False)

print(df.columns.tolist())
