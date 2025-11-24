import pandas as pd
df = pd.read_parquet("preprocessed.parquet")
print(df.shape)
print(df.columns)
print(df.head(2))

