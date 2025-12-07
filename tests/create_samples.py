import pandas as pd
import numpy as np

# Fixed path for all entries
path = "/netscratch/yelkheir/fft_test.wav"

# Create dataframe
data = {
    "ID": "How are you",
    "path": [path] * 10000,
    "label": np.random.choice(["bonafide", "spoof"], size=10000),
}

df = pd.DataFrame(data)

# Save to Parquet
df.to_parquet("/netscratch/yelkheir/DeepFense/DeepFense/train.parquet", index=False)
df.to_parquet("/netscratch/yelkheir/DeepFense/DeepFense/test.parquet", index=False)
