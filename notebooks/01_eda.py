import pandas as pd

# Correct delimiter is comma
df = pd.read_csv("data/raw/student-mat.csv", sep=",")

# Create failure target using final grade
df["failure"] = (df["G3"] < 10).astype(int)

print(df.head())
print("\nFailure counts:")
print(df["failure"].value_counts())
