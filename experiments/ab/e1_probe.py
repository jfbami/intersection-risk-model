"""Quick data probe for E1/E3/E5. Read-only."""
import numpy as np
import pandas as pd
from pipeline.fit_risk_model import load_and_join, prepare, MODES, SHARED_PREDICTORS

df, stats = prepare(load_and_join())
print("rows:", len(df))
print("SHARED_PREDICTORS:", SHARED_PREDICTORS)
print()
print("targets:", {m.target: int(df[m.target].sum()) for m in MODES})
print()
print("max_aadt describe:")
print(df["max_aadt"].describe())
print("quantiles:", df["max_aadt"].quantile([0, .05, .25, .5, .75, .95, .99, 1]).to_dict())
print()
print("num_legs value_counts:")
print(df["num_legs"].value_counts().sort_index())
print()
print("legs_cat value_counts:")
print(df["legs_cat"].value_counts().sort_index())
print()
print("arterial_class:", df["arterial_class"].value_counts().sort_index().to_dict())
print("bike_facility dtype:", df["bike_facility"].dtype, df["bike_facility"].value_counts().to_dict())
print("is_signalized dtype:", df["is_signalized"].dtype, df["is_signalized"].value_counts().to_dict())
print("max_speed_limit:", df["max_speed_limit"].value_counts().sort_index().to_dict())
print("years_observed unique:", df["years_observed"].unique())
print("offset unique:", df["offset"].unique())
print()
print("bike_centrality describe:")
print(df["bike_centrality"].describe())
print()
print("dtypes of key cols:")
print(df[["log_aadt", "log_bike_centrality", "offset", "num_legs", "legs_cat",
          "max_aadt", "is_signalized", "max_speed_limit", "bike_facility",
          "arterial_class"]].dtypes)
