import pandas as pd

def process_sales(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df[df["revenue"] > 0]
    df["profit_margin"] = (df["revenue"] - df["cost"]) / df["revenue"]
    df = df.groupby("region").agg({"revenue": "sum", "profit_margin": "mean"})
    df = df.sort_values("revenue", ascending=False)
    return df
