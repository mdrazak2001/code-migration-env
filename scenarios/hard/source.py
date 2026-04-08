import pandas as pd

def process_sales(filepath: str, exclude_regions: list = None) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["order_date"])
    df = df[df["status"] != "cancelled"].copy()

    if exclude_regions:
        df = df[~df["region"].isin(exclude_regions)]

    df["region"] = df["region"].str.strip().str.title()

    # Fill missing cost using per-region median
    df["cost"] = df["cost"].fillna(df.groupby("region")["cost"].transform("median"))

    # Rolling average per region (last 3 entries)
    df = df.sort_values(["region", "order_date"])
    df["region_rolling_rev"] = (
        df.groupby("region")["revenue"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    df["profit_margin"] = (df["revenue"] - df["cost"]) / df["revenue"]

    result = (
        df.groupby(["region", "status"]).agg(
            revenue_sum=("revenue", "sum"),
            revenue_count=("revenue", "count"),
            margin_mean=("profit_margin", "mean"),
            rolling_rev_mean=("region_rolling_rev", "mean"),
        )
        .reset_index()
        .sort_values(["revenue_sum", "region"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return result