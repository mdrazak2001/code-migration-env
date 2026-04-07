import polars as pl

def process_sales(filepath: str) -> pl.DataFrame:
    return (
        pl.scan_csv(filepath)
        .filter(pl.col("revenue") > 0)
        .with_columns([
            ((pl.col("revenue") - pl.col("cost")) / pl.col("revenue"))
            .alias("profit_margin")
        ])
        .group_by("region")
        .agg([
            pl.col("revenue").sum(),
            pl.col("profit_margin").mean()
        ])
        .sort("revenue", descending=True)
        .collect()
    )
