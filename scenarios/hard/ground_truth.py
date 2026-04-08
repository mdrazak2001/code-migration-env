import polars as pl

def process_sales(filepath: str, exclude_regions: list = None) -> pl.DataFrame:
    lf = pl.scan_csv(filepath)

    lf = lf.filter(pl.col("status") != "cancelled")

    if exclude_regions:
        lf = lf.filter(~pl.col("region").is_in(exclude_regions))

    lf = lf.with_columns([
        pl.col("region").str.strip_chars().str.to_titlecase()
    ])

    # Fill null cost with region median
    lf = lf.with_columns([
        pl.col("cost").fill_null(
            pl.col("cost").median().over("region")
        )
    ])

    # Rolling mean (window)
    lf = lf.sort(["region", "order_date"])

    lf = lf.with_columns([
        pl.col("revenue")
        .rolling_mean(window_size=3)
        .over("region")
        .alias("region_rolling_rev")
    ])

    lf = lf.with_columns([
        ((pl.col("revenue") - pl.col("cost")) / pl.col("revenue")).alias("profit_margin")
    ])

    return (
        lf.group_by(["region", "status"])
        .agg([
            pl.col("revenue").sum().alias("revenue_sum"),
            pl.col("revenue").count().alias("revenue_count"),
            pl.col("profit_margin").mean().alias("margin_mean"),
            pl.col("region_rolling_rev").mean().alias("rolling_rev_mean"),
        ])
        .sort(["revenue_sum", "region"], descending=[True, False])
        .collect()
    )