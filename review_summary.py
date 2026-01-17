import argparse

import pandas as pd


SUMMARY_COLUMNS = [
    "section",
    "keyword_root",
    "count",
    "context_rule",
    "context_rule_match",
    "context_rule_reason",
]


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    df = df.loc[:, ~df.columns.duplicated()].copy()
    for col in ["keyword_root", "context_rule", "context_rule_match", "context_rule_reason"]:
        if col not in df.columns:
            df[col] = ""

    keyword_counts = (
        df["keyword_root"]
        .fillna("")
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "keyword_root"})
    )
    keyword_counts = keyword_counts[keyword_counts["keyword_root"] != ""]

    records = []
    for row in keyword_counts.itertuples(index=False):
        records.append(
            {
                "section": "keyword_counts",
                "keyword_root": row.keyword_root,
                "count": int(row.count),
                "context_rule": "",
                "context_rule_match": "",
                "context_rule_reason": "",
            }
        )

    context_df = df[df["context_rule"].fillna("") != ""]
    if not context_df.empty:
        context_counts = (
            context_df.fillna("")
            .groupby(
                ["keyword_root", "context_rule", "context_rule_match", "context_rule_reason"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )
        for row in context_counts.itertuples(index=False):
            records.append(
                {
                    "section": "context_counts",
                    "keyword_root": row.keyword_root,
                    "count": int(row.count),
                    "context_rule": row.context_rule,
                    "context_rule_match": row.context_rule_match,
                    "context_rule_reason": row.context_rule_reason,
                }
            )

    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a review summary CSV from final_report.csv."
    )
    parser.add_argument(
        "--input",
        default="final_report.csv",
        help="Input CSV from text_mining_pipeline.py.",
    )
    parser.add_argument(
        "--output",
        default="review_summary.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    summary = build_summary(df)
    summary.to_csv(args.output, index=False)
    print(f"Saved review summary to {args.output}")


if __name__ == "__main__":
    main()
