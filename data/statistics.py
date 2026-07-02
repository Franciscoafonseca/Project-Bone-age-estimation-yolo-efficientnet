import os
from collections import Counter, defaultdict
import pandas as pd

def compute_distributions(preprocessed_dir, ann_file, sheet_name=1, out_csv=None, out_csv_thresholds=None):
    df = pd.read_excel(ann_file, sheet_name=sheet_name)

    age_col = df['Idade (anos) ao Rx']
    sex_col = df['Sexo']

    counts = defaultdict(Counter)

    # iterate subfolders and image files
    for f in os.listdir(preprocessed_dir):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"Skipping non-image file: {f}")
            continue

        number = int(os.path.splitext(f)[0])
        row_idx = number - 1

        # get age and sex
        age_val = age_col[row_idx]
        sex_val = sex_col[row_idx].replace(' ','')

        counts[age_val][sex_val] += 1

    # put ages as rows
    summary_df = pd.DataFrame.from_dict(counts, orient="index").fillna(0).astype(int)

    # sort by age (assuming numeric)
    summary_df = summary_df.sort_index()

    # add totals per row
    summary_df['Total'] = summary_df.sum(axis=1)

    # add totals row
    totals_row = summary_df.sum(axis=0)
    totals_row.name = 'Total'
    summary_df = pd.concat([summary_df, totals_row.to_frame().T])

    if out_csv:
        summary_df.to_csv(out_csv)

    print("Summary (age x sex) with totals:")
    print(summary_df)

    thresholds = [10, 12, 14, 16, 18, 21]

    # Expand counts to flat list of (age, sex, count)
    expanded = []
    for age, sex_counts in counts.items():
        for sex, c in sex_counts.items():
            expanded.append((age, sex, c))

    cum_counts = {}
    for t in thresholds:
        below = Counter()
        above = Counter()
        for age, sex, c in expanded:
            if age < t:
                below[sex] += c
                below['Total'] += c
            else:
                above[sex] += c
                above['Total'] += c

        cum_counts[f"<{t}"] = below
        cum_counts[f">={t}"] = above

    cum_df = pd.DataFrame.from_dict(cum_counts, orient="index").fillna(0).astype(int)

    # reorder columns so "Total" comes last
    cols = [c for c in cum_df.columns if c != "Total"] + ["Total"]
    cum_df = cum_df[cols]

    if out_csv_thresholds:
        cum_df.to_csv(out_csv_thresholds)

    print("\nCumulative threshold counts (by sex and total):")
    print(cum_df)

    return summary_df, counts

if __name__ == "__main__":
    compute_distributions(r'./data/preprocessed/clahe_center_crop_then_crop/without_treatment', r'./data/FCT2025_amostra_VF2.xlsx',sheet_name=1,out_csv=r'./distributions.csv', out_csv_thresholds=r'./cumulative_distributions.csv')