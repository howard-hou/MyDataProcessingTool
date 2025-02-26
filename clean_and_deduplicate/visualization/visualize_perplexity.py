""" visualize docfeat distribution """

import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pandas as pd
from scipy.stats import zscore


def get_files(input_dir):
    files = sorted(Path(input_dir).glob("*"))
    files = [file for file in files if file.name != ".DS_Store"]
    return files


# plot column distribution of dataframe
def plot_column_distribution(df, column, output_dir):
    plt.figure()
    df[column].hist(bins=100)
    plt.title(column)
    plt.savefig(output_dir / f"{column}.png")
    plt.close()


def visualize_perplexity(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = get_files(args.input_dir)
    records = []
    for file in tqdm(files):
        for line in open(file):
            j = json.loads(line)
            feat = {k: j[k] for k in j if k not in ["file_name", "doc_id"]}
            record = {
                "perplexity_avg": feat["perplexity_avg"],
                "perplexity_start": feat["perplexity_start"]["perplexity"],
                "perplexity_middle": feat["perplexity_middle"]["perplexity"],
                "perplexity_end": feat["perplexity_end"]["perplexity"],
            }
            records.append(record)
    df = pd.DataFrame.from_records(records)
    # Calculate the z-scores for each column
    z_scores = df.apply(zscore)
    # Define the threshold for outliers
    threshold = 3
    # Filter out rows with outliers
    df_no_outliers = df[(z_scores < threshold).all(axis=1)]
    print(df_no_outliers.head())
    for column in df.columns:
        plot_column_distribution(df_no_outliers, column, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory with documents.")
    parser.add_argument(
        "output_dir", help="Output directory to output minhash files to."
    )
    args = parser.parse_args()
    visualize_perplexity(args)
