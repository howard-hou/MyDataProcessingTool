""" visualize docfeat distribution """

import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pandas as pd


def get_files(input_dir):
    files = sorted(Path(input_dir).glob("*"))
    files = [file for file in files if file.name != ".DS_Store"]
    return files


# plot column distribution of dataframe
def plot_column_distribution(data, name, output_dir, type="hist"):
    plt.figure()
    if type == "hist":
        data.hist(bins=100)
    if type == "bar":
        data.plot.bar()
    plt.title(name)
    plt.savefig(output_dir / f"{name}.png")
    plt.close()


def visualize_language_feature(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = get_files(args.input_dir)
    records = []
    for file in tqdm(files):
        for line in open(file):
            j = json.loads(line)
            feat = j["language_distribution"]
            feat = sorted(feat.items(), key=lambda x: -x[1]["frequency"])
            lang = feat[0][0]
            top1_freq = feat[0][1]["frequency"]
            top1_median = feat[0][1]["median_score"]
            record = dict(
                top1_lang=lang, top1_lang_freq=top1_freq, top1_lang_median=top1_median
            )
            records.append(record)
    df = pd.DataFrame.from_records(records)
    for column in df.columns:
        if column == "top1_lang":
            data = df[column].value_counts(normalize=True)
            plot_column_distribution(data, column, output_dir, type="bar")
        else:
            data = df[column]
            plot_column_distribution(data, column, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory with documents.")
    parser.add_argument(
        "output_dir", help="Output directory to output minhash files to."
    )
    args = parser.parse_args()
    visualize_language_feature(args)
