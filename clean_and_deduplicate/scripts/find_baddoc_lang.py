""" visualize docfeat distribution """

from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pandas as pd

lang_freq_min_cutoff = 0.65
lang_score_min_cutoff = 0.75


def get_files(input_dir):
    files = sorted(Path(input_dir).glob("*"))
    files = [file for file in files if file.name != ".DS_Store"]
    return files


def get_text_and_output(df, fname2norm_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # group by file_name
    for fname, group in df.groupby("file_name"):
        norm_file = fname2norm_file[fname]
        group_doc_id = group["doc_id"].tolist()
        top1_lang_freq = group["top1_lang_freq"].tolist()
        top1_lang_median = group["top1_lang_median"].tolist()
        output_file = output_dir / f"{fname}.txt"
        fout = open(output_file, "w")
        for i, line in enumerate(open(norm_file)):
            if i in group_doc_id:
                lang_freq = top1_lang_freq[group_doc_id.index(i)]
                lang_median = top1_lang_median[group_doc_id.index(i)]
                j = json.loads(line)
                text = j["text"]
                feat_str = f"lang_freq: {lang_freq}, lang_median: {lang_median}"
                break_line = "*" * 100 + "\n"
                out = f"{fname}-{i}\n{feat_str}\n{text}\n{break_line}"
                fout.write(out)


def find_threshold_lang_feature(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = get_files(args.input_dir)
    norm_files = get_files(args.norn_dir)
    fname2norm_file = {file.name: file for file in norm_files}
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
                file_name=file.name,
                doc_id=j["doc_id"],
                top1_lang=lang,
                top1_lang_freq=top1_freq,
                top1_lang_median=top1_median,
            )
            records.append(record)
    df = pd.DataFrame.from_records(records)
    cond = (df["top1_lang_freq"] <= lang_freq_min_cutoff) & (
        df["top1_lang_median"] <= lang_score_min_cutoff
    )
    lang_df = df[cond]
    sub_output_dir = output_dir / "lang"
    get_text_and_output(lang_df, fname2norm_file, sub_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory with documents.")
    parser.add_argument("norn_dir", help="Input directory with documents.")
    parser.add_argument(
        "output_dir", help="Output directory to output minhash files to."
    )
    args = parser.parse_args()
    find_threshold_lang_feature(args)
