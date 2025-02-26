from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pandas as pd


def get_files(input_dir):
    files = sorted(Path(input_dir).glob("*"))
    files = [file for file in files if file.name != ".DS_Store"]
    return files


def process_docfeat_files(files):
    records = []
    for file in tqdm(files):
        for line in open(file):
            j = json.loads(line)
            records.append(j)
    df = pd.DataFrame.from_records(records)
    return df


def process_lang_files(files):
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
                file_name=j["file_name"],
                doc_id=j["doc_id"],
                top1_lang_freq=top1_freq,
                top1_lang_median=top1_median,
            )
            records.append(record)
    df = pd.DataFrame.from_records(records)
    return df


def process_perplexity_files(files):
    records = []
    for file in tqdm(files):
        for line in open(file):
            j = json.loads(line)
            feat = {k: j[k] for k in j if k not in ["file_name", "doc_id"]}
            record = {
                "file_name": j["file_name"],
                "doc_id": j["doc_id"],
                "perplexity_avg": feat["perplexity_avg"],
                "perplexity_start": feat["perplexity_start"]["perplexity"],
                "perplexity_middle": feat["perplexity_middle"]["perplexity"],
                "perplexity_end": feat["perplexity_end"]["perplexity"],
            }
            records.append(record)
    df = pd.DataFrame.from_records(records)
    return df


def check_keep_conditions(merged_df, filter_params, use_perplexity):
    lang_cond = (
        merged_df["top1_lang_freq"] >= filter_params["lang_freq_min_cutoff"]
    ) & (merged_df["top1_lang_median"] >= filter_params["lang_score_min_cutoff"])
    docfeat_cond = (
        (merged_df["document_length"] >= filter_params["document_length_min_cutoff"])
        & (merged_df["document_length"] <= filter_params["document_length_max_cutoff"])
        & (
            merged_df["character_repetition_ratio"]
            <= filter_params["character_repetition_max_cutoff"]
        )
        & (
            merged_df["special_character_ratio"]
            <= filter_params["special_characters_max_cutoff"]
        )
        & (merged_df["char_entropy"] >= filter_params["char_entropy_min_cutoff"])
    )
    if use_perplexity:
        perplexity_cond = (
            merged_df["perplexity_avg"] <= filter_params["perplexity_max_cutoff"]
        )
    # take the intersection of all conditions
    merged_df["keep"] = lang_cond & docfeat_cond
    if use_perplexity:
        merged_df["keep"] = merged_df["keep"] & perplexity_cond
    merged_df["drop"] = ~merged_df["keep"]
    return merged_df


def find_bad_document(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lang_files = get_files(args.lang_dir)
    docfeat_files = get_files(args.docfeat_dir)
    docfeat_df = process_docfeat_files(docfeat_files)
    lang_df = process_lang_files(lang_files)
    if args.perplexity_dir is not None:
        perplexity_files = get_files(args.perplexity_dir)
        perplexity_df = process_perplexity_files(perplexity_files)
    filter_params = args.filter_params
    # merge all dataframes by file_name and doc_id
    merged_df = pd.merge(docfeat_df, lang_df, on=["file_name", "doc_id"], how="left")
    if args.perplexity_dir is not None:
        merged_df = pd.merge(
            merged_df, perplexity_df, on=["file_name", "doc_id"], how="left"
        )
    # check keep conditions
    merged_df = check_keep_conditions(
        merged_df, filter_params, use_perplexity=args.perplexity_dir is not None
    )
    # output keep and drop documents
    keep_df = merged_df[merged_df["keep"]]
    drop_df = merged_df[merged_df["drop"]]
    keep_df.to_csv(output_dir / "keep.csv", index=False)
    drop_df.to_csv(output_dir / "drop.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("docfeat_dir", help="Input directory with documents.")
    parser.add_argument("lang_dir", help="Output directto.")
    parser.add_argument("perplexity_dir", help="Output directto.")
    parser.add_argument("output_dir", help="Output directto.")
    args = parser.parse_args()
    from parameters_filtering import parameters_filtering as filter_params

    args.filter_params = filter_params["book-v2"]
    find_bad_document(args)
