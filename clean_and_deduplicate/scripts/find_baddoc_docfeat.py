""" visualize docfeat distribution """

from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pandas as pd

character_repetition_max_cutoff = 0.4
char_entropy_min_cutoff = 3
special_characters_max_cutoff = 0.5
document_length_min_cutoff = 100
document_length_max_cutoff = 10000000


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
        output_file = output_dir / f"{fname}.txt"
        fout = open(output_file, "w")
        for i, line in enumerate(open(norm_file)):
            if i in group_doc_id:
                j = json.loads(line)
                text = j["text"]
                break_line = "*" * 100 + "\n"
                out = f"{fname}-{i}\n{text}\n{break_line}"
                fout.write(out)


def find_threshold_document_feature(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = get_files(args.input_dir)
    norm_files = get_files(args.norm_dir)
    fname2norm_file = {file.name: file for file in norm_files}
    records = []
    for file in tqdm(files):
        for line in open(file):
            j = json.loads(line)
            feat = {k: j[k] for k in j}
            records.append(feat)
    df = pd.DataFrame.from_records(records)
    character_repetition_df = df[
        df["character_repetition_ratio"] > character_repetition_max_cutoff
    ]
    print("character_repetition_df", character_repetition_df.shape)
    sub_output_dir = output_dir / "character_repetition"
    get_text_and_output(character_repetition_df, fname2norm_file, sub_output_dir)
    char_entropy_df = df[df["char_entropy"] < char_entropy_min_cutoff]
    print("char_entropy_df", char_entropy_df.shape)
    sub_output_dir = output_dir / "char_entropy"
    get_text_and_output(char_entropy_df, fname2norm_file, sub_output_dir)
    special_characters_df = df[
        df["special_character_ratio"] > special_characters_max_cutoff
    ]
    print("special_characters_df", special_characters_df.shape)
    sub_output_dir = output_dir / "special_characters"
    get_text_and_output(special_characters_df, fname2norm_file, sub_output_dir)
    document_length_df = df[
        (df["document_length"] < document_length_min_cutoff)
        | (df["document_length"] > document_length_max_cutoff)
    ]
    print("document_length_df", document_length_df.shape)
    sub_output_dir = output_dir / "document_length"
    get_text_and_output(document_length_df, fname2norm_file, sub_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory with documents.")
    parser.add_argument("norm_dir", help="Input directory with documents.")
    parser.add_argument(
        "output_dir", help="Output directory to output minhash files to."
    )
    args = parser.parse_args()
    find_threshold_document_feature(args)
