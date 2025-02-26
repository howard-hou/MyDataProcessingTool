"""
Script that normalizes text
"""
import argparse
import json
from itertools import repeat
from math import ceil
from multiprocessing import Pool, cpu_count
from os import listdir, makedirs, path

import ftfy
import jsonlines
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing data files.",
    )
    parser.add_argument(
        "-t",
        "--target_dir",
        type=str,
        help="Path to directory where normlaized data files will be stored.",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=cpu_count() - 2,
        help="Number of processes to use.",
    )
    return parser.parse_args()


def recreate_dataset(params):
    files, args, _ = params
    for _file in files:
        file_path = path.join(args.data_dir, _file)
        target_path = path.join(args.target_dir, _file)
        with jsonlines.open(file_path) as rdr:
            with open(target_path, "w") as f:
                for ob in rdr:
                    doc = ob["text"]
                    if doc:
                        doc = ftfy.fix_text(doc, normalization="NFC")
                        record = {"meta": ob["meta"], "text": doc}
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return True


def normalize_text(args):
    makedirs(args.target_dir, exist_ok=True)
    files = sorted(listdir(args.data_dir))

    n_chunks = ceil(len(files) / args.n_proc)
    remain = len(files) % args.n_proc
    # if files are less than number of processes
    # then set number of processes to number of files
    n_proc = args.n_proc
    if n_chunks == 1 and remain:
        n_proc = remain
    # split files into chunks
    files = [files[i : i + n_chunks] for i in range(0, len(files), n_chunks)]

    with Pool(processes=n_proc) as pool:
        pbar = tqdm(
            pool.imap(
                recreate_dataset,
                zip(
                    files,
                    repeat(args),
                    range(len(files)),
                ),
            ),
            total=len(files),
            desc="Normalizing text",
        )
        for test in pbar:
            pbar.update()
            if test:
                continue


if __name__ == "__main__":
    args = parse_args()
    normalize_text(args)
