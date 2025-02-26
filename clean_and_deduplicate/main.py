import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.append("./preprocessing")
from preprocessing import normalize_text, split_file
from dedup import (
    to_hash,
    generate_duplicate_pairs,
    generate_connected_components,
    generate_duplicates_dict,
)
from filtering import (
    compute_doc_feature,
    identify_language,
    find_bad_document,
    compute_perplexity,
    dump_document,
)
from config.parameters_filtering import parameters_filtering as params


def main(input_dir, output_dir, pipeline, line_count=10000, debug=False):
    print("pipeline:", pipeline)
    pipeline = pipeline.split("->")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    files = [f for f in input_dir.glob("*.jsonl")]
    # split files
    if "split" in pipeline:
        split_tmp_dir = Path(output_dir, "split_tmp")
        split_tmp_dir.mkdir(exist_ok=True)
        split_file.split_files(files, split_tmp_dir, line_count=line_count)
    # norm text
    norm_dir = Path(output_dir, "norm")
    norm_args = argparse.Namespace()
    norm_args.data_dir = split_tmp_dir
    norm_args.target_dir = norm_dir
    norm_args.n_proc = os.cpu_count() - 2
    if "norm" in pipeline:
        normalize_text.normalize_text(norm_args)

    # remove split tmp dir to save space
    if "split" in pipeline:
        shutil.rmtree(split_tmp_dir)

    # compute doc feature
    docfeat_dir = output_dir / "docfeat"
    docfeat_args = argparse.Namespace()
    docfeat_args.input_dir = norm_dir
    docfeat_args.output_dir = docfeat_dir
    docfeat_args.n_proc = os.cpu_count() - 2
    if "docfeat" in pipeline:
        compute_doc_feature.compute_document_feature(docfeat_args)

    # identify language
    lang_dir = Path(output_dir, "lang")
    lang_dir.mkdir(exist_ok=True)
    lang_args = argparse.Namespace()
    lang_args.input_dir = norm_dir
    lang_args.output_dir = lang_dir
    lang_args.model_path = "model/lid.176.bin"
    lang_args.n_proc = os.cpu_count() - 2
    if "lang" in pipeline:
        identify_language.identify_language(lang_args)

    # compute perplexity
    perplexity_dir = Path(output_dir, "perplexity")
    perplexity_dir.mkdir(exist_ok=True)
    perplexity_args = argparse.Namespace()
    perplexity_args.input_dir = norm_dir
    perplexity_args.output_dir = perplexity_dir
    perplexity_args.model_path = "model/RWKV-4-World-0.1B-v1-20230520-ctx4096.pth"
    perplexity_args.n_proc = os.cpu_count() - 2
    if "perplexity" in pipeline:
        compute_perplexity.compute_perplexity(perplexity_args)

    # generate minhash
    minhash_dir = Path(output_dir, "minhash")
    minhash_dir.mkdir(exist_ok=True)
    hash_args = argparse.Namespace()
    hash_args.input_dir = norm_dir
    hash_args.output_dir = minhash_dir
    hash_args.ngram = 13
    hash_args.cpu_count = os.cpu_count() - 2
    if "hash" in pipeline:
        to_hash.generate_hashes(hash_args)

    # generate duplicates
    dup_dir = Path(output_dir / "dup")
    dup_dir.mkdir(exist_ok=True)
    dup_pairs_args = argparse.Namespace()
    dup_pairs_args.input_dir = minhash_dir
    dup_pairs_args.out_file = dup_dir / "duplicate_pairs.txt"
    dup_pairs_args.range = 13
    dup_pairs_args.bands = 9
    dup_pairs_args.processes = os.cpu_count() - 2
    if "duppair" in pipeline:
        generate_duplicate_pairs.generate_pairs(dup_pairs_args)

    dup_connected_args = argparse.Namespace()
    dup_connected_args.input_dir = dup_dir
    dup_connected_args.out_file = dup_dir / "connected_components.pickle"
    if "dupconn" in pipeline:
        generate_connected_components.generate_connected_components_mp(
            dup_connected_args
        )

    dup_dict_args = argparse.Namespace()
    dup_dict_args.input_file = dup_dir / "connected_components.pickle"
    dup_dict_args.output_file = dup_dir / "duplicates.pickle"
    if "dupdict" in pipeline:
        generate_duplicates_dict.generate_duplicates(dup_dict_args)

    # generate bad documents list
    bad_doc_args = argparse.Namespace()
    bad_doc_args.docfeat_dir = docfeat_dir
    bad_doc_args.lang_dir = lang_dir
    bad_doc_args.perplexity_dir = None
    bad_doc_args.output_dir = dup_dir
    bad_doc_args.filter_params = params["book-v2"]
    if "baddoc" in pipeline:
        find_bad_document.find_bad_document(bad_doc_args)

    # filter bad documents and duplicates
    dedup_dir = Path(output_dir, "dedup")
    dedup_dir.mkdir(exist_ok=True)
    dump_args = argparse.Namespace()
    dump_args.input_dir = norm_dir
    dump_args.dedup_dir = dedup_dir
    dump_args.duplicate_file = dup_dir / "duplicates.pickle"
    dump_args.keep_file = dup_dir / "keep.csv"
    dump_args.n_proc = os.cpu_count() - 2
    dump_args.dry_run = False
    if "dump" in pipeline:
        dump_document.dump_good_documents(dump_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Dataset input directory.")
    parser.add_argument("output_dir", help="Dataset output directory.")
    parser.add_argument("--line_count", default=10000, type=int)
    parser.add_argument("--pipeline", default="split->norm", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.pipeline, args.line_count, args.debug)
