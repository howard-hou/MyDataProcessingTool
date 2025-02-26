import argparse
from collections import Counter, defaultdict
import json
import random
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from tqdm import tqdm


def get_files(input_dir):
    files = sorted(Path(input_dir).glob("*"))
    return files


def get_documents(input_file):
    with open(input_file) as rdr:
        for doc_id, doc in enumerate(rdr):
            yield doc, input_file, doc_id


def detect_language(file, output_dir, model_path, n=100):
    import fasttext

    fasttext.FastText.eprint = lambda x: None

    model = fasttext.load_model(model_path)
    buckets = []
    documents = get_documents(file)
    for doc, file_path, doc_id in documents:
        text = json.loads(doc)["text"]
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        if not sentences:
            continue
        # random sample n sentences
        random.shuffle(sentences)
        sentences = sentences[:n]
        # get file name
        file_name = Path(file_path).name
        # Predict the language
        predicted_language = model.predict(sentences)
        # get the language distribution
        if not predicted_language:
            continue
        lang2score = defaultdict(list)
        for lang, score in zip(*predicted_language):
            lang = lang[0].replace("__label__", "")
            lang2score[lang].append(score[0])
        lang2score = sorted(lang2score.items(), key=lambda x: -len(x[1]))
        language_distribution = {
            lang: dict(
                frequency=len(scores) / len(sentences),
                median_score=round(float(np.median(scores)), 3),
            )
            for lang, scores in lang2score
        }
        buckets.append(
            {
                "file_name": file_name,
                "doc_id": doc_id,
                "language_distribution": language_distribution,
            }
        )
    file_name = Path(file).name
    output_jsonl(output_dir, file_name, buckets)


def output_jsonl(output_dir, file_name, results):
    with open(output_dir / file_name, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def filter_exists_file(files, output_dir):
    # if output file exists, skip
    exists_filenames = [f.name for f in output_dir.iterdir()]
    print(f"Detect language: Found {len(exists_filenames)} existing files, skip them.")
    new_files = []
    for file in files:
        file_name = Path(file).name
        if file_name not in exists_filenames:
            new_files.append(file)
    return new_files


def identify_language(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = get_files(args.input_dir)
    files = filter_exists_file(files, output_dir)
    with Pool(args.n_proc, maxtasksperchild=1) as p:
        async_results = {}
        for file_ in files:
            file_name = Path(file_).name
            async_result = p.apply_async(
                detect_language, (file_, output_dir, args.model_path)
            )
            async_results[async_result] = file_name

        for async_result in tqdm(
            async_results, desc="Detect language", total=len(files)
        ):
            async_result.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory with documents.")
    parser.add_argument(
        "output_dir", help="Output directory to output minhash files to."
    )
    parser.add_argument(
        "--model_path", help="path to lid model.", default="model/lid.176.bin"
    )
    parser.add_argument("--n_proc", type=int, default=1, required=False)
    args = parser.parse_args()
    identify_language(args)
