import argparse
import json
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


def compute_fix_length_perplexity(s, pipeline, model, max_token=100):
    tokens = pipeline.encode("\n" + s)[:max_token]
    num_token = len(tokens)
    out, state = model.forward(tokens[:1], None)
    total_log_likelihood = out[tokens[0]].cpu().item()
    for i in range(1, len(tokens)):
        out, state = model.forward(tokens[i : i + 1], state)
        log_likelihood = out[tokens[i]].cpu().item()
        total_log_likelihood += log_likelihood
    perplexity = np.exp(-total_log_likelihood / num_token)
    perplexity = round(perplexity, 2)
    decoded = pipeline.decode(tokens)
    return dict(text=decoded, perplexity=perplexity, num_token=num_token)


def compute_one_file(file, output_dir, model_path):
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    model = RWKV(model=model_path, strategy="cpu fp32", verbose=False)
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    buckets = []
    documents = get_documents(file)
    for doc, file_path, doc_id in documents:
        text = json.loads(doc)["text"]
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        if not sentences:
            continue
        # sample 5%开始100token ... 50%开始100token ... 95%开始100token
        start = len(sentences) // 20
        middle = len(sentences) // 2
        end = len(sentences) // 20 * 19
        start_doc = "\n".join(sentences[start:middle])
        middle_doc = "\n".join(sentences[middle:end])
        end_doc = "\n".join(sentences[end:])

        # get file name
        file_name = Path(file_path).name
        # Predict the language
        perplexity_start = compute_fix_length_perplexity(start_doc, pipeline, model)
        perplexity_middle = compute_fix_length_perplexity(middle_doc, pipeline, model)
        perplexity_end = compute_fix_length_perplexity(end_doc, pipeline, model)
        perplexity_avg = np.mean(
            [
                perplexity_start["perplexity"],
                perplexity_middle["perplexity"],
                perplexity_end["perplexity"],
            ]
        ).round(2)
        total_num_token = (
            perplexity_start["num_token"]
            + perplexity_middle["num_token"]
            + perplexity_end["num_token"]
        )
        # get the language distribution
        buckets.append(
            {
                "file_name": file_name,
                "doc_id": doc_id,
                "perplexity_start": perplexity_start,
                "perplexity_middle": perplexity_middle,
                "perplexity_end": perplexity_end,
                "perplexity_avg": perplexity_avg,
                "total_num_token": total_num_token,
            }
        )
    file_name = Path(file).name
    output_jsonl(output_dir, file_name, buckets)


def output_jsonl(output_dir, file_name, results):
    with open(output_dir / file_name, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


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


def compute_perplexity(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = get_files(args.input_dir)
    files = filter_exists_file(files, output_dir)
    with Pool(args.n_proc, maxtasksperchild=1) as p:
        async_results = {}
        for file_ in files:
            file_name = Path(file_).name
            async_result = p.apply_async(
                compute_one_file, (file_, output_dir, args.model_path)
            )
            async_results[async_result] = file_name

        for async_result in tqdm(
            async_results, desc="Compute perplexity", total=len(files)
        ):
            async_result.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory with documents.")
    parser.add_argument(
        "output_dir", help="Output directory to output minhash files to."
    )
    parser.add_argument("--model_path", help="path to rwkv model.")

    parser.add_argument("--n_proc", type=int, default=1, required=False)
    args = parser.parse_args()
    compute_perplexity(args)
