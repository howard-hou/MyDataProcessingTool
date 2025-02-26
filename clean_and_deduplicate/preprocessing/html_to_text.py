import io
import json
import multiprocessing
import shutil
import sys
import time
from pathlib import Path

import py7zr
import trafilatura
from tqdm import tqdm


def read_chunks(input_file, chunk_size):
    if isinstance(input_file, str):
        f = open(input_file, encoding="utf-8")
    else:
        f = input_file  # input io here
    chunk = []
    for line in f:
        chunk.append(line)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def process_line(line):
    try:
        url, html = json.loads(line)
        text = trafilatura.extract(html)
    except:
        return None
    meta = dict(url=url)
    out = dict(meta=meta, text=text)
    out = json.dumps(out, ensure_ascii=False) + "\n"
    return out


def scrape_page(input_file, output_file, chunk_size):
    with open(output_file, "w", encoding="utf-8") as w:
        with multiprocessing.Pool(cpu_count) as pool:
            chunks = read_chunks(input_file, chunk_size)
            micro_chunk_size = chunk_size // cpu_count + 1
            for chunk in tqdm(chunks):
                chunk_results = pool.map(
                    process_line, chunk, chunksize=micro_chunk_size
                )

                for result in chunk_results:
                    if result is not None:
                        w.write(result)


def process_in_memory(input_file, output_dir):
    # Read the 7z archive file into memory
    with open(input_file, "rb") as file:
        archive_data = file.read()
    # Create an in-memory file-like object
    archive_file = io.BytesIO(archive_data)

    # Open the 7z archive using py7zr
    with py7zr.SevenZipFile(archive_file, mode="r") as z:
        # Extract all files to memory
        entry = z.getnames()[0]
        extracted_file = z.read(entry)

    # Process the extracted file
    output_file = output_dir / Path(entry).name
    print(f"processing {input_file} to {output_file}")
    scrape_page(extracted_file[entry], output_file, chunk_size)


def process_in_disk(input_file, output_dir):
    temp_extract_dir = Path("temp_extract")
    # Extract files from the archive to a temporary directory
    with py7zr.SevenZipFile(input_file, mode="r") as z:
        z.extractall(path=temp_extract_dir)
    unzip_file = list(temp_extract_dir.glob("*.jsonl"))[0]
    output_file = output_dir / Path(unzip_file).name
    print(f"processing {unzip_file} to {output_file}")
    scrape_page(unzip_file, output_file, chunk_size)

    shutil.rmtree(temp_extract_dir)
    if temp_extract_dir.exists():
        time.sleep(10)


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(exist_ok=True)
    chunk_size = 100000
    cpu_count = multiprocessing.cpu_count() - 2
    print(f"using {cpu_count} cpu, chunk size: {chunk_size}")
    process_in_memory(input_file, output_dir)
