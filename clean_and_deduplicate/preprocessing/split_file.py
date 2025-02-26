from pathlib import Path


def split_one_file(file_name, output_dir, line_count=10000, suffix_length=3):
    """
    Split a file into multiple files, each with `line_count` lines.
    """
    file_stem = file_name.stem
    output_dir = Path(output_dir)
    split_num = 0
    split_name = f"{split_num}".zfill(suffix_length)
    output_file = output_dir / f"{file_stem}.{split_name}"
    fout = open(output_file, "w")
    for i, line in enumerate(open(file_name)):
        fout.write(line)
        if (i + 1) % line_count == 0 and i != 0:
            fout.close()
            split_num += 1
            split_name = f"{split_num}".zfill(suffix_length)
            output_file = output_dir / f"{file_stem}.{split_name}"
            fout = open(output_file, "w")
    fout.close()


def split_files(files, output_dir, line_count=10000, suffix_length=3):
    """
    Split all files in `input_dir` into multiple files, each with `line_count` lines.
    """
    output_dir = Path(output_dir)
    for file_name in files:
        if file_name.is_file():
            split_one_file(
                file_name, output_dir, line_count, suffix_length=suffix_length
            )
    num_files = len(list(output_dir.glob("*")))
    print(
        f"Split to {num_files} files in {output_dir} with {line_count} lines per file"
    )
