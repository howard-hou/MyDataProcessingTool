import argparse
from collections import Counter, defaultdict
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import emoji
import math
import string
from .document import ModifyingDocuments
from .stopwords import stopwords
from .flagged_words import flagged_words
from .languages_id import langs_id
from .utils import round_dict_value


main_special_characters = string.punctuation + string.digits + string.whitespace
other_special_characters = (
    "    　    ￼’“”–ー一▬…✦�­£​•€«»°·═"
    "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
    "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
    "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
    "」﴾》"
)
emoji = list(emoji.UNICODE_EMOJI["en"].keys())

special_characters_default = set(main_special_characters + other_special_characters)
special_characters_default.update(emoji)


def load_stopwords(lang_dataset_id):
    stopwords_lang_id = langs_id.loc[
        langs_id["dataset_id"] == lang_dataset_id, "stopwords_id"
    ].iloc[0]
    if stopwords_lang_id:
        stopwords_lang = set(stopwords[stopwords_lang_id])
    else:
        stopwords_lang = None
    return stopwords_lang

def load_flagged_words(lang_dataset_id):
    flagged_words_lang_id = langs_id.loc[
        langs_id["dataset_id"] == lang_dataset_id, "flagged_words_id"
    ].iloc[0]
    if flagged_words_lang_id:
        flagged_words_lang = set(flagged_words[flagged_words_lang_id])
    else:
        flagged_words_lang = None
    return flagged_words_lang


def get_files(input_dir):
    files = sorted(Path(input_dir).glob("*"))
    return files


def get_documents(input_file):
    with open(input_file) as rdr:
        for doc_id, doc in enumerate(rdr):
            yield doc, input_file, doc_id


def compute_flagged_words_ratio(
    word_distribution,
    flagged_words,
):
    flagged_words_ratio = sum(
        [word_distribution[word] for word in word_distribution if word in flagged_words]
    )
    if flagged_words_ratio > 1.0:
        flagged_words_ratio = 1.0
    return flagged_words_ratio

def compute_stopwords_ratio(
        word_distribution,
        stopwords,
    ):
        stopwords_ratio = sum(
            [word_distribution[word] for word in word_distribution if word in stopwords]
        )
        if stopwords_ratio > 1.0:
            stopwords_ratio = 1.0
        return stopwords_ratio


def compute_special_character_ratio(char_distribution, special_characters):
    if len(char_distribution) == 0:
        return 0
    special_characters_ratio = sum(
        [char_distribution[char] for char in char_distribution if char in special_characters]
    )
    return special_characters_ratio


def compute_word_repetition_ratio(words, word_repetition_length):
    def get_freq_word_ngrams(words, n):
        word_ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        freq_word_ngrams = {}
        for word_ngram in word_ngrams:
            freq_word_ngrams[word_ngram] = freq_word_ngrams.get(word_ngram, 0) + 1
        return freq_word_ngrams

    freq_word_ngrams = get_freq_word_ngrams(words, word_repetition_length)
    if len(freq_word_ngrams) == 0:
        return 0
    freq_word_ngrams = list(freq_word_ngrams.values())
    word_repetition_ratio = sum(freq for freq in freq_word_ngrams if freq > 1) / sum(
        freq_word_ngrams
    )
    return word_repetition_ratio


def compute_character_repetition_ratio(document, character_repetition_length):
    def get_freq_character_ngrams(document, n):
        character_ngrams = [document[i : i + n] for i in range(len(document) - n + 1)]
        freq_character_ngrams = {}
        for character_ngram in character_ngrams:
            freq_character_ngrams[character_ngram] = (
                freq_character_ngrams.get(character_ngram, 0) + 1
            )
        return freq_character_ngrams

    freq_character_ngrams = get_freq_character_ngrams(
        document, character_repetition_length
    )
    if len(freq_character_ngrams) == 0:
        return 0
    freq_character_ngrams = list(freq_character_ngrams.values())
    freq_character_ngrams = sorted(freq_character_ngrams, reverse=True)
    val_less_than_one = len([el for el in freq_character_ngrams if el > 1])
    num_rep_character_ngrams = min(
        int(np.sqrt(len(freq_character_ngrams))),
        len(freq_character_ngrams) - val_less_than_one,
    )
    character_repetition_ratio = sum(
        freq_character_ngrams[:num_rep_character_ngrams]
    ) / sum(freq_character_ngrams)
    return character_repetition_ratio

def compute_entropy(dictionary):
    total_count = sum(dictionary.values())
    entropy = 0.0

    for count in dictionary.values():
        probability = count / total_count
        entropy -= probability * math.log2(probability)

    return entropy

def compute_char_distribution(document):
    document = [c.lower() for c in document]
    return pd.value_counts(document, normalize=True).to_dict()

def compute_word_distribution(words):
    return pd.value_counts(words, normalize=True).to_dict()

def compute_char_entropy(char_distribution):
    return compute_entropy(char_distribution)

def compute_word_entropy(word_distribution):
    return compute_entropy(word_distribution)


def compute_features(
    file, 
    output_dir, 
    max_text_length=100000,
    character_repetition_length=10,
    word_repetition_length=5
):
    buckets = []
    documents = get_documents(file)
    for doc, file_path, doc_id in documents:
        # get file name
        file_name = Path(file_path).name
        text = json.loads(doc)["text"]
        document_length = len(text)
        text_truncated = text[:max_text_length]
        # words_truncated = ModifyingDocuments.get_words_from_document(
        #     text_truncated,
        #     sentencepiece_model_tok=None,
        #     lower_case=True,
        #     strip_characters=special_characters_default,
        # )
        # number_of_words = len(words_truncated)
        char_distribution = compute_char_distribution(text_truncated)
        # word_distribution = compute_word_distribution(words_truncated)
        
        character_repetition_ratio = compute_character_repetition_ratio(
            text_truncated, character_repetition_length=character_repetition_length
        )
        # word_repetition_ratio = compute_word_repetition_ratio(
        #     words_truncated, word_repetition_length=word_repetition_length
        # )
        special_character_ratio = compute_special_character_ratio(
            char_distribution, special_characters=special_characters_default
        )
        # stopword_ratio = compute_stopwords_ratio(
        #     word_distribution, 
        #     stopwords=load_stopwords(lang_dataset_id="en"),
        # )
        # flagged_word_ratio = compute_flagged_words_ratio(
        #     word_distribution,
        #     flagged_words=load_flagged_words(lang_dataset_id="en"),
        # )
        char_entropy = compute_char_entropy(char_distribution)
        # word_entropy = compute_word_entropy(word_distribution)
        buckets.append(
            round_dict_value(
            {
                "file_name": file_name,
                "doc_id": doc_id,
                "document_length": document_length,
                # "number_of_words": number_of_words,
                "character_repetition_ratio": character_repetition_ratio,
                # "word_repetition_ratio": word_repetition_ratio,
                "special_character_ratio": special_character_ratio,
                # "stopword_ratio": stopword_ratio,
                # "flagged_word_ratio": flagged_word_ratio,
                "char_entropy": char_entropy,
                # "word_entropy": word_entropy,
            }
            )
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
    print(f"Doc feature: Found {len(exists_filenames)} existing files, skip them.")
    new_files = []
    for file in files:
        file_name = Path(file).name
        if file_name not in exists_filenames:
            new_files.append(file)
    return new_files


def compute_document_feature(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = get_files(args.input_dir)
    files = filter_exists_file(files, output_dir)
    with Pool(args.n_proc, maxtasksperchild=1) as p:
        async_results = {}
        for file_ in files:
            file_name = Path(file_).name
            async_result = p.apply_async(compute_features, (file_, output_dir))
            async_results[async_result] = file_name

        for async_result in tqdm(
            async_results, desc="Compute docfeat", total=len(files)
        ):
            async_result.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory with documents.")
    parser.add_argument(
        "output_dir", help="Output directory to output minhash files to."
    )
    parser.add_argument("--n_proc", type=int, default=1, required=False)
    args = parser.parse_args()
    compute_document_feature(args)
