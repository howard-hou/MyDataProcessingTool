from datasketch import MinHash, MinHashLSH
from nltk import ngrams
import re
import string


def get_features(s, width, is_chinese=False):
    # lower cased
    s = s.lower()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    s = re.sub(r"\s+", " ", s.strip())
    # split into words by whitespace
    if is_chinese:
        s = list(s)
        width = min(len(s), width)
        return map(lambda x: "".join(x), ngrams(s, width))
    s = s.split()
    width = min(len(s), width)
    return map(lambda x: " ".join(x), ngrams(s, width))


def find_duplicates(documents, threshold=0.8, is_chinese=False):
    duplicates = []
    minhashes = {}
    lsh = MinHashLSH(threshold=threshold)

    for doc_id, document in enumerate(documents):
        # Create a MinHash object for the document
        minhash = MinHash(num_perm=128)
        ngram = 13
        for x in get_features(document, ngram, is_chinese=is_chinese):
            minhash.update(x.encode("utf8"))

        matches = lsh.query(minhash)
        for prev_id in matches:
            duplicates.append((doc_id, prev_id))

        # Add the MinHash to the LSH index
        lsh.insert(doc_id, minhash)
        minhashes[doc_id] = minhash

    return duplicates


# Example usage
documents = [
    "This is the first document",
    "12V蓄电池\n蓄电池是将化学能直接转化成电能的一种装置，是按可再充电设计的电池，通过可逆的化学反应实现再充电，通常是指铅酸蓄电池，它是电池中的一种，属于二次电池。\n- 中文名\n- 12V蓄电池\n- 含 义\n- 将化学能直接转化成电能\n12V蓄电池综述编辑\n蓄电池是将化学能直接转化成电能的一种装置，是按可再充电设计的电池，通过可逆的化学反应实现再充电，通常是指铅酸蓄电池，它是电池中的一种，属于二次电池。它的工作原理：充电时利用外部的电能使内部活性物质再生，把电能储存为化学能，需要放电时再次把化学能转换为电能输出，比如生活中常用的手机电池等。",
    "And this is the third one",
    "12V蓄电池\n蓄电池是将化学能直接转化成电能的一种装置，是按可再充电设计的电池，通过可逆的化学反应实现再充电，通常是指铅酸蓄电池，它是电池中的一种，属于二次电池。\n- 中文名\n- 12V蓄电池\n- 含 义\n- 将化学能直接转化成电能\n12V蓄电池综述编辑\n蓄电池是将化学能直接转化成电能的一种装置，是按可再充电设计的电池，通过可逆的化学反应实现再充电，通常是指铅酸蓄电池，它是电池中的一种，属于二次电池。它的工作原理：充电时利用外部的电能使内部活性物质再生，把电能储存为化学能，需要放电时再次把化学能转换为电能输出，比如生活中常用的手机电池等。",
    "And this is the third one",
]

dupes = find_duplicates(documents)
print("Duplicate pairs:")
for dupe_pair in dupes:
    print(f"Document {dupe_pair[0]} and Document {dupe_pair[1]}")
