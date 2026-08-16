from collections import defaultdict
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def read_text_file(input_path: str) -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()

def pretokenize_simple(text : str) -> dict[tuple[bytes, ...], int]:
    """
    把文本按空格切开，并统计每个词出现次数。

    例如:
    "low low lower"
    会变成:
    {
        (b"l", b"o", b"w"): 2,
        (b"l", b"o", b"w", b"e", b"r"): 1
    }
    """
    word_counts = defaultdict(int)
    words = text.split()

    for word in words:
        pieces = tuple(bytes([b]) for b in word.encode("utf-8") ) # 将每个词转换为字节序列的元组
        word_counts[pieces] += 1

    return word_counts


def pretokenize(
    text: str,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """
    正式一点的 pre-tokenization：
    1. 先按 special token 切开
    2. 对每一段普通文本，用 GPT-2 regex 预分词
    3. 每个 pre-token 转成 UTF-8 bytes
    4. 统计频率
    """
    word_counts = defaultdict(int)

    if special_tokens:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        special_pattern = "|".join(escaped_tokens)
        chunks = re.split(special_pattern, text)
    else:
        chunks = [text]

    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            token = match.group(0)
            token_bytes = token.encode("utf-8")
            pieces = tuple(bytes([b]) for b in token_bytes)
            word_counts[pieces] += 1

    return dict(word_counts)

def count_pairs(counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    统计所有相邻 token pair 出现次数。

    输入:
    {
        (b"c", b"a", b"t"): 3,
        (b"d", b"o", b"g"): 1
    }

    输出:
    {
        (b"c", b"a"): 3,
        (b"a", b"t"): 3,
        (b"d", b"o"): 1,
        (b"o", b"g"): 1
    }
    """
    pair_counts = defaultdict(int)

    for word, count in counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count

    return pair_counts

def find_best_pair(pair_count: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    """
    找出现次数最多的 pair。

    如果频率相同，选择字典序更大的 pair。
    CS336 要求 tie-break 时选择 lexicographically greater pair。
    """
    if not pair_count:
        return None

    best_pair = max(pair_count.items(), key=lambda item: (item[1], item[0]))[0]
    return best_pair

def contains_pair(
    word: tuple[bytes, ...],
    pair_to_merge: tuple[bytes, bytes],
) -> bool:
    """
    判断一个 word 里面是否包含 pair_to_merge。

    例如:
    word = (b"c", b"a", b"t")
    pair_to_merge = (b"a", b"t")
    返回 True

    pair_to_merge = (b"d", b"o")
    返回 False
    """
    for i in range(len(word) - 1):
        if word[i] == pair_to_merge[0] and word[i + 1] == pair_to_merge[1]:
            return True

    return False



def merge_pair_in_word(
    word: tuple[bytes, ...],
    pair_to_merge: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    """
    在一个 word 里，把指定 pair 合并。

    例如:
    word = (b"c", b"a", b"t")
    pair_to_merge = (b"a", b"t")

    输出:
    (b"c", b"at")
    """

    new_word = []
    i = 0

    while i < len(word):
        if (
            i < len(word) - 1
            and word[i] == pair_to_merge[0]
            and word[i + 1] == pair_to_merge[1]
        ):
            merged_token = word[i] + word[i + 1]
            new_word.append(merged_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1

    return tuple(new_word)

def merge_pair_in_vocab(
        word_counts: dict[tuple[bytes, ...], int],
        pair_to_merge: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:

    """
    对整个 word_counts 执行一次 BPE merge。
    """

    new_word_counts = defaultdict(int)

    for word, count in word_counts.items():
        if contains_pair(word, pair_to_merge):
            new_word = merge_pair_in_word(word, pair_to_merge)
            new_word_counts[new_word] += count
        else:
            new_word_counts[word] += count

    return dict(new_word_counts)

def train_bpe_simple(text: str, vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    一个教学版 BPE。
    输入直接是字符串，不是文件路径。
    暂时不处理 special_tokens。
    暂时用空格 split，而不是 GPT-2 regex。
    """
    vocab = {i: bytes([i]) for i in range(256)}  # 初始化 vocab，包含所有单字节字符
    merge = []

    word_counts = pretokenize_simple(text)

    while len(vocab) < vocab_size:
        pair_counts = count_pairs(word_counts)
        best_pair = find_best_pair(pair_counts)

        if best_pair is None:
            break

        new_token = best_pair[0] + best_pair[1]

        new_id = len(vocab)
        vocab[new_id] = new_token
        merge.append(best_pair)

        word_counts = merge_pair_in_vocab(word_counts, best_pair)


    return vocab, merge

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]]:
    """
    训练 BPE。
    输入是文件路径。
    special_tokens 是一个字符串列表，里面的 token 会被加入 vocab。
    """
    text = read_text_file(input_path)
    vocab = {i: bytes([i]) for i in range(256)}

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    merge = []

    word_counts = pretokenize(text, special_tokens)

    while len(vocab) < vocab_size:
        pair_counts = count_pairs(word_counts)
        best_pair = find_best_pair(pair_counts)

        if best_pair is None:
            break

        new_token = best_pair[0] + best_pair[1]

        new_id = len(vocab)
        vocab[new_id] = new_token
        merge.append(best_pair)

        word_counts = merge_pair_in_vocab(word_counts, best_pair)

    return vocab, merge


if __name__ == "__main__":
    text = "cat cat dog cat"
    vocab, merges = train_bpe_simple(text, vocab_size=260)

    print("merges:")
    for merge in merges:
        print(merge, "->", merge[0] + merge[1])

    print("new vocab tokens:")
    for i in range(256, len(vocab)):
        print(i, vocab[i])
