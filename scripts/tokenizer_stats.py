import pickle
import time

from cs336_basics.tokenizer import Tokenizer


with open("tinystories_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("tinystories_merges.pkl", "rb") as f:
    merges = pickle.load(f)


tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=["<|endoftext|>"],
)


with open(
    "data/TinyStoriesV2-GPT4-valid.txt",
    "r",
    encoding="utf-8",
) as f:
    text = f.read()


start = time.perf_counter()

ids = tokenizer.encode(text)

elapsed = time.perf_counter() - start


num_bytes = len(text.encode("utf-8"))
num_tokens = len(ids)

compression_ratio = num_bytes / num_tokens
throughput = num_bytes / elapsed


print("Bytes:", num_bytes)
print("Tokens:", num_tokens)

print(
    "Compression ratio:",
    compression_ratio,
    "bytes/token",
)

print(
    "Throughput:",
    throughput,
    "bytes/second",
)