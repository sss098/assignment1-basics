import pickle
import numpy as np

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
    "data/TinyStoriesV2-GPT4-train.txt",
    "r",
    encoding="utf-8",
) as f:
    ids = list(tokenizer.encode_iterable(f))



ids = np.array(
    ids,
    dtype=np.uint16,
)

np.save(
    "tinystories_train.npy",
    ids,
)

print("Number of tokens:", len(ids))
print("dtype:", ids.dtype)
print("min:", ids.min())
print("max:", ids.max())
