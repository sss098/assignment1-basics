import pickle
import time

from cs336_basics.bpe import train_bpe

start = time.perf_counter()

vocab, merges = train_bpe(
    input_path="data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10000,
    special_tokens=["|endoftext|"]
)

elapsed = time.perf_counter() - start

print("Training finished")
print("Vocab size", len(vocab))
print("Number of merges:", len(merges))
print("Training time", elapsed, "seconds")

longest_id, longest_token = max(
    vocab.items(),
    key=lambda item: item[1]
)

print("Longest token:")
print("id:", longest_id)
print("bytes:", longest_token)
print("length:", len(longest_token))
print(
    "text:",
    longest_token.decode("utf-8", errors="replace"),
)


with open("tinystories_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("tinystories_merges.pkl", "wb") as f:
    pickle.dump(merges, f)
