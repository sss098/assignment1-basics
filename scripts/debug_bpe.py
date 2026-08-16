import time
from cs336_basics.bpe import train_bpe

start = time.perf_counter()

vocab, merges = train_bpe(
    input_path="data/TinyStoriesV2-GPT4-valid.txt",
    vocab_size=1000,
    special_tokens=["|endoftext|"]

)

end = time.perf_counter()

print("vocab size:", len(vocab))
print("number of merges:", len(merges))
print("training time:", end - start, "seconds")

print("\nLast 20 tokens:")

for token_id in range(max(256, len(vocab) - 20), len(vocab)):
    token_bytes = vocab[token_id]

    print(
        token_id,
        token_bytes,
        "->",
        token_bytes.decode("utf-8", errors="replace")
    )


longest_id, longest_token = max(
    vocab.items(),
    key=lambda item: len(item[1])
)

print("Longest token id:", longest_id)
print("Longest token bytes:", longest_token)
print("Length in bytes:", len(longest_token))
print(
    "Decoded:",
    longest_token.decode("utf-8", errors="replace")
)