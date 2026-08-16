import pickle

from cs336_basics.tokenizer import Tokenizer

with open("tinystories_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("tinystories_merges.pkl", "rb") as f:
    merges = pickle.load(f)

tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=["|endoftext|"]
)

text = "Once upon a time, there was a little cat."
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print("Original:")
print(text)

print("\nToken IDs:")
print(ids)

for token_id in ids:
    token_bytes = vocab[token_id]

    print(
        token_id,
        repr(token_bytes),
        repr(token_bytes.decode("utf-8", errors="replace")),
    )

print("\nNumber of tokens:")
print(len(ids))

print("\nDecoded:")
print(decoded)

print("\nExact match:")
print(text == decoded)