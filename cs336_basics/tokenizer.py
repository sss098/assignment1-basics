import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        self.bytes_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in self.bytes_to_id:
                self.vocab[next_id] = token_bytes
                self.bytes_to_id[token_bytes] = next_id
                next_id += 1

        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        import pickle

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens=special_tokens)

    def decode(self, ids: list[int]) -> str:
        """
        将 token ids 转换为字符串。
        """
        token_bytes = []

        for token_id in ids:
            token_bytes.append(self.vocab[token_id])

        all_bytes = b"".join(token_bytes)

        return all_bytes.decode("utf-8", errors="replace")

    def _apply_merges(self, pieces: list[bytes]) -> list[bytes]:
        """
        对一个 pre-token 的 bytes pieces 应用 BPE merges。

        例如:
        pieces = [b"t", b"h", b"e"]
        merges = [(b"t", b"h"), (b"th", b"e")]

        返回:
        [b"the"]
        """
        while True:
            best_pair = None
            best_rank = None

            for i in range(len(pieces) -1):
                pair = (pieces[i], pieces[i+1])

                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]

                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = pair


            if best_pair is None:
                break

            new_pieces = []
            i = 0

            while i < len(pieces):
                if(
                    i < len(pieces) - 1
                    and pieces[i] == best_pair[0]
                    and pieces[i+1] == best_pair[1]
                ):
                    new_pieces.append(pieces[i] + pieces[i+1])
                    i += 2
                else:
                    new_pieces.append(pieces[i])
                    i += 1

            pieces = new_pieces

        return pieces

    def encode_ordinary(self, text: str) -> list[int]:
        token_ids = []

        # token_bytes = text.encode("utf-8")
        # pieces = [bytes([b]) for b in token_bytes]

        # pieces = self._apply_merges(pieces)

        for match in re.finditer(PAT, text):
            token = match.group(0)

            token_bytes = token.encode("utf-8")
            pieces = [bytes([b]) for b in token_bytes]

            pieces = self._apply_merges(pieces)


            for piece in pieces:
                token_ids.append(self.bytes_to_id[piece])

        return token_ids

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self.encode_ordinary(text)

        token_ids = []

        short_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in short_special_tokens]
        special_pattern = "(" + "|".join(escaped_tokens) + ")"

        chunks = re.split(special_pattern, text)

        for chunk in chunks:
            if chunk == "":
                continue

            if chunk in self.special_tokens:
                token_bytes = chunk.encode("utf-8")
                token_ids.append(self.bytes_to_id[token_bytes])
            else:
                token_ids.extend(self.encode_ordinary(chunk))

        return token_ids

    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id


if __name__ == "__main__":
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    tokenizer = Tokenizer(vocab, merges)

    text = tokenizer.decode([104, 101, 108, 108, 111])
    print(text)
