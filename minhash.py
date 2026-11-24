import hashlib
import re

import numpy as np

SEED = 1115
NUM_PERM = 128
PRIME = (1 << 31) - 1

_TOKEN = re.compile(r"[a-z0-9]+")
_HEXISH = re.compile(r"[0-9a-f]*[0-9][0-9a-f]*")
_DIGITS = re.compile(r"\d+")


def tokens(text, mask=False):
    toks = _TOKEN.findall(str(text).lower())
    if mask:
        toks = ["0" if _HEXISH.fullmatch(t) else _DIGITS.sub("0", t) for t in toks]
    return toks


def shingles(text, k=3, mask=False):
    toks = tokens(text, mask)
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def similarity(sig_a, sig_b):
    return float(np.mean(sig_a == sig_b))


def _base_hash(shingle):
    return int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "little") % PRIME


class MinHash:
    def __init__(self, num_perm=NUM_PERM, seed=SEED):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, PRIME, num_perm, dtype=np.uint64)[:, None]
        self.b = rng.integers(0, PRIME, num_perm, dtype=np.uint64)[:, None]

    def signature(self, items):
        sig = np.full(self.num_perm, PRIME, dtype=np.uint64)
        x = np.fromiter((_base_hash(s) for s in items), dtype=np.uint64, count=len(items))
        for i in range(0, len(x), 4096):
            sig = np.minimum(sig, ((self.a * x[i:i + 4096] + self.b) % PRIME).min(axis=1))
        return sig

    def text(self, text, mask=False):
        return self.signature(shingles(text, 3, mask))


class LSH:
    def __init__(self, bands=32, rows=4):
        self.bands, self.rows = bands, rows
        self.tables = [{} for _ in range(bands)]

    def _bands(self, sig):
        return [sig[i * self.rows:(i + 1) * self.rows].tobytes() for i in range(self.bands)]

    def add(self, key, sig):
        for table, band in zip(self.tables, self._bands(sig)):
            table.setdefault(band, []).append(key)

    def query(self, sig):
        found = {}
        for table, band in zip(self.tables, self._bands(sig)):
            for key in table.get(band, ()):
                found[key] = True
        return list(found)

    def probability(self, j):
        return 1 - (1 - j ** self.rows) ** self.bands
