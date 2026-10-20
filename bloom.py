"""Bloom filters and Bloom-filter-based context compression (COMP 480/580, HW1).

Two Bloom filter structures are implemented from scratch (no Bloom filter libraries):

* ``BloomFilter`` -- a classic single Bloom filter over a packed bit array.
* ``BitSlicedBloomIndex`` -- one Bloom filter per context chunk, all sharing the same
  R and k hash functions, stored *bit-sliced* (transposed): row ``p`` is a bit vector over
  the chunks whose filter has bit ``p`` set.  A membership query for one term against
  *all* chunks is then just the AND of k rows (k * C/8 bytes touched), instead of C
  separate filter probes.  This is the signature-file / BitFunnel idea.

Both use enhanced double hashing (Kirsch & Mitzenmacher; Dillinger & Manolios) derived from
one keyed 128-bit BLAKE2b digest per item, so k hash functions cost one hash computation.

``BloomContextCompressor`` uses the bit-sliced index to pick the chunks of a long context
that are relevant to a question; only those chunks are sent to the LLM.
"""

import hashlib
import math
import re
import time

import numpy as np


# --------------------------------------------------------------------------- hashing

def hash_positions(item, k, R, seed=0):
    """k bit positions in [0, R) for ``item`` via enhanced double hashing."""
    d = hashlib.blake2b(item.encode("utf-8"), digest_size=16,
                        key=seed.to_bytes(8, "little")).digest()
    h1 = int.from_bytes(d[:8], "little")
    h2 = int.from_bytes(d[8:], "little") | 1
    return [(h1 + i * h2 + (i * i * i - i) // 6) % R for i in range(k)]


def theoretical_fpr(n, R, k):
    """Standard approximation (1 - e^{-kn/R})^k."""
    if n == 0:
        return 0.0
    return (1.0 - math.exp(-k * n / R)) ** k


def optimal_k(R, n):
    return max(1, round(R / max(n, 1) * math.log(2)))


# --------------------------------------------------------------------------- filters

class BloomFilter:
    """A single Bloom filter with R bits and k hash functions."""

    def __init__(self, R, k, seed=0):
        self.R, self.k, self.seed = R, k, seed
        self.bits = np.zeros((R + 7) // 8, dtype=np.uint8)
        self.n = 0

    def add(self, item):
        for p in hash_positions(item, self.k, self.R, self.seed):
            self.bits[p >> 3] |= np.uint8(1 << (p & 7))
        self.n += 1

    def __contains__(self, item):
        return all(self.bits[p >> 3] & (1 << (p & 7))
                   for p in hash_positions(item, self.k, self.R, self.seed))

    @property
    def nbytes(self):
        return self.bits.nbytes

    def fill_ratio(self):
        return int(np.unpackbits(self.bits)[: self.R].sum()) / self.R


class BitSlicedBloomIndex:
    """C Bloom filters (one per chunk) of R bits each, stored as an R x C bit matrix."""

    def __init__(self, R, k, n_docs, seed=0):
        self.R, self.k, self.n_docs, self.seed = R, k, n_docs, seed
        self.slices = np.zeros((R, (n_docs + 7) // 8), dtype=np.uint8)
        self._pos_cache = {}  # transient hash cache; not part of the stored index

    def positions(self, item):
        pos = self._pos_cache.get(item)
        if pos is None:
            pos = np.array(hash_positions(item, self.k, self.R, self.seed), dtype=np.int64)
            self._pos_cache[item] = pos
        return pos

    def add_many(self, doc, items):
        """Insert all ``items`` into the filter of chunk ``doc``."""
        if not items:
            return
        rows = np.concatenate([self.positions(t) for t in items])
        self.slices[rows, doc >> 3] |= np.uint8(1 << (doc & 7))

    def query(self, item):
        """Boolean vector over chunks: does chunk c's filter (possibly falsely) contain item?"""
        word = np.bitwise_and.reduce(self.slices[self.positions(item)], axis=0)
        return np.unpackbits(word, bitorder="little")[: self.n_docs].astype(bool)

    def contains(self, doc, item):
        return bool(np.all(self.slices[self.positions(item), doc >> 3] & (1 << (doc & 7))))

    def clear_cache(self):
        self._pos_cache.clear()

    @property
    def nbytes(self):
        return self.slices.nbytes


class ExactIndex:
    """Oracle with no false positives (Python sets); used only as a reference point."""

    def __init__(self, n_docs):
        self.n_docs = n_docs
        self.sets = [set() for _ in range(n_docs)]

    def add_many(self, doc, items):
        self.sets[doc].update(items)

    def query(self, item):
        return np.array([item in s for s in self.sets], dtype=bool)

    @property
    def nbytes(self):
        import sys
        return sum(sys.getsizeof(s) + sum(sys.getsizeof(t) for t in s) for s in self.sets)


# --------------------------------------------------------------------------- text

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours ourselves out
over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves also many much whose
""".split())

_NUM_COMMA = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOKEN = re.compile(r"[^\W_]+(?:\.\d+)?")


def _stem(t):
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and not t[-2].isdigit():
        return t[:-1]
    return t


def tokenize(text):
    text = _NUM_COMMA.sub("", text.lower())
    return [_stem(t) for t in _TOKEN.findall(text)]


def features(text):
    """Unigrams + adjacent-content-word bigrams (stopwords dropped)."""
    content = [t for t in tokenize(text) if t not in STOPWORDS]
    feats = set(content)
    feats.update(f"{a}_{b}" for a, b in zip(content, content[1:]))
    return feats


_LIST_ITEM = re.compile(r"^\s*(?:\d{1,4}[.)]|[-*•])\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


def _pack(pieces, sep, max_words):
    out, cur, cur_w = [], [], 0
    for p in pieces:
        w = len(p.split())
        if cur and cur_w + w > max_words:
            out.append(sep.join(cur))
            cur, cur_w = [], 0
        cur.append(p)
        cur_w += w
    if cur:
        out.append(sep.join(cur))
    return out


def _split_long(block, max_words):
    if len(block.split()) <= max_words:
        return [block]
    out = []
    for piece in _pack(_SENT_SPLIT.split(block), " ", max_words):
        if len(piece.split()) <= max_words:
            out.append(piece)
            continue
        for sub in _pack(piece.split("\n"), "\n", max_words):  # long tables / line runs
            words = sub.split()
            if len(words) <= max_words:
                out.append(sub)
            else:  # last resort: hard word split
                out.extend(" ".join(words[i:i + max_words])
                           for i in range(0, len(words), max_words))
    return out


def split_chunks(text, max_words=80, min_words=15):
    """Split into paragraph / list-item blocks, cut long ones at sentence boundaries,
    and glue very short blocks (headings, captions) onto the block that follows."""
    blocks, cur = [], []
    for line in text.split("\n"):
        if not line.strip() or _LIST_ITEM.match(line):
            if cur:
                blocks.append("\n".join(cur))
            cur = [line] if line.strip() else []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))

    pieces = [p for b in blocks for p in _split_long(b, max_words) if p.strip()]
    chunks, carry = [], ""
    for p in pieces:
        p = carry + "\n" + p if carry else p
        if len(p.split()) < min_words:
            carry = p
        else:
            chunks.append(p)
            carry = ""
    if carry:
        if chunks:
            chunks[-1] += "\n" + carry
        else:
            chunks.append(carry)
    return chunks


# --------------------------------------------------------------------------- compressor

class BloomContextCompressor:
    """Index a context once, then compress it per question.

    Retrieval: every chunk's terms go into that chunk's Bloom filter.  For a question, each
    query term is checked against all chunk filters at once (bit-sliced AND).  The number of
    positive chunks gives a (false-positive inflated) document frequency, from which an IDF
    weight is computed; a chunk's score is the IDF-weighted number of query terms it
    (apparently) contains.  An optional second hop (off by default) adds the rarest new terms from the best
    chunks (bridge entities for multi-hop questions).  All chunks scoring at least
    ``alpha * max_score`` are kept, plus the neighbours of the top chunks and the
    document's opening chunk, in original order.
    """

    def __init__(self, text, R=512, k=4, seed=0, max_words=80, min_words=15, exact=False):
        t0 = time.perf_counter()
        self.chunks = split_chunks(text, max_words, min_words)
        C = len(self.chunks)
        self.chunk_feats = [features(c) for c in self.chunks]
        self.index = ExactIndex(C) if exact else BitSlicedBloomIndex(R, k, C, seed)
        for i, f in enumerate(self.chunk_feats):
            self.index.add_many(i, f)
        self.build_time = time.perf_counter() - t0
        self.R, self.k, self.exact = R, k, exact
        self.n_items = np.array([len(f) for f in self.chunk_feats])
        # Chunk features are kept only so experiments can measure FPR exactly;
        # retrieval itself touches only the Bloom index (and chunk text for hop 2).

    @property
    def n_chunks(self):
        return len(self.chunks)

    @property
    def index_bytes(self):
        return self.index.nbytes

    def _score(self, terms, weights):
        C = self.n_chunks
        hits = np.stack([self.index.query(t) for t in terms]) if terms else np.zeros((0, C), bool)
        df = hits.sum(axis=1)
        idf = np.log(1.0 + (C - df + 0.5) / (df + 0.5))
        return (np.asarray(weights) * idf) @ hits, hits, idf

    def compress(self, question, alpha=0.15, neighbors=1, n_expand=3, keep_head=1,
                 hops=1, hop_terms=4, hop_weight=0.5):
        t0 = time.perf_counter()
        terms = sorted(features(question))
        weights = [1.0] * len(terms)
        score, hits, idf = self._score(terms, weights)

        if hops >= 2 and score.max() > 0:
            # Bridge terms: rarest (by Bloom-estimated df) unigrams of the 2 best chunks.
            qset = set(terms)
            cands = {t for c in np.argsort(-score, kind="stable")[:2]
                     for t in self.chunk_feats[c] if "_" not in t and t not in qset}
            cands = sorted(cands)
            if cands:
                df = np.array([self.index.query(t).sum() for t in cands])
                extra = [cands[i] for i in np.argsort(df, kind="stable")[:hop_terms]]
                terms = terms + extra
                weights = weights + [hop_weight] * len(extra)
                score, hits, idf = self._score(terms, weights)

        C = self.n_chunks
        if score.max() <= 0:
            keep = np.ones(C, bool)  # nothing matched: fall back to the full context
        else:
            keep = score >= alpha * score.max()
            for c in np.argsort(-score, kind="stable")[:n_expand]:
                keep[max(0, c - neighbors): c + neighbors + 1] = True
            keep[:keep_head] = True

        parts, prev = [], -1
        for c in np.flatnonzero(keep):
            if prev >= 0 and c != prev + 1:
                parts.append("[...]")
            parts.append(self.chunks[c])
            prev = c
        text = "\n\n".join(parts)
        info = {
            "retrieval_time": time.perf_counter() - t0,
            "chunks_kept": int(keep.sum()),
            "chunks_total": C,
            "terms": terms,
            "positives": hits.sum(axis=1).tolist(),
        }
        return text, info
