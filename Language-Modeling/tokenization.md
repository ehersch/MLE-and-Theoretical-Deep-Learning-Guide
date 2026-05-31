# Tokenization

Tokenization is the process of converting raw text into discrete tokens that a model can process. The choice of tokenizer profoundly affects model capacity, multilingual coverage, and arithmetic reasoning — it is not a neutral preprocessing step.

## Why not characters or words?

**Character-level:** sequences get very long (hundreds of tokens per sentence), making attention expensive and long-range dependencies harder to learn.

**Word-level:** vocabulary explosion for morphologically rich languages; out-of-vocabulary tokens for any unseen word; can't share representations between "run", "running", "runner".

**Subword tokenization** splits at a learned granularity: frequent words stay whole, rare words decompose. You get bounded vocabulary size, graceful handling of novel words, and shared morphological structure.

---

## Byte Pair Encoding (BPE)

Originally a data compression algorithm. The idea: iteratively merge the most frequent adjacent pair of symbols.

**Algorithm:**

```
Initialize vocab = set of all characters + <EOW>
corpus = words split into characters

repeat N times:
    count all adjacent symbol pairs across corpus
    merge the most frequent pair into a new symbol
    update corpus with the merge
```

**Example:**

```
Corpus: "low low low lower lower newest newest"
Initial: l o w </w>  l o w e r </w>  n e w e s t </w>

Step 1: most frequent pair → (e, s) → merge to "es"
Step 2: most frequent pair → (es, t) → merge to "est"
Step 3: most frequent pair → (l, o) → merge to "lo"
...
```

After enough merges, common words become single tokens. BPE is greedy and produces deterministic tokenization.

**GPT-2/GPT-4 use BPE over bytes** (not Unicode code points), so it can tokenize any byte sequence without UNK tokens. The base vocabulary is 256 bytes, then merges expand it.

```python
# Simplified BPE merge step
from collections import Counter

def get_pairs(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    return {word.replace(bigram, replacement): freq 
            for word, freq in vocab.items()}
```

---

## WordPiece

Used by BERT. Similar to BPE but the merge criterion is **likelihood** rather than raw frequency:

$$\text{score}(A, B) = \frac{P(AB)}{P(A) \cdot P(B)}$$

This favors merging pairs that co-occur much more than chance — it's a pointwise mutual information (PMI) criterion. Tokens for subwords are prefixed with `##` to indicate continuation: `"running"` → `["run", "##ning"]`.

**Key difference from BPE:** WordPiece scores consider the full language model likelihood; BPE is purely frequency-based.

---

## SentencePiece / Unigram Language Model

Used by LLaMA, T5, Gemma. Works directly on raw unicode (no pre-tokenization), making it language-agnostic.

**Unigram LM tokenizer:**
- Start with a large candidate vocabulary
- Assign each token a probability $p(x_i)$
- The probability of a segmentation $\mathbf{x} = (x_1, \ldots, x_n)$ is:
$$P(\mathbf{x}) = \prod_{i=1}^n p(x_i)$$
- Find the most probable segmentation via Viterbi
- Iteratively prune tokens whose removal minimally reduces corpus likelihood

**Sampling:** SentencePiece supports sampling multiple segmentations during training, which regularizes the model against tokenization artifacts.

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe',  # or 'unigram'
    character_coverage=0.9995,
)

sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
sp.encode("Hello world", out_type=str)
# ['▁Hello', '▁world']  (▁ marks word start)
```

---

## Vocabulary size tradeoffs

| Vocab size | Pros | Cons |
|------------|------|------|
| Small (~8k) | Short sequences, good generalization | Many splits, poor number/name handling |
| Medium (~32–64k) | Balance; used by most LLMs | — |
| Large (~100k+) | Fewer splits, better multilingual | Large embedding tables, slower softmax |

**Embedding table cost:** vocab size $V$, hidden dim $d$ → $V \times d$ parameters. At $V=128{,}000$, $d=4096$: ~500M params just for embeddings.

**Arithmetic:** Numbers are notoriously badly tokenized. "1234" might be `["12", "34"]` or `["1", "2", "3", "4"]` depending on the tokenizer. This makes digit-by-digit arithmetic hard. Some models (e.g., Llama 3) explicitly tokenize each digit separately.

---

## Tokenization artifacts and pathologies

- **Whitespace sensitivity:** `" token"` and `"token"` are often different tokens (the leading space is significant in BPE over bytes).
- **Case sensitivity:** capitalization often produces different token IDs.
- **Non-English languages:** BPE trained on English-dominated corpora will over-fragment non-English text, using more tokens per word and effectively giving those languages less model capacity per character.
- **The "solid gold magikarp" problem:** tokens can appear in the vocabulary but have near-zero frequency in training data. The embedding for such tokens is undertrained and can cause erratic model behavior.

---

## Tiktoken (OpenAI)

OpenAI's fast BPE tokenizer. Key features: written in Rust for speed, operates on bytes, supports encoding special tokens like `<|endoftext|>`.

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
tokens = enc.encode("Hello, world!")
# [9906, 11, 1917, 0]
enc.decode(tokens)
# 'Hello, world!'
```

---

## Summary

| Tokenizer | Used by | Algorithm | Base unit |
|-----------|---------|-----------|-----------|
| BPE (byte-level) | GPT-2/3/4, LLaMA 2 | Frequency merges | Bytes |
| WordPiece | BERT, DistilBERT | PMI merges | Characters |
| Unigram LM | T5, LLaMA 3, Gemma | Probabilistic pruning | Unicode |
| Tiktoken BPE | GPT-4, o1 | Frequency merges | Bytes |
