# Word Vectors

> Before understanding where we are, we must understand where we've been.

__Core finding: a model can't just use words--words must become n-dimenisonal vectors__

_However, words can't be one-hot vectors, they must have some notion of closeness or similarity, etc._

## Word2Vec: Represent words using its context

Go through each position $t$ in text with center word $c$ and contect words $o$. 

Compute $P(W_{t+j} \mid W_t)$ and go over all windows and positions:

$$L(\theta) = \prod_{t=1}^T \prod_{-m\leq j \leq m} P(W_{t+j} \mid W_t; \theta)$$

Minimize the objective (negative log likelihood): $$J(\theta) = -\frac{1}{T}L(\theta) = -\frac{1}{T} \sum_{t=1}^T \sum_{-m\leq j \leq m} \log P(W_{t+j} \mid W_t; \theta)$$

How to compute $P(W_{t+j} \mid W_t; \theta)$? (Softmax) Use 2 vectors per word $w$: $v_w$ when $w$ is center and $u_w$ when $w$ contct.

$$P(o\mid c) = \frac{exp(u_o^Tv_c)}{\sum_{w\in V}exp(u_w^Tv_c)}$$

In practice, $U$ and $V$ are large embedding matrices and at each iteration, we update $\theta$ with GD.

In practice, this denominator is too costly so we use negative sampling. Sample over some $k$ words instead of all words.

### Two Word2Vec Variants:

1. Skip-Grams (SG): predice contect words (position independent) given contect (what we showed above).
2. Continuous Bag of Words: predict center from unordered bag of context clues.


## Other embedding techniques

GloVe focuses on vector differences. For example, ensuring the vector from man to woman is quivalent as the vector from king to queen.

## SVD embedding

Depending on rank, a high-dimensional matrix can be decomposed to a few components of maximal variance. Words can be embedded in these features. The matrix may have each word alongside occurrence counts in each document. This helps decide which words may frequently co-occur based on contexts.