# Agents, Tool Use, and RAG

LLMs are stateless text predictors. Agents extend them with memory, tools, and the ability to take actions in the world. RAG (Retrieval-Augmented Generation) gives them access to external knowledge without retraining.

---

## RAG: Retrieval-Augmented Generation

**The problem:** LLMs have a knowledge cutoff and fixed parametric memory. They hallucinate facts not in training data. Fine-tuning for new knowledge is expensive and unreliable.

**The solution:** retrieve relevant documents at query time and include them in the context.

```
User query → Retriever → Top-k documents
                ↓
User query + documents → LLM → Grounded response
```

### Retriever

**Dense retrieval (bi-encoder):** encode query and documents with the same encoder; find nearest neighbors in embedding space.

```python
from sentence_transformers import SentenceTransformer
import faiss

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Offline: build index
doc_embeddings = encoder.encode(documents)  # (N, d)
index = faiss.IndexFlatL2(d)
index.add(doc_embeddings)

# At query time
query_emb = encoder.encode([query])         # (1, d)
distances, indices = index.search(query_emb, k=5)
retrieved_docs = [documents[i] for i in indices[0]]
```

**Sparse retrieval (BM25):** classical TF-IDF style scoring. Fast, no GPU needed, no semantic understanding.

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1(1 - b + b \cdot |d|/\text{avgdl})}$$

**Hybrid retrieval:** combine dense and sparse scores (RRF — Reciprocal Rank Fusion). Often best in practice.

### Chunking strategy

Documents must be split into chunks before indexing. Tradeoffs:
- Small chunks (128 tokens): precise retrieval, may lack context
- Large chunks (512 tokens): more context, noisier retrieval

**Hierarchical chunking:** index small chunks, but retrieve parent chunk for context. Best of both worlds.

**Semantic chunking:** split at sentence boundaries where embedding similarity drops — natural topic boundaries.

### Reranking

The initial retrieval may return 20–50 candidates. A **cross-encoder** reranker scores (query, document) pairs jointly (more expensive but more accurate):

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, doc) for doc in retrieved])
top_docs = [retrieved[i] for i in scores.argsort()[-5:][::-1]]
```

### Generation with context

Feed retrieved documents in the LLM's context window:

```
System: Answer questions using only the provided context. 
        If the answer is not in the context, say "I don't know."

Context:
[Document 1]: ...
[Document 2]: ...

Question: {user_query}
```

**Faithfulness vs. helpfulness tension:** the model may know the answer from parametric memory but should ground its response in the retrieved context. Instruction design matters.

### Advanced RAG

**HyDE (Hypothetical Document Embeddings):** generate a hypothetical answer first, then retrieve using that answer's embedding rather than the original query embedding. Bridges the semantic gap between short queries and long documents.

**RAG-Fusion:** generate multiple query reformulations, retrieve for each, merge results with RRF. Reduces query sensitivity.

**Iterative RAG:** retrieve → generate partial answer → retrieve again based on what's still uncertain → finalize. Used in ReAct-style agents.

---

## Tool Use

LLMs can call external functions by generating structured text that the runtime parses and executes.

### Function calling format (OpenAI/Anthropic style)

Define available tools:
```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string"},
      "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"]
  }
}
```

The model generates a tool call (structured JSON), the runtime executes it, the result is fed back:

```
User: What's the weather in Paris?
Assistant: <tool_call>{"name": "get_weather", "args": {"location": "Paris"}}</tool_call>
[Runtime executes: API call → {"temp": 18, "condition": "cloudy"}]
Assistant: It's currently 18°C and cloudy in Paris.
```

**Training for tool use:** models learn to call tools from:
- SFT on tool-use demonstrations
- RLHF/RL where correct tool calls are rewarded
- Synthetic data: generate tool call scenarios at scale

### Common tools

| Tool | Purpose |
|------|---------|
| Web search | Current events, factual grounding |
| Code interpreter | Math, data analysis, plotting |
| Calculator | Exact arithmetic |
| File read/write | Long-term storage |
| API calls | Weather, databases, services |
| Browser | Dynamic web content |

---

## Agent architectures

An **agent** is a loop: observe → think → act → observe → ...

### ReAct (Yao et al., 2022)

Interleave **Re**asoning and **Act**ing in the same generation:

```
Thought: I need to find the population of France. Let me search.
Action: search("current population of France")
Observation: France has a population of approximately 68 million (2024).
Thought: Now I have the answer.
Action: finish("France's population is approximately 68 million.")
```

ReAct dramatically reduces hallucination on knowledge-intensive tasks vs. prompting alone.

### MRKL / Chain-of-Thought with Tools

Route subtasks to specialized tools (calculator for math, search for facts, code interpreter for computation). The LLM acts as a router and aggregator.

### Code as Action (Program-Aided Language models)

Instead of calling tools one by one, generate **code** that calls multiple tools, handles conditionals, and aggregates results:

```python
# LLM generates this code:
import requests
data = requests.get(f"https://api.weather.com/current?city={city}").json()
avg_temp = sum(d["temp"] for d in data["forecast"]) / len(data["forecast"])
print(f"Average forecast temp: {avg_temp:.1f}°C")
```

The code is executed in a sandboxed interpreter. The LLM sees the output and continues.

### Multi-agent systems

Multiple specialized agents collaborate:
- **Orchestrator agent:** plans and delegates
- **Executor agents:** specialized (coder, researcher, critic)
- **Critic agent:** reviews and corrects

```
User request → Orchestrator
    ├── Research agent → web search → summarize
    ├── Coder agent → write + run code
    └── Critic agent → review output
         ↓
    Final response
```

**Challenges:** error propagation (one agent's mistake cascades), coordination overhead, context window management across agents.

---

## Memory architectures

| Type | Description | Example |
|------|-------------|---------|
| In-context (working) | Documents in the current context | Retrieved chunks, conversation history |
| External vector store | Embeddings of past interactions | Long-term user memory |
| Episodic | Logs of past conversations | "Last week you asked about..." |
| Procedural | Learned behaviors via fine-tuning | Skills baked into model weights |

**Mem0 / MemGPT patterns:** summarize and compress old context into a structured memory store; retrieve relevant memories at the start of new conversations.

---

## RAG vs. long context vs. fine-tuning

| Approach | When to use | Tradeoff |
|----------|-------------|---------|
| RAG | Frequently updated, large, precise knowledge | Retrieval latency; quality depends on retriever |
| Long context (128k+) | Moderate corpus, complex reasoning across docs | Very expensive inference; context length limits |
| Fine-tuning / midtraining | Stable domain, style, behavioral changes | Training cost; knowledge can hallucinate |
| Hybrid | Mostly | Best quality at cost of complexity |

The emerging wisdom: use RAG for factual recall, long context for complex multi-document reasoning, and fine-tuning for style/behavior adaptation. Don't expect fine-tuning to reliably inject precise facts.
