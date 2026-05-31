# Language Modeling

A comprehensive guide to modern large language models, from architecture fundamentals through training, inference, and alignment. Inspired by Stanford CS224N and CS336.

## Contents

| File | Topics |
|------|--------|
| [tokenization.md](tokenization.md) | BPE, WordPiece, SentencePiece, vocab tradeoffs |
| [architecture.md](architecture.md) | Transformer deep dive, positional encodings, attention math |
| [attention_variants.md](attention_variants.md) | MHA, GQA, MQA, sparse, linear attention, MoE, SSMs |
| [pretraining.md](pretraining.md) | Objectives, data pipelines, scaling laws, training stability |
| [systems_and_hardware.md](systems_and_hardware.md) | GPU architecture, memory hierarchy, roofline model, Triton |
| [parallelism.md](parallelism.md) | Data, tensor, pipeline, sequence parallelism; ZeRO/FSDP |
| [flash_attention.md](flash_attention.md) | Tiling, online softmax, FlashAttention 1/2/3 |
| [inference.md](inference.md) | KV caching, speculative decoding, continuous batching |
| [quantization.md](quantization.md) | INT8/INT4, PTQ, QAT, GPTQ, AWQ |
| [scaling_laws.md](scaling_laws.md) | Kaplan, Chinchilla, inference-time compute, emergence |
| [midtraining.md](midtraining.md) | Domain adaptation, continued pretraining, data mixing |
| [posttraining.md](posttraining.md) | SFT, RLHF, DPO, GRPO, reward modeling |
| [peft.md](peft.md) | LoRA, QLoRA, adapters, prompt/prefix tuning |
| [evaluation.md](evaluation.md) | Perplexity, benchmarks, LLM-as-judge |
| [agents_and_rag.md](agents_and_rag.md) | Tool use, RAG, agentic frameworks |
| [reasoning_models.md](reasoning_models.md) | Chain-of-thought, RLVR, o1/DeepSeek-R1 style training |

## Reading order

**From scratch:** tokenization → architecture → attention_variants → pretraining → scaling_laws → posttraining

**Systems track:** systems_and_hardware → parallelism → flash_attention → inference → quantization

**Alignment track:** posttraining → peft → reasoning_models → evaluation
