---
library_name: transformers
license: apache-2.0
language:
- en
tags:
- fill-mask
- masked-lm
- long-context
- modernbert
pipeline_tag: fill-mask
inference: false
---

# ModernBERT

## Table of Contents
1. [Model Summary](#model-summary)
2. [Usage](#Usage)
3. [Evaluation](#Evaluation)
4. [Limitations](#limitations)
5. [Training](#training)
6. [License](#license)
7. [Citation](#citation)

## Model Summary

ModernBERT is a modernized bidirectional encoder-only Transformer model (BERT-style) pre-trained on 2 trillion tokens of English and code data with a native context length of up to 8,192 tokens. ModernBERT leverages recent architectural improvements such as:

- **Rotary Positional Embeddings (RoPE)** for long-context support.  
- **Local-Global Alternating Attention** for efficiency on long inputs.  
- **Unpadding and Flash Attention** for efficient inference.  

ModernBERT’s native long context length makes it ideal for tasks that require processing long documents, such as retrieval, classification, and semantic search within large corpora. The model was trained on a large corpus of text and code, making it suitable for a wide range of downstream tasks, including code retrieval and hybrid (text + code) semantic search.

It is available in the following sizes:

- [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) - 22 layers, 149 million parameters
- [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) - 28 layers, 395 million parameters

For more information about ModernBERT, we recommend our [release blog post](https://huggingface.co/blog/modernbert) for a high-level overview, and our [arXiv pre-print](https://arxiv.org/abs/2412.13663) for in-depth information.

*ModernBERT is a collaboration between [Answer.AI](https://answer.ai), [LightOn](https://lighton.ai), and friends.*

## Usage

You can use these models directly with the `transformers` library starting from v4.48.0:

```sh
pip install -U transformers>=4.48.0
```

Since ModernBERT is a Masked Language Model (MLM), you can use the `fill-mask` pipeline or load it via `AutoModelForMaskedLM`. To use ModernBERT for downstream tasks like classification, retrieval, or QA, fine-tune it following standard BERT fine-tuning recipes.

**⚠️ If your GPU supports it, we recommend using ModernBERT with Flash Attention 2 to reach the highest efficiency. To do so, install Flash Attention as follows, then use the model as normal:**

```bash
pip install flash-attn
```

Using `AutoModelForMaskedLM`:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# To get predictions for the mask:
masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print("Predicted token:", predicted_token)
# Predicted token:  Paris
```

Using a pipeline:

```python
import torch
from transformers import pipeline
from pprint import pprint

pipe = pipeline(
    "fill-mask",
    model="answerdotai/ModernBERT-base",
    torch_dtype=torch.bfloat16,
)

input_text = "He walked to the [MASK]."
results = pipe(input_text)
pprint(results)
```

**Note:** ModernBERT does not use token type IDs, unlike some earlier BERT models. Most downstream usage is identical to standard BERT models on the Hugging Face Hub, except you can omit the `token_type_ids` parameter.

## Evaluation

We evaluate ModernBERT across a range of tasks, including natural language understanding (GLUE), general retrieval (BEIR), long-context retrieval (MLDR), and code retrieval (CodeSearchNet and StackQA).

**Key highlights:**
- On GLUE, ModernBERT-base surpasses other similarly-sized encoder models, and ModernBERT-large is second only to Deberta-v3-large.
- For general retrieval tasks, ModernBERT performs well on BEIR in both single-vector (DPR-style) and multi-vector (ColBERT-style) settings.
- Thanks to the inclusion of code data in its training mixture, ModernBERT as a backbone also achieves new state-of-the-art code retrieval results on CodeSearchNet and StackQA.

### Base Models

| Model       | IR (DPR)     | IR (DPR)     | IR (DPR)     | IR (ColBERT)  | IR (ColBERT)  | NLU  | Code | Code |
|-------------|--------------|--------------|--------------|---------------|---------------|------|------|------|
|             | BEIR         | MLDR_OOD     | MLDR_ID      | BEIR          | MLDR_OOD      | GLUE | CSN  | SQA  |
| BERT        | 38.9         | 23.9         | 32.2         | 49.0          | 28.1          | 84.7 | 41.2 | 59.5 |
| RoBERTa     | 37.7         | 22.9         | 32.8         | 48.7          | 28.2          | 86.4 | 44.3 | 59.6 |
| DeBERTaV3   | 20.2         | 5.4          | 13.4         | 47.1          | 21.9          | 88.1 | 17.5 | 18.6 |
| NomicBERT   | 41.0         | 26.7         | 30.3         | 49.9          | 61.3          | 84.0 | 41.6 | 61.4 |
| GTE-en-MLM  | 41.4         | **34.3**    |**44.4**   | 48.2          | 69.3          | 85.6 | 44.9 | 71.4 |
| ModernBERT  | **41.6**    | 27.4         | 44.0         | **51.3**    | **80.2**      | **88.4** | **56.4** |**73.6**|

---

### Large Models

| Model       | IR (DPR)     | IR (DPR)     | IR (DPR)     | IR (ColBERT)  | IR (ColBERT)  | NLU  | Code | Code |
|-------------|--------------|--------------|--------------|---------------|---------------|------|------|------|
|             | BEIR         | MLDR_OOD     | MLDR_ID      | BEIR          | MLDR_OOD      | GLUE | CSN  | SQA  |
| BERT        | 38.9         | 23.3         | 31.7         | 49.5          | 28.5          | 85.2 | 41.6 | 60.8 |
| RoBERTa     | 41.4         | 22.6         | 36.1         | 49.8          | 28.8          | 88.9 | 47.3 | 68.1 |
| DeBERTaV3   | 25.6         | 7.1          | 19.2         | 46.7          | 23.0          | **91.4**| 21.2 | 19.7 |
| GTE-en-MLM  | 42.5         | **36.4**    | **48.9**  | 50.7          | 71.3          | 87.6 | 40.5 | 66.9 |
| ModernBERT  | **44.0**    | 34.3         | 48.6         | **52.4**     | **80.4**     | 90.4 |**59.5** |**83.9**|

*Table 1: Results for all models across an overview of all tasks. CSN refers to CodeSearchNet and SQA to StackQA. MLDRID refers to in-domain (fine-tuned on the training set) evaluation, and MLDR_OOD to out-of-domain.*

ModernBERT’s strong results, coupled with its efficient runtime on long-context inputs, demonstrate that encoder-only models can be significantly improved through modern architectural choices and extensive pretraining on diversified data sources.


## Limitations

ModernBERT’s training data is primarily English and code, so performance may be lower for other languages. While it can handle long sequences efficiently, using the full 8,192 tokens window may be slower than short-context inference. Like any large language model, ModernBERT may produce representations that reflect biases present in its training data. Verify critical or sensitive outputs before relying on them.

## Training

- Architecture: Encoder-only, Pre-Norm Transformer with GeGLU activations.
- Sequence Length: Pre-trained up to 1,024 tokens, then extended to 8,192 tokens.
- Data: 2 trillion tokens of English text and code.
- Optimizer: StableAdamW with trapezoidal LR scheduling and 1-sqrt decay.
- Hardware: Trained on 8x H100 GPUs.

See the paper for more details.

## License

We release the ModernBERT model architectures, model weights, training codebase under the Apache 2.0 license.

## Citation

If you use ModernBERT in your work, please cite:

```
@misc{modernbert,
      title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference}, 
      author={Benjamin Warner and Antoine Chaffin and Benjamin Clavié and Orion Weller and Oskar Hallström and Said Taghadouini and Alexis Gallagher and Raja Biswas and Faisal Ladhak and Tom Aarsen and Nathan Cooper and Griffin Adams and Jeremy Howard and Iacopo Poli},
      year={2024},
      eprint={2412.13663},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13663}, 
}
```