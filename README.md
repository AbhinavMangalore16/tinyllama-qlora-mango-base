# 🥭 TinyLlama-QLoRA-Mango

A fine-tuned version of the [TinyLlama-1.1B](https://huggingface.co/cognitivecomputations/TinyLlama-1.1B) model using [QLoRA](https://arxiv.org/abs/2305.14314), tailored for conversational generation on mango-related prompts.

🔗 **Hugging Face Model Repo**: [abhinavm16104/TinyLlama-1.1B-qlora-mango](https://huggingface.co/abhinavm16104/TinyLlama-1.1B-qlora-mango)

---

## 📌 Description

This model is a compact and efficient conversational language model based on TinyLlama-1.1B, fine-tuned using QLoRA for low-resource adaptation. It demonstrates enhanced generation quality for domain-specific queries, particularly fruit-related ones like **mangoes** 🥭.

## 🧠 Training Details

- **Base model**: [TinyLlama-1.1B](https://huggingface.co/cognitivecomputations/TinyLlama-1.1B)
- **Fine-tuning method**: [QLoRA (Quantized LoRA)](https://arxiv.org/abs/2305.14314)
- **Precision**: 4-bit
- **Tokenizer**: Same as the base model

## 📦 Usage

```python
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("abhinavm16104/TinyLlama-1.1B-qlora-mango")
tokenizer = AutoTokenizer.from_pretrained("abhinavm16104/TinyLlama-1.1B-qlora-mango")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = """<|user|>
Tell me something about mangoes.</s>
<|assistant|>"""

print(pipe(prompt, max_new_tokens=100)[0]['generated_text'])
