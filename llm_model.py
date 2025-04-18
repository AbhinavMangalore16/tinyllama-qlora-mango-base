# 1. Install dependencies (You can skip this in .py or use requirements.txt instead)
# pip install -q accelerate peft bitsandbytes transformers trl sentencepiece triton huggingface_hub

import torch
import shutil
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, pipeline
)
from datasets import load_dataset
from peft import (
    LoraConfig, prepare_model_for_kbit_training,
    get_peft_model, AutoPeftModelForCausalLM
)
from trl import SFTTrainer
from huggingface_hub import login, create_repo, upload_folder

# 2. Load and format dataset
def format_prompt(ex):
    chat = ex["messages"]
    return {"text": tokenizer.apply_chat_template(chat, tokenize=False)}

print("🔄 Loading dataset...")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
dataset = dataset.shuffle(seed=42).select(range(3000))

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
dataset = dataset.map(format_prompt)

# 3. Load base model with 4-bit quantization
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("🧠 Loading quantized model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 4. Apply QLoRA using PEFT
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj"
    ]
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# 5. Define training arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True
)

# 6. Train the model
print("🚀 Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512,
    peft_config=peft_config,
)
trainer.train()

# 7. Save and merge model
print("💾 Saving merged model...")
trainer.model.save_pretrained("TinyLlama-1.1B-qlora-mango")
model = AutoPeftModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-qlora-mango",
    low_cpu_mem_usage=True,
    device_map="auto"
)
merged_model = model.merge_and_unload()

#8. Testing text generation
prompt = """<|user|>
Tell me something about mangoes.</s>
<|assistant|>"""

pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer)
print("📜 Sample Output:")
print(pipe(prompt, max_new_tokens=100)[0]["generated_text"])