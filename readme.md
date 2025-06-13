# 🏥 Locally Trained Medical LLM Chatbot with LoRA Fine-tuning

This project demonstrates how to:

✅ Fine-tune a large causal language model (like **Mistral-7B**) on custom medical instructions using **LoRA (Low Rank Adaptation)**.  
✅ Quantize the base model to 4-bit for efficient training and serving with **bitsandbytes**.  
✅ Provide a friendly **Chatbot UI with Gradio**, allowing you to chat with your fine-tuned medical LLM in real time.

---

## 🌟 Features

- **Model:** [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) — lightweight, powerful, and adaptable.
- **Training:** LoRA fine-tuning for efficient adaptation without touching base weights.
- **Quantization:** 4-bit (bitsandbytes) for reduced VRAM usage.
- **UI:** Gradio chatbot UI for interactive Q&A.
- **Domain:** Specialized for **clinical scenarios**, patient questions, and health advice.

---

## 🛠 Tech Stack

- **Transformers:** Hugging Face `transformers`
- **bitsandbytes:** 4-bit quantization
- **peft:** Parameter-Efficient Fine-Tuning with LoRA
- **Training:** `TrainingArguments` + `TrainingCollator`
- **Chatbot UI:** Gradio
- **Model:** [Mistral-7B-v0.1]

---

## 🔹 Installation

```bash
pip install -q transformers datasets accelerate peft bitsandbytes gradio
````

---

## 🔹 Hugging Face Authentication (optional)

To download weights from Hugging Face:

```python
from huggingface_hub import login
login(token="your_huggingface_token")
```

---

## 🔹 Quantization & LoRA Setup (quick view)

```python
model_id = "mistralai/Mistral-7B-v0.1"

# Quantize to 4-bit
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config,
    trust_remote_code=True
)

# Prepare LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'], task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_config)
```

---

## 🔹 Prepare Dataset & Tokenize

```python
dataset = Dataset.from_list(examples)

def format(example):
    prompt = f"""You are a licensed medical professional...
### Clinical Instruction:
{example['instruction']}
### Patient Data:
{example['input']}
### Response:
{example['output']}""" 
    return tokenizer(prompt, padding='max_length', truncation=True, max_length=512)

tokenized = dataset.map(format)
```

---

## 🔹 Fine-tune Model

```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_steps=10,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

trainier = Trainer(
    model=model,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=training_args
)

trainier.train()
```

---

## 🔹 Gradio UI (Chatbot)

```python
def generate_response(instruction, patient_input):
    prompt = f"""You are a licensed medical professional...
### Clinical Instruction:
{instruction}
### Patient Data:
{patient_input}
### Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[-1].strip()

chatbot_ui = gr.Interface(
    fn=generate_response,
    inputs=[gr.Textbox(), gr.Textbox()],
    outputs=gr.Textbox(), 
    title="🩺 Medical LLM Chatbot"
)

chatbot_ui.launch()
```

---

## 🔹 🔥 Additional Notes

* **Fine-tuning:** We applied **LoRA** to retain base knowledge while adding specialized knowledge.
* **bitsandbytes:** Enables **4-bit precision**, reducing VRAM usage.
* **Training:** The script performs **supervised fine-tuning** with custom patient scenarios.

---

## 🌟 Possible Applications

✅ **Medical Q\&A:** Provide tailored answers to patient questions.
✅ **Decision Support:** Assist health practitioners with diagnostics and treatment.
✅ **Chatbot:** Deploy to a hospital’s internal service or health platform.

---

## 📝 License

This code is for **research and education**.
Please use it responsibly and under applicable regulations.

