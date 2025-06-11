#possible models to train locally and finetune and use without api 
#llama 2 7b, mistral 7b, tinyllama 1.1b, gpt2,gptneo
#transformers used for inference and training
#lora fine tuning 
#gradio/streamlit chatbot + ui
#huggingface phind cerebras has og,finetuned,code-based,small trainable models
#using mistral 7b as that accepts lora finetuning and transformers
#so initally we just get the pretrained model weights, we train our dataset with it, and do fine tuning with lora without totally destroying the the model
#undergoes prompt formatting,tokenization,forward pass,comaprision, backprop
#Each token passes through layers of attention and feed-forward logic,It learns which parts of the prompt are most important through Transformers 
#LoRA adds tiny low-rank matrices that adapt the model without changing the base weights
from huggingface_hub import login
login(token="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")
!pip install -q transformers datasets accelerate peft bitsandbytes gradio
!nvidia-smi
# ========================
# 🔬 Locally-Loaded Healthcare Chatbot with LoRA + Mistral
# ========================

# ========================
# 🚀 Load Quantized Mistral Model with BitsAndBytesConfig
# ========================
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "mistralai/Mistral-7B-v0.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Fix missing pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# BitsAndBytes quant config for 4-bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=quant_config
)

# ========================
# 🩺 Apply LoRA for Medical Fine-Tuning
# ========================
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========================
# 🧬 Medical Prompt Dataset
# ========================
from datasets import Dataset

examples = [
    {"instruction": "Provide a possible diagnosis", "input": "Patient complains of chest pain radiating to left arm, shortness of breath, and sweating.", "output": "Possible diagnosis: Acute myocardial infarction. Recommend immediate ECG and cardiac enzymes."},
    {"instruction": "Suggest treatment options", "input": "35-year-old with newly diagnosed type 2 diabetes.", "output": "Recommend lifestyle changes, metformin as first-line medication, and monitor A1C regularly."},
    {"instruction": "Explain lab result", "input": "TSH level is 8.5 mIU/L", "output": "Indicates hypothyroidism. Consider levothyroxine therapy after confirming with T3 and T4 levels."},
    {"instruction": "Recommend next steps", "input": "70-year-old male, smoker, presents with hemoptysis.", "output": "Order chest X-ray/CT scan to rule out lung cancer or TB. Refer to pulmonology."},
    {"instruction": "Answer patient query", "input": "Can I take aspirin with warfarin?", "output": "Combining both increases bleeding risk. Only use together if medically justified and under supervision."},
    {"instruction": "Suggest medication", "input": "Patient with high LDL and no cardiac history.", "output": "Recommend statin therapy like atorvastatin 10–20 mg to lower LDL levels."},
    {"instruction": "Provide alternatives", "input": "Patient has type 2 diabetes and is intolerant to metformin.", "output": "Consider alternatives like DPP-4 inhibitors (e.g., sitagliptin) or SGLT2 inhibitors."},
    {"instruction": "Explain side effects", "input": "Patient started on losartan.", "output": "May cause dizziness, elevated potassium, and renal function decline. Monitor labs."},
    {"instruction": "Recommend antiplatelet therapy", "input": "Post-angioplasty patient discharged home.", "output": "Recommend aspirin + clopidogrel for 6–12 months as dual antiplatelet therapy."},
    {"instruction": "Provide a possible diagnosis", "input": "Patient presents with tremors, anxiety, and weight loss.", "output": "Possible hyperthyroidism. Recommend thyroid panel for confirmation."},
    {"instruction": "Suggest treatment options", "input": "Patient diagnosed with Helicobacter pylori infection.", "output": "Triple therapy: PPI + clarithromycin + amoxicillin/metronidazole for 14 days."},
    {"instruction": "Explain lab result", "input": "Serum creatinine increased from 1.2 to 2.0 mg/dL.", "output": "Suggests acute kidney injury. Investigate causes like dehydration or nephrotoxic drugs."},
    {"instruction": "Recommend next steps", "input": "Young female with new-onset seizures.", "output": "Order brain MRI and EEG. Consider neurology referral."},
    {"instruction": "Answer patient query", "input": "Do statins affect memory?", "output": "Rarely, statins may cause reversible cognitive effects. Discuss risk-benefit with physician."},
    {"instruction": "Suggest medication", "input": "Hypertensive diabetic patient needing BP control.", "output": "Consider ACE inhibitor like enalapril for both BP and renal protection."},
    {"instruction": "Provide alternatives", "input": "Patient allergic to ACE inhibitors.", "output": "Use ARBs like losartan or valsartan as alternatives."},
    {"instruction": "Explain side effects", "input": "Patient on amitriptyline for chronic pain.", "output": "Watch for drowsiness, dry mouth, weight gain, and QT prolongation."},
    {"instruction": "Recommend drug class", "input": "Asthmatic child with frequent night-time symptoms.", "output": "Use inhaled corticosteroids for long-term control. Consider montelukast if allergic."},
    {"instruction": "Explain drug mechanism", "input": "How does metformin work?", "output": "Metformin decreases hepatic glucose production and increases insulin sensitivity."},
    {"instruction": "Discuss drug interaction", "input": "Patient takes simvastatin and clarithromycin.", "output": "Dangerous interaction. Risk of rhabdomyolysis. Avoid combining."},
    {"instruction": "Provide a possible diagnosis", "input": "Elderly patient with memory loss, apathy, and difficulty performing tasks.", "output": "Possible diagnosis: Alzheimer's disease. Recommend cognitive testing and MRI."},
    {"instruction": "Suggest treatment options", "input": "Child with fever, sore throat, and no cough.", "output": "Consider strep pharyngitis. Prescribe amoxicillin or penicillin after rapid strep test."},
    {"instruction": "Explain lab result", "input": "INR is 4.2 in a patient on warfarin.", "output": "INR is too high. Hold warfarin and consider vitamin K depending on bleeding risk."},
    {"instruction": "Recommend next steps", "input": "Middle-aged man with PSA 15 ng/mL.", "output": "Elevated PSA. Refer to urology for prostate biopsy."},
    {"instruction": "Answer patient query", "input": "Can I drink alcohol while on metronidazole?", "output": "Avoid alcohol. Metronidazole may cause disulfiram-like reaction (flushing, nausea)."},
    {"instruction": "Suggest medication", "input": "Postmenopausal woman with osteoporosis.", "output": "Recommend bisphosphonates like alendronate. Ensure calcium + vitamin D intake."},
    {"instruction": "Provide alternatives", "input": "Patient with GERD not tolerating PPIs.", "output": "Try H2 blockers like ranitidine or famotidine. Consider lifestyle modifications."},
    {"instruction": "Explain side effects", "input": "Patient started on SSRIs for depression.", "output": "Common side effects include GI upset, sexual dysfunction, and insomnia."},
    {"instruction": "Recommend vaccine", "input": "65-year-old male with COPD.", "output": "Advise pneumococcal and influenza vaccines. Consider Tdap booster."},
    {"instruction": "Provide a possible diagnosis", "input": "Patient reports fatigue, pale skin, and spoon-shaped nails.", "output": "Likely iron-deficiency anemia. Recommend CBC and iron studies."},
    {"instruction": "Suggest treatment options", "input": "Teenager with acne resistant to topical therapy.", "output": "Consider oral doxycycline or isotretinoin under dermatology supervision."},
    {"instruction": "Explain lab result", "input": "ESR is elevated at 70 mm/hr.", "output": "Suggests inflammation. Rule out infection, autoimmune disease, or malignancy."},
    {"instruction": "Answer patient query", "input": "Is turmeric effective for arthritis?", "output": "Some evidence supports anti-inflammatory effects. Safe in moderate doses but not a replacement for NSAIDs."},
    {"instruction": "Suggest medication", "input": "Patient with migraine aura and severe headaches.", "output": "Use triptans like sumatriptan. Consider beta-blockers or topiramate for prevention."},
    {"instruction": "Provide alternatives", "input": "Patient with diabetes and renal impairment can't take metformin.", "output": "Consider GLP-1 receptor agonists or insulin."},
    {"instruction": "Explain side effects", "input": "Patient on clozapine for schizophrenia.", "output": "Monitor for agranulocytosis, seizures, and myocarditis. Regular CBC required."},
    {"instruction": "Recommend screening", "input": "50-year-old woman, asymptomatic.", "output": "Screen for breast, cervical, colorectal cancer. Check lipid profile and glucose."},
    {"instruction": "Provide a possible diagnosis", "input": "Child with barking cough, stridor, and hoarseness.", "output": "Likely croup. Treat with steroids and racemic epinephrine if severe."},
    {"instruction": "Suggest treatment options", "input": "Patient with mild COVID-19 and no risk factors.", "output": "Home isolation, hydration, symptomatic care. No antivirals needed unless worsening."},
    {"instruction": "Explain lab result", "input": "Ferritin is 900 ng/mL in a patient with liver disease.", "output": "Could be acute phase reactant or hemochromatosis. Correlate clinically."},
    {"instruction": "Answer patient query", "input": "Is it safe to take melatonin long term?", "output": "Generally safe. Use lowest effective dose. Monitor for daytime drowsiness."},
    {"instruction": "Suggest medication", "input": "Hypertensive patient not controlled on amlodipine.", "output": "Add ACE inhibitor or thiazide diuretic. Monitor BP and electrolytes."},
    {"instruction": "Provide alternatives", "input": "Patient on NSAIDs with GI upset.", "output": "Use selective COX-2 inhibitors like celecoxib or add PPI cover."},
    {"instruction": "Explain side effects", "input": "Patient on rifampin for TB.", "output": "Can cause orange urine, hepatotoxicity, and drug interactions. Monitor LFTs."},
    {"instruction": "Recommend antiplatelet therapy", "input": "Patient with ischemic stroke history.", "output": "Recommend lifelong aspirin or clopidogrel unless contraindicated."},
    {"instruction": "Explain use", "input": "What is atorvastatin used for?", "output": "Used to lower LDL cholesterol and prevent cardiovascular events."},
    {"instruction": "Describe mechanism", "input": "How does omeprazole work?", "output": "Omeprazole is a PPI that inhibits gastric acid secretion by blocking H+/K+ ATPase pump."},
    {"instruction": "List indications", "input": "Indications for metoprolol.", "output": "Used for hypertension, angina, post-MI, arrhythmias, and heart failure (select types)."},
    {"instruction": "Warn about drug interaction", "input": "Patient takes fluoxetine and starts tramadol for pain.", "output": "This combination increases the risk of serotonin syndrome. Use with caution or consider alternatives."},
    {"instruction": "Discuss side effects", "input": "Patient using isotretinoin for acne.", "output": "May cause teratogenicity, dry skin, liver toxicity, and mood changes. Pregnancy test and monitoring required."},
    {"instruction": "Explain test result", "input": "HbA1c is 9.2%", "output": "Poor glycemic control. Indicates average blood glucose ~220 mg/dL. Intensify therapy and lifestyle changes."},
    {"instruction": "Respond empathetically", "input": "Patient fears cancer after blood in stool.", "output": "While rectal bleeding can be due to hemorrhoids or fissures, we must rule out serious causes. A colonoscopy is advised."},
    {"instruction": "Highlight safety risk", "input": "Elderly patient on benzodiazepines.", "output": "Increased risk of falls, confusion, and dependence. Tapering and safer alternatives are recommended."},
    {"instruction": "Explain disease risk", "input": "Obese 45-year-old with high blood pressure.", "output": "At risk for cardiovascular disease, stroke, and diabetes. Lifestyle modification and medication needed."},
    {"instruction": "Clarify patient doubt", "input": "Is skipping statins OK if cholesterol improves?", "output": "No. Stopping statins can increase cardiac risk. Continue unless advised otherwise."},
    {"instruction": "Advise safety measure", "input": "Parent asks about fever in infant under 2 months.", "output": "Fever in this age group is an emergency. Immediate medical evaluation is needed."},
    {"instruction": "Discuss over-the-counter risks", "input": "Patient frequently uses ibuprofen for headaches.", "output": "Chronic use may cause gastric ulcers, kidney issues, or rebound headaches. Evaluate for underlying cause."},
    {"instruction": "Handle vaccine concerns", "input": "Worried about COVID-19 vaccine side effects.", "output": "Common side effects are mild. Benefits outweigh risks, especially for high-risk individuals. Discuss any specific fears."}
]


dataset = Dataset.from_list(examples)

# ========================
# 📜 Format Prompts for Training
# ========================
def format(example):
    prompt = f"""
You are a licensed medical professional. Provide a **concise, accurate, and clinically safe** response to the following patient query. Use a tone that is:
- Accurate like a doctor 🧬
- Empathetic like a therapist 💬
- Direct like a warning label 🚫

### Clinical Instruction:
{example['instruction']}

### Patient Data:
{example['input']}

### Response:
{example['output']}
"""
    return tokenizer(prompt, padding="max_length", truncation=True, max_length=512)


tokenized = dataset.map(format)

# ========================
# 🧠 Training Setup
# ========================
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=1,
    save_steps=10,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# ========================
# 🧑‍⚕️ Gradio Medical Chatbot UI
# ========================
import gradio as gr

def generate_response(instruction, user_input):
    prompt = f"""
You are a licensed medical professional. Provide a **concise, accurate, and clinically safe** response to the following patient query. Use a tone that is:
- Accurate like a doctor 🧬
- Empathetic like a therapist 💬
- Direct like a warning label 🚫

### Clinical Instruction:
{instruction}

### Patient Data:
{user_input}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in full_output:
        return full_output.split("### Response:")[-1].strip()
    else:
        return full_output.strip()


chatbot_ui = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Instruction"),
        gr.Textbox(label="Patient Input")
    ],
    outputs=gr.Textbox(label="LLM Medical Response"),
    title="🩺 Medical LLM Chatbot",
    theme="compact"
)

chatbot_ui.launch(share=True)
