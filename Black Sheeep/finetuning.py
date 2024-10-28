from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistral7b")  # Replace with your desired model
model = AutoModelForCausalLM.from_pretrained("mistral7b")

# Tokenize the input text for model training
tokenized_inputs = tokenizer(list(df['input_text']), padding=True, truncation=True, return_tensors="pt")

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['input_text']])

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    learning_rate=5e-5,
    warmup_steps=500,
)

# Initialize the Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Start the fine-tuning process
trainer.train()
