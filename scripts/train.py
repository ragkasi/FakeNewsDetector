from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
# Get
# dataset['train'] — 80%
# dataset['test'] — 20%
dataset = load_dataset("liar")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

def simplify_label(example):
    name = dataset["train"].features["label"].names[ example["label"] ]
    example["label"] = int(name in ["pants‑fire","false","barely‑true"])
    return example

dataset = dataset.map(simplify_label)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the text field (could also try combining title + text later for improved performance):
def tokenize(example):
    return tokenizer(example["statement"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)
# Set the format to torch and specify the columns to include
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
# Load the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
)
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)
# Train the model
trainer.train()
# Evaluate the model
trainer.save_model("models/bert-liar-fake-news")



