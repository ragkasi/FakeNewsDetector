from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
# Reload dataset and tokenizer
dataset = load_dataset("liar")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

def simplify_label(example):
    name = dataset["train"].features["label"].names[ example["label"] ]
    example["label"] = int(name in ["pants‑fire","false","barely‑true"])
    return example

dataset = dataset.map(simplify_label)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Tokenize the text field (can try combining title + text later for improved performance):
def tokenize(example):
    return tokenizer(example["statement"], truncation=True, padding="max_length", max_length=128)
# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
# Load model
model = AutoModelForSequenceClassification.from_pretrained("models/bert-liar-fake-news")
# Set up Trainer for evaluation
training_args = TrainingArguments(output_dir="./results", per_device_eval_batch_size=8)
trainer = Trainer(model=model, args=training_args)
# Evaluate
metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
print(metrics)