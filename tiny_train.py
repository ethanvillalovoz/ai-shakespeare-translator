import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments
)

"""
Script for training a T5 model on a tiny, hand-crafted modern→Shakespearean English dataset.
Validates the translation pipeline and demonstrates the importance of dataset alignment.
"""

# Device selection: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

modern = [
    "Hello",
    "Goodbye",
    "How are you?",
    "Thank you",
    "I love you",
]
shakespeare = [
    "Hail!",
    "Fare thee well!",
    "How dost thou?",
    "I thank thee.",
    "I do love thee.",
]

tokenizer = T5Tokenizer.from_pretrained("t5-small")
inputs = tokenizer(modern, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
targets = tokenizer(shakespeare, padding='max_length', truncation=True, max_length=16, return_tensors="pt")

class TinyDataset(Dataset):
    """PyTorch Dataset for the tiny, hand-crafted translation pairs."""
    def __init__(self, inputs, targets):
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
        self.labels = targets["input_ids"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

dataset = TinyDataset(inputs, targets)

model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

training_args = TrainingArguments(
    output_dir="./tiny_results",
    num_train_epochs=100,
    per_device_train_batch_size=2,
    save_strategy="no",
    logging_steps=1,
    learning_rate=5e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)

trainer.train()
model.save_pretrained("tiny-shakespeare-t5")
tokenizer.save_pretrained("tiny-shakespeare-t5")
print("Tiny model trained and saved.")