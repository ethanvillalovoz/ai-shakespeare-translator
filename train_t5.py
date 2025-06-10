import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import transformers

# Print the version of the transformers library
print("Transformers version:", transformers.__version__)

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

# Load tokenized data
data = torch.load("data/tokenized_train.pt")

class ShakespeareDataset(Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# Create dataset
dataset = ShakespeareDataset(data)

# Optionally, split into train/val
train_size = int(0.70 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-large")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=30,  # 30 epochs as requested
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=3e-5,
    weight_decay=0.01,
    push_to_hub=False,
    fp16=False,  # Set to True if your GPU supports it
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,  # Only keep the 2 most recent/best checkpoints
    gradient_accumulation_steps=2,  # Optional: increase if you want a larger effective batch size
)

# Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train!
trainer.train()

# Save the final model
model.save_pretrained("shakespeare-t5-model")
tokenizer.save_pretrained("shakespeare-t5-model")
print("Training complete. Model saved to shakespeare-t5-model/")