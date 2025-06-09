import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
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
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",  # <-- CORRECT
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    weight_decay=0.01,
    push_to_hub=False,
    fp16=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train!
trainer.train()

# Save the final model
model.save_pretrained("shakespeare-t5-model")
tokenizer.save_pretrained("shakespeare-t5-model")
print("Training complete. Model saved to shakespeare-t5-model/")