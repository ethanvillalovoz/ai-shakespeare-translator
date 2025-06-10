"""
Script for preprocessing and tokenizing the Shakespearean/Modern English dataset.
Saves tokenized tensors for model training.
"""

from datasets import load_dataset
from transformers import T5Tokenizer
import torch

# Load the dataset
dataset = load_dataset("Roudranil/shakespearean-and-modern-english-conversational-dataset")
train_data = dataset['train']

def clean_text(text):
    """Clean and normalize input text for tokenization."""
    return " ".join(text.strip().split())

modern_texts = [
    clean_text(item['translated_dialog']) 
    for item in train_data 
    if item['translated_dialog'] is not None and item['og_response'] is not None
]
shakespeare_texts = [
    clean_text(item['og_response']) 
    for item in train_data 
    if item['translated_dialog'] is not None and item['og_response'] is not None
]

tokenizer = T5Tokenizer.from_pretrained('t5-small')

inputs = tokenizer(modern_texts, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
targets = tokenizer(shakespeare_texts, padding='max_length', truncation=True, max_length=64, return_tensors="pt")

# Save the full tokenized dataset for training
torch.save({
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "labels": targets["input_ids"]
}, "data/tokenized_train.pt")

print("Full tokenized dataset saved to data/tokenized_train.pt")