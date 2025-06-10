"""
Script for inference using the full Shakespearean T5 model.
Demonstrates translation on the full, noisy dataset.
"""

from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("outputs/shakespeare-t5-model")
tokenizer = T5TokenizerFast.from_pretrained("outputs/shakespeare-t5-model")

def translate_to_shakespeare(text):
    """Translate modern English to Shakespearean English using the full T5 model."""
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=64, num_beams=5)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example usage
modern = "How are you doing today?"
shakespearean = translate_to_shakespeare(modern)
print(f"Modern: {modern}")
print(f"Shakespearean: {shakespearean}")