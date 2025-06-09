from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("shakespeare-t5-model")
tokenizer = T5Tokenizer.from_pretrained("shakespeare-t5-model")

def translate_to_shakespeare(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=64, num_beams=5)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example usage
modern = "How are you doing today?"
shakespearean = translate_to_shakespeare(modern)
print(f"Modern: {modern}")
print(f"Shakespearean: {shakespearean}")