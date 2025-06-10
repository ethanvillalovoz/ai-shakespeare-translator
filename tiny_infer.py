from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the tiny model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("tiny-shakespeare-t5")
tokenizer = T5Tokenizer.from_pretrained("tiny-shakespeare-t5")

def translate_to_shakespeare(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=16)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

test_modern = [
    "Hello",
    "Goodbye",
    "How are you?",
    "Thank you",
    "I love you",
]

for modern in test_modern:
    shakespearean = translate_to_shakespeare(modern)
    print(f"Modern: {modern}")
    print(f"Shakespearean: {shakespearean}")
    print("-" * 30)