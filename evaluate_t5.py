from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load the dataset
dataset = load_dataset("Roudranil/shakespearean-and-modern-english-conversational-dataset")
test_set = dataset['test']

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("shakespeare-t5-model")
tokenizer = T5Tokenizer.from_pretrained("shakespeare-t5-model")

def translate_to_shakespeare(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=64, num_beams=5)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

bleu_scores = []
smooth = SmoothingFunction().method4

# Evaluate on the first 50 test samples (adjust as needed)
for i in range(50):
    print(f"Sample {i+1}:")
    modern = test_set[i]['translated_dialog']
    target = test_set[i]['og_response']
    pred = translate_to_shakespeare(modern)
    print(f"Modern: {modern}")
    print(f"Target: {target}")
    print(f"Predicted: {pred}")
    score = sentence_bleu([target.split()], pred.split(), smoothing_function=smooth)
    print(f"BLEU: {score:.3f}")
    print("-" * 40)
    bleu_scores.append(score)

print(f"Average BLEU score: {sum(bleu_scores)/len(bleu_scores):.3f}")