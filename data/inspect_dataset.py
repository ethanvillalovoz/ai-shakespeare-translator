from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("Roudranil/shakespearean-and-modern-english-conversational-dataset")

# Print the first 5 pairs
for i in range(5):
    print("Modern English:", dataset['train'][i]['translated_dialog'])
    print("Shakespearean English:", dataset['train'][i]['og_response'])
    print("-" * 40)