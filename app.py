import streamlit as st

st.set_page_config(page_title="Shakespearean English Translator", page_icon="🎭", layout="centered")

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

"""
Streamlit app for Modern English → Shakespearean English translation using a fine-tuned T5 model.
Demonstrates the importance of dataset alignment with good and bad example translations.
"""

st.title("🎭 Shakespearean English Translator")
st.write("Translate modern English to Shakespearean English using a fine-tuned T5 model.")

@st.cache_resource
def load_model():
    MODEL_DIR = "tiny-shakespeare-t5"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    device = (
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

def translate(text, max_length=64):
    """Translate modern English to Shakespearean English using the loaded T5 model."""
    input_text = f"translate English to Shakespearean: {text}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

st.subheader("Try it out!")
user_input = st.text_area("Enter modern English:", "How are you?")
if st.button("Translate"):
    with st.spinner("Translating..."):
        result = translate(user_input)
    st.success(f"Shakespearean: {result}")

st.markdown("---")
st.subheader("Examples (Good)")
good_examples = [
    "Hello",
    "Goodbye",
    "How are you?",
    "Thank you",
    "I love you"
]
for ex in good_examples:
    st.write(f"**English:** {ex}")
    st.write(f"**Shakespearean:** {translate(ex)}")
    st.write("")

st.subheader("Examples (Bad/Out-of-Domain)")
bad_examples = [
    "Can you help me with my homework?",
    "What's the weather like in Paris?",
    "This neural network is overfitting.",
    "I need to book a flight.",
    "Let's grab some coffee."
]
for ex in bad_examples:
    st.write(f"**English:** {ex}")
    st.write(f"**Shakespearean:** {translate(ex)}")
    st.write("")

st.markdown("---")
st.info("This demo uses a tiny, perfectly aligned dataset. Results on real, noisy data will be much worse. See the README and blog for details.")
