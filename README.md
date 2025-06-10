# ai-shakespeare-translator

A neural machine translation project to convert modern English into Shakespearean English using seq2seq and transformer models (T5).  
This project demonstrates the importance of high-quality, aligned datasets for text style transfer.

---

## Overview

**ai-shakespeare-translator** leverages the T5 transformer model to translate modern English into Shakespearean English. The project is built and tested on Apple Silicon (M1/M2) using a Conda environment, and supports CUDA, MPS (Apple Silicon), and CPU devices.  
It serves as a case study in the critical role of dataset alignment for successful style transfer in NLP.

---

## Features

- **Modern → Shakespearean Translation:** Converts modern English sentences into Shakespearean style using a fine-tuned T5 model.
- **Flexible Hardware Support:** Runs on Apple Silicon (MPS), CUDA GPUs, or CPU.
- **Tiny Dataset Demo:** Includes a working pipeline with a tiny, hand-crafted dataset to demonstrate correct model behavior.
- **Extensible Training Scripts:** Easily swap in larger or better-aligned datasets for improved results.

---

## Dataset

- **Source:** [Shakespearean and Modern English Conversational Dataset (Hugging Face)](https://huggingface.co/datasets/Roudranil/shakespearean-and-modern-english-conversational-dataset)
- **Note:** Model performance is highly dependent on the alignment and quality of the dataset. For best results, use line-by-line translations.

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ethanvillalovoz/ai-shakespeare-translator.git
    cd ai-shakespeare-translator
    ```

2. **Create and activate a Conda environment:**
    ```bash
    conda create -n ai_shake python=3.10
    conda activate ai_shake
    pip install -r requirements.txt
    ```
    *(Apple Silicon users: ensure you have PyTorch with MPS support. See [PyTorch Apple Silicon install guide](https://pytorch.org/get-started/locally/).)*

---

## Usage

### Training

- **Tiny Dataset Demo:**  
  Run `tiny_train.py` to train on a small, hand-crafted dataset and verify the pipeline.
- **Full Dataset:**  
  Use `train_t5.py` to train on the full conversational dataset (see notes on dataset alignment).

### Inference

- Use `tiny_infer.py` to test the model on sample sentences.

---

## Example Results

With a tiny, perfectly aligned dataset, the model learns direct mappings:

| Modern English | Shakespearean Output | BLEU | ROUGE-L F1 |
|----------------|---------------------|------|------------|
| Hello          | Hail!               | 1.00 | 1.00       |
| Goodbye        | Fare thee well!     | 1.00 | 1.00       |
| How are you?   | How dost thou?      | 1.00 | 1.00       |
| Thank you      | I thank thee.       | 1.00 | 1.00       |
| I love you     | I do love thee.     | 1.00 | 1.00       |

With larger, less-aligned datasets, results may be generic or inaccurate:

| Modern English                        | Shakespearean Output (noisy/full) |
|----------------------------------------|-----------------------------------|
| Can you help me with my homework?      | I know not.                       |
| What's the weather like in Paris?      | I know not.                       |
| This neural network is overfitting.    | I know not.                       |
| I need to book a flight.               | I know not.                       |
| Let's grab some coffee.                | I know not.                       |

**Key lesson:** Data quality and alignment are critical for success.

---

## BLEU and ROUGE Metrics

- **Tiny set results:**  
  - **Average BLEU:** 1.00 (perfect) for in-domain examples  
  - **Average ROUGE-L F1:** 1.00 (perfect) for in-domain examples  
  - Out-of-domain/noisy examples: BLEU and ROUGE scores drop sharply

---

## Training Curves

- The tiny model converges quickly and achieves near-zero loss.
- The large model on noisy data converges more slowly and may not reach low loss, highlighting the importance of dataset alignment.

---

## What I Learned

- **Data Quality is Everything:** Even the best models fail without well-aligned data.
- **Hardware Flexibility:** Apple Silicon (MPS) and Conda make it easy to train models on a Mac.
- **Pipeline Validation:** Always test your pipeline with a tiny, controlled dataset before scaling up.

---

## Limitations and Future Work

- **Dataset Alignment:** The main bottleneck is finding or curating a truly parallel dataset.
- **Generalization:** The model’s ability to generalize is limited by the diversity and quality of training pairs.
- **User Interface:** Future work could include a web or command-line interface for interactive translation.
- **Other Styles:** The same pipeline could be adapted for other literary or historical styles.

---

## References

- [T5 Model Documentation](https://huggingface.co/docs/transformers/model_doc/t5)
- [Shakespearean and Modern English Conversational Dataset](https://huggingface.co/datasets/Roudranil/shakespearean-and-modern-english-conversational-dataset)

---

## Author

Ethan Villalovoz
