# ai-shakespeare-translator: Modern English to Shakespearean with Transformers

## What I Built

**ai-shakespeare-translator** is a neural machine translation system that converts modern English into Shakespearean English using the T5 transformer model. The project explores the challenges of text style transfer, especially the critical role of dataset alignment in achieving meaningful results. It is designed to run efficiently on Apple Silicon (M1/M2) using a Conda environment, but also supports CUDA and CPU.

**Key components:**
- **Training Pipeline:** Scripts for training T5 on both tiny, hand-crafted datasets and larger conversational datasets.
- **Device Flexibility:** Seamless support for CUDA, Apple Silicon (MPS), and CPU, making it easy to train and test on a Mac.
- **Tiny Dataset Demo:** A controlled experiment showing that the pipeline works perfectly when given a small, well-aligned dataset.
- **Extensible Scripts:** Easily swap in new datasets or model variants for further experimentation.

---

## How the Translation Pipeline Works

The core of the system is a sequence-to-sequence (seq2seq) model based on T5. The pipeline includes:
- **Data Preparation:** Tokenizes modern and Shakespearean English pairs, ensuring they are aligned for training.
- **Model Training:** Fine-tunes T5 on the provided dataset, with support for early stopping, checkpointing, and hardware acceleration.
- **Inference:** Generates Shakespearean translations for new modern English inputs.

A key experiment was training on a tiny, hand-crafted dataset (e.g., “Hello” → “Hail!”), which demonstrated perfect learning and highlighted that the main bottleneck is dataset quality, not model architecture or training procedure.

---

## Visualizing Model Behavior

- **Tiny Dataset:** The model achieves perfect translation on the tiny set, e.g.:

  | Modern English | Shakespearean Output |
  |---------------|---------------------|
  | Hello         | Hail!               |
  | Goodbye       | Fare thee well!     |
  | How are you?  | How dost thou?      |
  | Thank you     | I thank thee.       |
  | I love you    | I do love thee.     |

- **Larger Dataset:** When trained on a larger, less-aligned dataset, the model produces generic or inaccurate outputs, underscoring the importance of parallel data.

---

## Recent Improvements

- **Apple Silicon Support:** Added MPS device support for fast training on Mac hardware.
- **Early Stopping and Checkpointing:** Prevents overfitting and saves the best model automatically.
- **Pipeline Validation:** Always tests the pipeline with a tiny, controlled dataset before scaling up.
- **Flexible Training Scripts:** Modular code for easy experimentation with different datasets and model sizes.

---

## Results

**Quantitative Evaluation:**

- **Tiny Dataset:**  
  - BLEU: 1.00 (perfect)  
  - ROUGE-L F1: 1.00 (perfect)  
  - The model produces accurate, fluent Shakespearean translations for all in-domain examples.

- **Noisy/Large Dataset:**  
  - BLEU and ROUGE scores drop significantly.
  - Outputs are often generic or incorrect, e.g., “I know not.”

**Qualitative Examples:**

| Modern English | Shakespearean Output |
|----------------|---------------------|
| Hello          | Hail!               |
| Goodbye        | Fare thee well!     |
| How are you?   | How dost thou?      |
| Thank you      | I thank thee.       |
| I love you     | I do love thee.     |

| Modern English                        | Shakespearean Output (noisy/full) |
|----------------------------------------|-----------------------------------|
| Can you help me with my homework?      | I know not.                       |
| What's the weather like in Paris?      | I know not.                       |
| This neural network is overfitting.    | I know not.                       |
| I need to book a flight.               | I know not.                       |
| Let's grab some coffee.                | I know not.                       |

**Training Curves:**  
- The tiny model’s loss drops rapidly and stabilizes.
- The large model’s loss decreases slowly and may plateau, reflecting the challenge of learning from noisy data.

---

*These results highlight the critical importance of dataset alignment for successful style transfer in NLP.*

---

## Why It Matters

Text style transfer is a classic NLP challenge, and Shakespearean English is a fun, high-variance target. This project demonstrates that:
- **Model performance is fundamentally limited by data quality.**
- Even state-of-the-art models like T5 cannot “invent” good translations without well-aligned training data.
- Validating your pipeline with a tiny, controlled dataset is essential before scaling up.

The lessons here generalize to many NLP tasks: data curation is often more important than model tweaking.

---

## What I Learned

- **Data Quality is Everything:** Even the best models fail without well-aligned data.
- **Hardware Flexibility:** Apple Silicon (MPS) and Conda make it easy to train models on a Mac.
- **Pipeline Validation:** Always test your pipeline with a tiny, controlled dataset before scaling up.
- **Documentation and Experiment Tracking:** Keeping clear notes and results is crucial for understanding what works and why.

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

*Author: Ethan Villalovoz*