# Watermark Removal: Preserving Semantics

This repository contains the code and resources for the project **"Watermark Removal: Preserving Semantics"**, developed as part of the DS-207: Introduction to NLP course at IISc. The project evaluates the robustness of text watermarking techniques against various attacks and proposes hybrid attacks to remove watermarks while maintaining semantic integrity.

## Project Overview

Text watermarking ensures the traceability of AI-generated content, but it is vulnerable to attacks such as paraphrasing and translation. This project investigates:

- **Watermarking Techniques**: Implementation of popular watermarking methods:
  - KGW (Kirchenbauer et al., 2023)
  - BBW (Yang et al., 2023)
  - SIR (Liu et al., 2024)
- **Attacks**: Evaluation of watermark robustness against:
  - Cross-Lingual Translation
  - Recursive Paraphrasing
  - Re-Translation
- **Proposed Hybrid Attacks**:
  - Pivot Translation + Paraphrasing
  - Re-Translation + Paraphrasing
  - Recursive Paraphrasing + Re-Translation

## Methodology

- **Dataset**: XSum dataset for English and translated Mandarin prompts.
- **Model**: Llama-2-7b for text generation, watermarking, and attack evaluation.
- **Metrics**:
  - Watermark Confidence
  - Perplexity
  - BERTScore
  - Word Edit Distance
  - ROC-AUC

## Key Findings

- Hybrid attacks effectively reduce watermark confidence while maintaining text quality.
- Pivot Translation + Paraphrasing achieves the most significant reduction in watermark confidence.
- Recursive Paraphrasing + Re-Translation offers a balance of low perplexity and reduced watermark detection.

## Usage

### Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gourishankerJK/WaterMark.git
   cd WaterMark
