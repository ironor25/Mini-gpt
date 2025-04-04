# Mini-gpt

Mini-gpt is a lightweight implementation of a GPT-like model designed for document completion tasks. It is based on the principles outlined in the "Attention is All You Need" paper and uses a pre-trained architecture. Note that this model is not fine-tuned for specific tasks, so it may not provide accurate answers to questions.

## Features
- Implements a transformer-based architecture with multi-head self-attention.
- Pre-trained on text data for document completion.
- Supports GPU acceleration for faster training and inference.

## Installation

To set up the project, ensure you have the following dependencies installed:

1. **PyTorch** with CUDA support:
   - `torch==2.5.1`
   - `torchvision==0.20.1`
   - `torchaudio==2.5.1`
   - Use the following index URL for CUDA 12.4:
     ```
     --index-url https://download.pytorch.org/whl/cu124
     ```

2. **CUDA Toolkit**:
   - Ensure you have the appropriate version of the CUDA Toolkit installed for your GPU.

3. **cuDNN**:
   - Install the cuDNN library compatible with your CUDA version.

To install the required Python packages, run:
```bash
pip install -r [requirements.txt](http://_vscodecontentref_/0)