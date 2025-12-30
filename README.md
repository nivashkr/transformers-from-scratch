# TransformerCPP: Building Transformers from Scratch in C++

*“The best way to understand something is to build it from scratch.”* — Andrej Karpathy (probably)
\
Check out the [GitHub repo](https://github.com/nivashkr/transformers-from-scratch). 

## Why Write Yet Another Transformer?

Deep learning frameworks are amazing. They let us build, train, and deploy state-of-the-art models with just a few lines of code. But sometimes, the magic feels a bit too magical. What’s really happening under the hood? How do those matrix multiplications, attention heads, and layer normalizations actually work? 

Inspired by Andrej Karpathy’s legendary “from scratch” neural network tutorials, I set out to answer these questions for myself. The result is **TransformerCPP**: a high-performance, dependency-free C++ implementation of the Transformer architecture, built from the ground up. No PyTorch, no TensorFlow, no Eigen, not even BLAS. Just C++17, a compiler, and a lot of curiosity.

## What Is TransformerCPP?

TransformerCPP is a complete, modular implementation of the Transformer model as described in the “Attention Is All You Need” paper. It’s designed for CPU execution, with multi-threading and SIMD optimizations for speed. The codebase is organized so you can see, touch, and modify every part of the model—from tensor operations to attention mechanisms to training loops.

## Design Philosophy

- **Transparency**: Every line of code is meant to be readable and hackable. If you want to know how backpropagation works, or how multi-head attention is implemented, you can just look.
- **Performance**: The implementation is optimized for CPUs, using thread pools and SIMD where possible. You can train and run models efficiently, even on modest hardware.
- **Modularity**: Components are separated cleanly—tensors, layers, models, data loaders—so you can experiment, swap things out, or extend the codebase.
- **Minimal Dependencies**: Only the C++ standard library is used. No external deep learning frameworks or math libraries.

## What’s Inside?

### Core Tensor Operations

- Custom tensor class with broadcasting, reshaping, and arithmetic
- Thread-pooled execution for heavy computations
- Automatic differentiation for backpropagation

### Neural Network Layers

- Linear layers (weights + biases)
- Multi-head attention (the heart of the Transformer)
- Position-wise feed-forward networks
- Layer normalization and dropout
- Embedding and positional encoding

### Model Architecture

- Encoder stack with self-attention
- Decoder stack with masked self-attention and encoder-decoder attention
- Full Transformer model combining encoder and decoder

### Data Processing

- Character-level tokenization
- Batch processing and sequence handling
- DataLoader for training and inference

### Configuration and Utilities

- Config parser for model hyperparameters
- Thread pool for parallel execution
- Helper functions for common operations

## How Do You Build It?

You’ll need a C++17-compatible compiler and CMake (3.14+).

```bash
git clone https://github.com/nivashkr/transformers-from-scratch.git
cd transformer.cpp
mkdir build
cd build
cmake ..
make
```

## How Do You Run It?

The project supports both training and inference. Configuration is handled via `config.ini`.

### Example `config.ini`:

```ini
inference_mode = true
load_existing_weights = true
weights_filename = transformer_weights.bin
data_filename = ../data/tiny_shakespeare.txt

embed_dim = 256
max_sequence_length = 100
num_layers = 8
num_heads = 8
ff_hidden_dim = 1024
dropout_rate = 0.1
pad_token_id = 0.0

learning_rate = 0.0005
num_epochs = 100
batch_size = 16
input_seq_length = 10
decoder_seq_length = 10

max_generate_length = 100
initial_prompt = ROMEO:

num_threads = 500
```

### Training

1. Set `inference_mode = false` in `config.ini`
2. Adjust training parameters as needed
3. Run:

```bash
./neural_network
```

### Inference

1. Set `inference_mode = true`
2. Ensure `load_existing_weights = true` and `weights_filename` is correct
3. Set your `initial_prompt` and `max_generate_length`
4. Run:

```bash
./neural_network
```

## Testing

A test suite for tensor operations is included:

```bash
./test_tensor
```

## Performance Tips

- Set `num_threads` in `config.ini` to match your hardware for best results.
- Multi-threading is used for matrix multiplications, element-wise ops, and attention.
- Compile with SIMD flags for extra speed.

## Final Thoughts

TransformerCPP is my attempt to demystify the Transformer architecture and make it accessible to anyone willing to dive into the code. If you’re curious about how deep learning works at the lowest level, or if you want a fast, flexible Transformer implementation in C++, this project is for you.


---

