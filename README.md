# TransformerCPP: Transformers from Scratch in C++

[GitHub repo](https://github.com/nivashkr/transformers-from-scratch)

## Why I Built This

Most deep learning code today is written using big frameworks like PyTorch or TensorFlow. These are great for getting results fast, but they hide a lot of the details. If you want to really understand how a Transformer works, or if you want to see what’s happening at the lowest level, you have to dig deeper.

I wanted to know what’s really going on inside a Transformer. So I wrote one from scratch in C++. No frameworks, no shortcuts. Just the basics: tensors, layers, attention, and training loops. This project is about learning by building, and about making every part of the model visible and modifiable.

## What This Project Is

TransformerCPP is a full implementation of the Transformer model, as described in the “Attention Is All You Need” paper. It’s written in C++17, and it doesn’t use any external deep learning libraries. Everything is built from the ground up, including the tensor operations, the attention mechanism, and the training process.

The code is organized so you can see how each part works. If you want to change something, you can. If you want to understand something, you can read the code directly.

## Main Ideas

- **Transparency**: You can see and change every part of the model. There’s no hidden magic.
- **Performance**: The code is optimized for CPUs. It uses threads and SIMD instructions to make things faster.
- **Modularity**: Each part of the model (tensors, layers, data loading) is separated, so you can work on one part without breaking the rest.
- **No Dependencies**: Only the C++ standard library is used. No external math or neural network libraries.

## What’s Inside

### Core Tensor Operations

- Custom tensor class with support for broadcasting, reshaping, and arithmetic
- Thread pool for running heavy computations in parallel
- Automatic differentiation for backpropagation

### Neural Network Layers

- Linear layers (matrix multiply + bias)
- Multi-head attention (the main part of the Transformer)
- Feed-forward networks
- Layer normalization and dropout
- Embedding and positional encoding

### Model Structure

- Encoder stack with self-attention
- Decoder stack with masked self-attention and encoder-decoder attention
- Full Transformer model that combines encoder and decoder

### Data Handling

- Character-level tokenization
- Batch processing and sequence management
- DataLoader for training and inference

### Utilities

- Config file parser for hyperparameters
- Thread pool for parallel execution
- Helper functions for common tasks

## How to Build

You need a C++17 compiler and CMake (3.14+).

```bash
git clone https://github.com/nivashkr/transformers-from-scratch.git
cd transformers-from-scratch
mkdir build
cd build
cmake ..
make
```

## How to Use

All settings are in `config.ini`. You can switch between training and inference by changing a single line.

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
2. Adjust training parameters if needed
3. Run:

```bash
./neural_network
```

The model will train on your dataset and save the weights.

### Inference

1. Set `inference_mode = true`
2. Make sure `load_existing_weights = true` and `weights_filename` is correct
3. Set your `initial_prompt` and `max_generate_length`
4. Run:

```bash
./neural_network
```

The model will load the weights and generate text.

## Testing

There’s a test suite for tensor operations:

```bash
./test_tensor
```

## Performance Notes

- Set `num_threads` in `config.ini` to match your CPU for best speed.
- Multi-threading is used for matrix multiplications, element-wise operations, and attention.
- Compile with SIMD flags for extra speed if your compiler supports it.

## Final Thoughts

This project is for anyone who wants to see how Transformers work at the lowest level, or who wants a fast, flexible Transformer implementation in C++. If you want to learn, experiment, or just see what’s possible without big frameworks, take a look at the code.

If you have questions or want to contribute, check out the [GitHub repo](https://github.com/nivashkr/transformers-from-scratch).


ps: inspired by karpathy's zero to hero neural network series.
---
