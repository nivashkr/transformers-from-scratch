# TransformerCPP

A high-performance C++ implementation of the Transformer architecture from scratch, optimized for CPU computation.

## Overview

TransformerCPP is a complete implementation of the Transformer model architecture described in the "Attention Is All You Need" paper. This project aims to provide an efficient C++ implementation without external dependencies on deep learning frameworks.

## Design Philosophy

The core design principles of this project are:

1. **Performance**: The implementation is optimized for CPU execution with multi-threading support for computationally intensive operations.

2. **Modularity**: The codebase is organized in a modular way with clear separation between different components (tensor operations, layers, models).

3. **Flexibility**: The architecture supports both training and inference modes, with configurable parameters.

4. **Minimal Dependencies**: The implementation relies only on the C++ standard library, with no external dependencies on deep learning frameworks.

## Project Structure

The project is organized into several main components:

### Core Tensor Operations
- Custom tensor implementation with support for broadcasting, reshaping, and basic arithmetic operations
- Thread-pooled execution for performance-critical operations
- Automatic differentiation for backpropagation

### Neural Network Layers
- Linear layers with weights and biases
- Multi-head attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Dropout for regularization
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
- Configuration parser for model hyperparameters
- Thread pool implementation for parallel execution
- Helper functions for various operations

## Building the Project

### Requirements
- C++17 compatible compiler
- CMake (version 3.14 or higher)

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/KrishM123/transformer.cpp.git
cd transformer.cpp

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
make
```

## Running the Project

The project can be run in two modes: training and inference.

### Configuration

Before running, you can modify the parameters in `config.ini`:

```ini
# Model mode
inference_mode = true           # Set to false for training
load_existing_weights = true    # Whether to load pre-trained weights
weights_filename = transformer_weights.bin
data_filename = ../data/tiny_shakespeare.txt

# Model architecture
embed_dim = 256                 # Embedding dimension
max_sequence_length = 100       # Maximum sequence length
num_layers = 8                  # Number of encoder/decoder layers
num_heads = 8                   # Number of attention heads
ff_hidden_dim = 1024            # Feed-forward hidden dimension
dropout_rate = 0.1              # Dropout rate
pad_token_id = 0.0              # Padding token ID

# Training parameters
learning_rate = 0.0005          # Learning rate for Adam optimizer
num_epochs = 100                # Number of training epochs
batch_size = 16                 # Batch size
input_seq_length = 10           # Input sequence length
decoder_seq_length = 10         # Decoder sequence length

# Inference parameters
max_generate_length = 100       # Maximum length to generate
initial_prompt = ROMEO:         # Initial prompt for text generation

# Performance parameters
num_threads = 500               # Number of threads for parallel execution
```

### Training Mode

To train the model:

1. Set `inference_mode = false` in `config.ini`
2. Configure training parameters as needed
3. Run the executable:

```bash
./neural_network
```

The model will train on the specified dataset and save the weights to the specified file.

### Inference Mode

To generate text with a trained model:

1. Set `inference_mode = true` in `config.ini`
2. Make sure `load_existing_weights = true` and `weights_filename` points to a valid weights file
3. Configure the `initial_prompt` and `max_generate_length` as desired
4. Run the executable:

```bash
./neural_network
```

The model will load the weights and generate text based on the initial prompt.

## Testing

The project includes a test suite for the tensor operations:

```bash
# Run the tensor tests
./test_tensor
```

## Performance Considerations

- The `num_threads` parameter in `config.ini` controls parallel execution. For optimal performance, set this to a value appropriate for your hardware.
- Multi-threading is applied to computationally intensive operations such as matrix multiplication, element-wise operations, and attention calculations.
- The implementation uses SIMD optimizations when compiled with appropriate flags.
