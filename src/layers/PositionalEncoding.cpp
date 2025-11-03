#include "layers/PositionalEncoding.h"

PositionalEncoding::PositionalEncoding(int max_sequence_length, int embed_dim)
    : max_sequence_length_(max_sequence_length), embed_dim_(embed_dim) {
  // Pre-calculate the positional encodings matrix
  positional_encodings_ = Tensor::create({max_sequence_length_, embed_dim_});
  std::vector<float> &encodings_data = positional_encodings_->data_ref();

  for (int pos = 0; pos < max_sequence_length_; ++pos) {
    for (int i = 0; i < embed_dim_; ++i) {
      if (i % 2 == 0) {
        // Sine for even dimensions
        encodings_data[pos * embed_dim_ + i] = std::sin(
            pos / std::pow(10000.0f, static_cast<float>(i) / embed_dim_));
      } else {
        // Cosine for odd dimensions
        encodings_data[pos * embed_dim_ + i] = std::cos(
            pos / std::pow(10000.0f, static_cast<float>(i - 1) / embed_dim_));
      }
    }
  }
}

std::shared_ptr<Tensor>
PositionalEncoding::forward(const std::shared_ptr<Tensor> &input) {
  const std::vector<int> &input_shape = input->get_shape();
  if (input_shape.size() != 3) {
    throw std::runtime_error("Positional Encoding input must be a 3D tensor "
                             "(batch_size, sequence_length, embed_dim).");
  }

  size_t batch_size = input_shape[0];
  size_t sequence_length = input_shape[1];
  size_t embed_dim = input_shape[2];

  if (embed_dim != embed_dim_) {
    throw std::runtime_error(
        "Input embedding dimension mismatch in Positional Encoding.");
  }

  if (sequence_length > max_sequence_length_) {
    throw std::runtime_error("Input sequence length exceeds "
                             "max_sequence_length in Positional Encoding.");
  }

  // Add the pre-calculated positional encodings to the input tensor.
  std::shared_ptr<Tensor> output = Tensor::create(input_shape);
  const std::vector<float> &input_data = input->get_data();
  std::vector<float> &output_data = output->data_ref();
  const std::vector<float> &encoding_data = positional_encodings_->get_data();

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t s = 0; s < sequence_length; ++s) {
      size_t input_start_idx = (b * sequence_length + s) * embed_dim;
      size_t encoding_start_idx = s * embed_dim;

      for (size_t d = 0; d < embed_dim; ++d) {
        output_data[input_start_idx + d] =
            input_data[input_start_idx + d] +
            encoding_data[encoding_start_idx + d];
      }
    }
  }
  output->set_creator_op(
      OperationType::Add); // Doesn't rly matter what this is cause you don't
                           // train the positional encodings
  output->set_parents({input, positional_encodings_});

  return output;
}