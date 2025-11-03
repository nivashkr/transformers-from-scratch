#ifndef TRANSFORMER_CPP_POSITIONALENCODING_H
#define TRANSFORMER_CPP_POSITIONALENCODING_H

#include "../utils/Tensor.h"
#include <cmath>
#include <memory>
#include <vector>

class PositionalEncoding {
public:
  // Constructor
  PositionalEncoding(int max_sequence_length, int embed_dim);

  // Forward pass
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input);

  // Destructor
  ~PositionalEncoding() = default;

private:
  std::shared_ptr<Tensor> positional_encodings_; // Pre-calculated

  int max_sequence_length_;
  int embed_dim_;
};

#endif // TRANSFORMER_CPP_POSITIONALENCODING_H
