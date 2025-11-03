#ifndef TRANSFORMER_CPP_ENCODER_H
#define TRANSFORMER_CPP_ENCODER_H

#include "EncoderLayer.h"
#include <memory>
#include <vector>

class Tensor;

class Encoder {
public:
  // Constructor
  Encoder(int num_layers, int embed_dim, int num_heads, int ff_hidden_dim,
          float dropout_rate);

  // Forward pass
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input,
                                  std::shared_ptr<Tensor> &padding_mask,
                                  bool is_training);

  // Destructor
  ~Encoder() = default;

private:
  std::vector<EncoderLayer> layers_;

  int num_layers_;
  int embed_dim_;
  int num_heads_;
  int ff_hidden_dim_;
  float dropout_rate_;
};

#endif // TRANSFORMER_CPP_ENCODER_H
