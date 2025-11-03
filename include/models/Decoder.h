#ifndef TRANSFORMER_CPP_DECODER_H
#define TRANSFORMER_CPP_DECODER_H

#include "DecoderLayer.h"
#include <memory>
#include <vector>

class Tensor;

class Decoder {
public:
  // Constructor
  Decoder(int num_layers, int embed_dim, int num_heads, int ff_hidden_dim,
          float dropout_rate);

  // Forward pass
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &target_input,
                                  std::shared_ptr<Tensor> &encoder_output,
                                  std::shared_ptr<Tensor> &look_ahead_mask,
                                  std::shared_ptr<Tensor> &padding_mask,
                                  bool is_training);

  // Destructor
  ~Decoder() = default;

private:
  std::vector<DecoderLayer> layers_;

  int num_layers_;
  int embed_dim_;
  int num_heads_;
  int ff_hidden_dim_;
  float dropout_rate_;
};

#endif // TRANSFORMER_CPP_DECODER_H
