#include "models/Decoder.h"
#include <iostream>

Decoder::Decoder(int num_layers, int embed_dim, int num_heads,
                 int ff_hidden_dim, float dropout_rate)
    : num_layers_(num_layers), embed_dim_(embed_dim), num_heads_(num_heads),
      ff_hidden_dim_(ff_hidden_dim), dropout_rate_(dropout_rate) {
  for (int i = 0; i < num_layers_; ++i) {
    layers_.emplace_back(embed_dim_, num_heads_, ff_hidden_dim_, dropout_rate_);
  }
}

std::shared_ptr<Tensor>
Decoder::forward(std::shared_ptr<Tensor> &target_input,
                 std::shared_ptr<Tensor> &encoder_output,
                 std::shared_ptr<Tensor> &look_ahead_mask,
                 std::shared_ptr<Tensor> &padding_mask, bool is_training) {
  std::shared_ptr<Tensor> output = target_input;

  for (size_t i = 0; i < layers_.size(); ++i) {
    output = layers_[i].forward(output, encoder_output, look_ahead_mask,
                                padding_mask, is_training);
  }

  return output;
}