#include "models/EncoderLayer.h"
#include <iostream>

EncoderLayer::EncoderLayer(int embed_dim, int num_heads, int ff_hidden_dim,
                           float dropout_rate)
    : self_attention_(embed_dim, num_heads), layernorm1_(embed_dim),
      dropout1_(dropout_rate), feed_forward_(embed_dim, ff_hidden_dim),
      layernorm2_(embed_dim), dropout2_(dropout_rate),
      dropout_rate_(dropout_rate) {}

std::shared_ptr<Tensor>
EncoderLayer::forward(std::shared_ptr<Tensor> &input,
                      std::shared_ptr<Tensor> &padding_mask, bool is_training) {
  // Input shape: (batch_size, sequence_length, embed_dim)

  std::shared_ptr<Tensor> attention_output =
      self_attention_.forward(input, input, input, padding_mask);

  std::shared_ptr<Tensor> attention_output_dropped =
      dropout1_.forward(attention_output, is_training);

  // output = input + attention_output_dropped
  std::shared_ptr<Tensor> attention_residual =
      *input + attention_output_dropped;
  std::shared_ptr<Tensor> layernorm1_output =
      layernorm1_.forward(attention_residual);

  std::shared_ptr<Tensor> feed_forward_output =
      feed_forward_.forward(layernorm1_output);

  std::shared_ptr<Tensor> feed_forward_output_dropped =
      dropout2_.forward(feed_forward_output, is_training);

  // output = layernorm1_output + feed_forward_output_dropped
  std::shared_ptr<Tensor> feed_forward_residual =
      *layernorm1_output + feed_forward_output_dropped;
  std::shared_ptr<Tensor> layernorm2_output =
      layernorm2_.forward(feed_forward_residual);

  return layernorm2_output;
}