#include "models/DecoderLayer.h"
#include <iostream>

DecoderLayer::DecoderLayer(int embed_dim, int num_heads, int ff_hidden_dim,
                           float dropout_rate)
    : masked_self_attention_(embed_dim, num_heads), layernorm1_(embed_dim),
      dropout1_(dropout_rate), cross_attention_(embed_dim, num_heads),
      layernorm2_(embed_dim), dropout2_(dropout_rate),
      feed_forward_(embed_dim, ff_hidden_dim), layernorm3_(embed_dim),
      dropout3_(dropout_rate), dropout_rate_(dropout_rate) {}

std::shared_ptr<Tensor>
DecoderLayer::forward(std::shared_ptr<Tensor> &target_input,
                      std::shared_ptr<Tensor> &encoder_output,
                      std::shared_ptr<Tensor> &look_ahead_mask,
                      std::shared_ptr<Tensor> &padding_mask, bool is_training) {
  // target_input shape: (batch_size, target_sequence_length, embed_dim)
  // encoder_output shape: (batch_size, input_sequence_length, embed_dim)

  std::shared_ptr<Tensor> masked_attention_output =
      masked_self_attention_.forward(target_input, target_input, target_input,
                                     look_ahead_mask);

  std::shared_ptr<Tensor> masked_attention_output_dropped =
      dropout1_.forward(masked_attention_output, is_training);

  std::shared_ptr<Tensor> masked_attention_residual =
      *target_input + masked_attention_output_dropped;
  std::shared_ptr<Tensor> layernorm1_output =
      layernorm1_.forward(masked_attention_residual);

  std::shared_ptr<Tensor> cross_attention_output = cross_attention_.forward(
      layernorm1_output, encoder_output, encoder_output, padding_mask);

  std::shared_ptr<Tensor> cross_attention_output_dropped =
      dropout2_.forward(cross_attention_output, is_training);

  std::shared_ptr<Tensor> cross_attention_residual =
      *layernorm1_output + cross_attention_output_dropped;
  std::shared_ptr<Tensor> layernorm2_output =
      layernorm2_.forward(cross_attention_residual);

  std::shared_ptr<Tensor> feed_forward_output =
      feed_forward_.forward(layernorm2_output);

  std::shared_ptr<Tensor> feed_forward_output_dropped =
      dropout3_.forward(feed_forward_output, is_training);

  std::shared_ptr<Tensor> feed_forward_residual =
      *layernorm2_output + feed_forward_output_dropped;
  std::shared_ptr<Tensor> layernorm3_output =
      layernorm3_.forward(feed_forward_residual);

  return layernorm3_output;
}
