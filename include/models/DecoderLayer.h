#ifndef TRANSFORMER_CPP_DECODERLAYER_H
#define TRANSFORMER_CPP_DECODERLAYER_H

#include "../layers/Dropout.h"
#include "../layers/LayerNorm.h"
#include "../layers/MultiHeadAttention.h"
#include "../layers/PositionwiseFeedForward.h"
#include <memory>
#include <vector>

class Tensor;

class DecoderLayer {
public:
  // Constructor
  DecoderLayer(int embed_dim, int num_heads, int ff_hidden_dim,
               float dropout_rate = 0.1f);

  // Forward pass
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &target_input,
                                  std::shared_ptr<Tensor> &encoder_output,
                                  std::shared_ptr<Tensor> &look_ahead_mask,
                                  std::shared_ptr<Tensor> &padding_mask,
                                  bool is_training);

  // Destructor
  ~DecoderLayer() = default;

private:
  MultiHeadAttention masked_self_attention_;
  LayerNorm layernorm1_;
  Dropout dropout1_;
  MultiHeadAttention cross_attention_;
  LayerNorm layernorm2_;
  Dropout dropout2_;
  PositionwiseFeedForward feed_forward_;
  LayerNorm layernorm3_;
  Dropout dropout3_;

  float dropout_rate_;
};

#endif // TRANSFORMER_CPP_DECODERLAYER_H
