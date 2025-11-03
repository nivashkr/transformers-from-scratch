#ifndef TRANSFORMER_CPP_TRANSFORMER_H
#define TRANSFORMER_CPP_TRANSFORMER_H

#include "../layers/Embedding.h"
#include "../layers/Linear.h"
#include "../layers/PositionalEncoding.h"
#include "../utils/Masking.h"
#include "Decoder.h"
#include "Encoder.h"
#include <memory>
#include <string>
#include <vector>

class Tensor;

class Transformer {
public:
  Transformer(int input_vocab_size, int target_vocab_size, int embed_dim,
              int max_sequence_length, int num_layers, int num_heads,
              int ff_hidden_dim, float dropout_rate, float pad_token_id = 0.0f);

  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &encoder_input_ids,
          const std::shared_ptr<Tensor> &decoder_input_ids, bool is_training);

  void save_weights(const std::string &filename) const;
  void load_weights(const std::string &filename);

  ~Transformer() = default;

private:
  Embedding encoder_embedding_;
  PositionalEncoding encoder_positional_encoding_;
  Encoder encoder_;

  Embedding decoder_embedding_;
  PositionalEncoding decoder_positional_encoding_;
  Decoder decoder_;

  Linear final_linear_;

  int input_vocab_size_;
  int target_vocab_size_;
  int embed_dim_;
  int max_sequence_length_;
  int num_layers_;
  int num_heads_;
  int ff_hidden_dim_;
  float dropout_rate_;
  float pad_token_id_;

  std::shared_ptr<Tensor>
  create_encoder_padding_mask(const std::shared_ptr<Tensor> &encoder_input_ids);
  std::shared_ptr<Tensor> create_decoder_self_attention_mask(
      const std::shared_ptr<Tensor> &decoder_input_ids);
  std::shared_ptr<Tensor> create_decoder_cross_attention_mask(
      const std::shared_ptr<Tensor> &encoder_input_ids);
};

#endif // TRANSFORMER_CPP_TRANSFORMER_H
