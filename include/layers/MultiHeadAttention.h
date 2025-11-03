#ifndef TRANSFORMER_CPP_MULTIHEADATTENTION_H
#define TRANSFORMER_CPP_MULTIHEADATTENTION_H

#include "Linear.h"
#include <memory>
#include <vector>

class Tensor;

class MultiHeadAttention {
public:
  // Constructor
  MultiHeadAttention(int embed_dim, int num_heads);

  // Forward pass
  // query, key, and value can be the same tensor for self-attention, or
  // different for cross-attention
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &query,
                                  std::shared_ptr<Tensor> &key,
                                  std::shared_ptr<Tensor> &value,
                                  std::shared_ptr<Tensor> mask = nullptr);

  // Destructor
  ~MultiHeadAttention() = default;

private:
  Linear query_proj_;
  Linear key_proj_;
  Linear value_proj_;
  Linear output_proj_;

  int embed_dim_;
  int num_heads_;
  int head_dim_;

  // Helper function to calculate scaled dot-product attention
  std::shared_ptr<Tensor> scaled_dot_product_attention(
      std::shared_ptr<Tensor> &q, std::shared_ptr<Tensor> &k,
      std::shared_ptr<Tensor> &v, std::shared_ptr<Tensor> mask = nullptr);
};

#endif // TRANSFORMER_CPP_MULTIHEADATTENTION_H
