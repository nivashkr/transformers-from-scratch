#include "layers/MultiHeadAttention.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads)
    : embed_dim_(embed_dim), num_heads_(num_heads),
      head_dim_(embed_dim / num_heads), query_proj_(embed_dim, embed_dim),
      key_proj_(embed_dim, embed_dim), value_proj_(embed_dim, embed_dim),
      output_proj_(embed_dim, embed_dim) {
  if (embed_dim_ % num_heads_ != 0) {
    throw std::runtime_error(
        "Embedding dimension must be divisible by the number of heads.");
  }
}

std::shared_ptr<Tensor> MultiHeadAttention::scaled_dot_product_attention(
    std::shared_ptr<Tensor> &q, std::shared_ptr<Tensor> &k,
    std::shared_ptr<Tensor> &v, std::shared_ptr<Tensor> mask) {
  // q, k, v shapes are typically (batch_size, num_heads, sequence_length,
  // head_dim)

  // Calculate attention scores: scores = Q . K^T
  // K^T needs transposing the last two dimensions: (batch_size, num_heads,
  // head_dim, sequence_length)
  std::vector<int> k_shape = k->get_shape();
  std::vector<int> k_transpose_perm(k_shape.size());
  std::iota(k_transpose_perm.begin(), k_transpose_perm.end(), 0);
  // Swap the last two dimensions
  if (k_transpose_perm.size() >= 2) {
    std::swap(k_transpose_perm[k_transpose_perm.size() - 1],
              k_transpose_perm[k_transpose_perm.size() - 2]);
  }
  std::shared_ptr<Tensor> k_transposed = k->transpose(k_transpose_perm);

  // Perform batched matrix multiplication: (batch_size, num_heads,
  // sequence_length, head_dim) . (batch_size, num_heads, head_dim,
  // sequence_length) Result shape: (batch_size, num_heads, sequence_length,
  // sequence_length)
  std::shared_ptr<Tensor> scores = q->dot(k_transposed);

  // Scale the scores by the square root of the head dimension
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  std::shared_ptr<Tensor> scale_tensor = Tensor::create(
      {1}, std::make_shared<std::vector<float>>(std::vector<float>{scale}));
  std::shared_ptr<Tensor> scaled_scores = *scores * scale_tensor;

  if (mask != nullptr) {
    // Mask shape is (batch_size, 1, sequence_length, sequence_length) or
    // (batch_size, sequence_length, sequence_length)
    scaled_scores = *scaled_scores + mask;
  }

  // Apply softmax to get attention weights
  std::shared_ptr<Tensor> attention_weights =
      scaled_scores->softmax(scaled_scores->get_shape().size() - 1);

  // Multiply attention weights by values: attention_output = weights . V
  // weights shape: (batch_size, num_heads, sequence_length, sequence_length)
  // v shape: (batch_size, num_heads, sequence_length, head_dim)
  // Result shape: (batch_size, num_heads, sequence_length, head_dim)
  std::shared_ptr<Tensor> attention_output = attention_weights->dot(v);

  return attention_output;
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(
    std::shared_ptr<Tensor> &query, std::shared_ptr<Tensor> &key,
    std::shared_ptr<Tensor> &value, std::shared_ptr<Tensor> mask) {
  // query, key, value input shapes are (batch_size, sequence_length, embed_dim)

  const std::vector<int> &query_shape = query->get_shape();
  const std::vector<int> &key_shape = key->get_shape();
  const std::vector<int> &value_shape = value->get_shape();

  if (query_shape.size() != 3 || key_shape.size() != 3 ||
      value_shape.size() != 3 || query_shape[0] != key_shape[0] ||
      query_shape[0] != value_shape[0] || query_shape[2] != embed_dim_ ||
      key_shape[2] != embed_dim_ || value_shape[2] != embed_dim_) {
    throw std::runtime_error(
        "Invalid input tensor shapes for MultiHeadAttention forward.");
  }

  size_t batch_size = query_shape[0];
  size_t query_sequence_length = query_shape[1];
  size_t key_sequence_length = key_shape[1];

  // Result shape: (batch_size, sequence_length, embed_dim) for q_proj, k_proj,
  // v_proj
  std::shared_ptr<Tensor> q_proj = query_proj_.forward(query);
  std::shared_ptr<Tensor> k_proj = key_proj_.forward(key);
  std::shared_ptr<Tensor> v_proj = value_proj_.forward(value);

  // Reshape from (batch_size, sequence_length, embed_dim) to (batch_size,
  // sequence_length, num_heads, head_dim)
  std::shared_ptr<Tensor> q_reshaped = q_proj->reshape(
      {(int)batch_size, (int)query_sequence_length, num_heads_, head_dim_});
  std::shared_ptr<Tensor> k_reshaped = k_proj->reshape(
      {(int)batch_size, (int)key_sequence_length, num_heads_, head_dim_});
  std::shared_ptr<Tensor> v_reshaped = v_proj->reshape(
      {(int)batch_size, (int)key_sequence_length, num_heads_, head_dim_});

  // Transpose to get (batch_size, num_heads, sequence_length, head_dim)
  // Permutation: [0, 2, 1, 3] for a 4D tensor
  std::vector<int> perm = {0, 2, 1, 3};
  std::shared_ptr<Tensor> q_split_heads = q_reshaped->transpose(perm);
  std::shared_ptr<Tensor> k_split_heads = k_reshaped->transpose(perm);
  std::shared_ptr<Tensor> v_split_heads = v_reshaped->transpose(perm);

  std::shared_ptr<Tensor> attention_output_per_head =
      scaled_dot_product_attention(q_split_heads, k_split_heads, v_split_heads,
                                   mask);

  // Transpose back to (batch_size, query_sequence_length, num_heads, head_dim)
  // Inverse permutation of [0, 2, 1, 3] is [0, 2, 1, 3]
  std::vector<int> inverse_perm = {0, 2, 1, 3};
  std::shared_ptr<Tensor> attention_output_transposed =
      attention_output_per_head->transpose(inverse_perm);

  // Reshape to (batch_size, query_sequence_length, embed_dim)
  std::shared_ptr<Tensor> attention_output_concat =
      attention_output_transposed->reshape(
          {(int)batch_size, (int)query_sequence_length, embed_dim_});

  std::shared_ptr<Tensor> final_output =
      output_proj_.forward(attention_output_concat);

  return final_output;
}
