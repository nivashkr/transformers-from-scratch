#include "utils/Masking.h"
#include <algorithm>
#include <iostream>

std::shared_ptr<Tensor> create_look_ahead_mask(int sequence_length) {
  std::shared_ptr<Tensor> mask =
      Tensor::create({sequence_length, sequence_length});
  std::vector<float> &mask_data = mask->data_ref();

  // Create a lower triangular mask with -inf in the upper triangle
  float negative_infinity = -std::numeric_limits<float>::infinity();

  for (int i = 0; i < sequence_length; ++i) {
    for (int j = 0; j < sequence_length; ++j) {
      if (j > i) {
        mask_data[i * sequence_length + j] = negative_infinity;
      } else {
        mask_data[i * sequence_length + j] = 0.0f;
      }
    }
  }

  mask = mask->reshape({1, 1, sequence_length, sequence_length});

  return mask;
}

std::shared_ptr<Tensor>
create_padding_mask(const std::shared_ptr<Tensor> &input_ids,
                    float pad_token_id) {
  const std::vector<int> &input_shape = input_ids->get_shape();
  if (input_shape.size() != 2) {
    throw std::runtime_error("Input IDs for padding mask must be a 2D tensor "
                             "(batch_size, sequence_length).");
  }

  size_t batch_size = input_shape[0];
  size_t sequence_length = input_shape[1];

  std::shared_ptr<Tensor> mask =
      Tensor::create({(int)batch_size, (int)sequence_length});
  std::vector<float> &mask_data = mask->data_ref();
  const std::vector<float> &input_ids_data = input_ids->get_data();

  for (size_t i = 0; i < batch_size * sequence_length; ++i) {
    if (input_ids_data[i] == pad_token_id) {
      mask_data[i] = -std::numeric_limits<float>::infinity();
    } else {
      mask_data[i] = 0.0f;
    }
  }

  // Create a mask of shape (batch_size, 1, 1, sequence_length) and rely on
  // broadcasting rules to expand it over the query sequence length dimension.
  std::shared_ptr<Tensor> final_mask =
      mask->reshape({(int)batch_size, 1, 1, (int)sequence_length});

  return final_mask;
}