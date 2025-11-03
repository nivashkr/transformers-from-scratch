#ifndef TRANSFORMER_CPP_MASKING_H
#define TRANSFORMER_CPP_MASKING_H

#include "Tensor.h"
#include <limits>
#include <memory>
#include <vector>

// Generates a look-ahead mask to prevent attention to future tokens in the
// decoder.
std::shared_ptr<Tensor> create_look_ahead_mask(int sequence_length);

// Generates a padding mask to ignore padded elements in sequences.
std::shared_ptr<Tensor>
create_padding_mask(const std::shared_ptr<Tensor> &input_ids,
                    float pad_token_id = 0.0f);

#endif // TRANSFORMER_CPP_MASKING_H
