#include "layers/PositionwiseFeedForward.h"

PositionwiseFeedForward::PositionwiseFeedForward(int input_dim, int hidden_dim)
    : fc1_(input_dim, hidden_dim), fc2_(hidden_dim, input_dim),
      input_dim_(input_dim), hidden_dim_(hidden_dim) {}

std::shared_ptr<Tensor>
PositionwiseFeedForward::forward(std::shared_ptr<Tensor> &input) {
  // The feed-forward network is applied to the last dimension (input_dim)
  // independently for each position.

  // First linear transformation
  std::shared_ptr<Tensor> output_fc1 = fc1_.forward(input);

  // Apply activation function
  std::shared_ptr<Tensor> activated_output = activation_.forward(output_fc1);

  // Second linear transformation
  std::shared_ptr<Tensor> output_fc2 = fc2_.forward(activated_output);

  return output_fc2;
}