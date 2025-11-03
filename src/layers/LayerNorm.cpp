#include "layers/LayerNorm.h"
#include <cmath>
#include <iostream>
#include <numeric>

LayerNorm::LayerNorm(int normalized_shape, float epsilon)
    : normalized_shape_(normalized_shape), epsilon_(epsilon) {
  // Gamma and Beta are learnable parameters, shape is {normalized_shape_}
  gamma_ = Tensor::create(std::vector<int>{normalized_shape_}, true);
  beta_ = Tensor::create(std::vector<int>{normalized_shape_}, true);

  // Initialize gamma to ones and beta to zeros
  std::shared_ptr<std::vector<float>> gamma_data =
      std::make_shared<std::vector<float>>(normalized_shape_, 1.0f);
  gamma_->set_data(gamma_data);

  std::shared_ptr<std::vector<float>> beta_data =
      std::make_shared<std::vector<float>>(normalized_shape_, 0.0f);
  beta_->set_data(beta_data);
}

std::shared_ptr<Tensor>
LayerNorm::forward(const std::shared_ptr<Tensor> &input) {
  const std::vector<int> &input_shape = input->get_shape();
  if (input_shape.empty() || input_shape.back() != normalized_shape_) {
    throw std::runtime_error("Input tensor's last dimension must match "
                             "normalized_shape in LayerNorm.");
  }

  // Calculate mean and variance along the last dimension
  size_t last_dim_size = normalized_shape_;
  size_t num_elements = input->num_elements();
  size_t outer_dims_elements = num_elements / last_dim_size;

  // Store intermediate values for backward pass
  mean_ = Tensor::create(
      std::vector<int>(input_shape.begin(), input_shape.end() - 1));
  std::shared_ptr<std::vector<float>> mean_data =
      std::make_shared<std::vector<float>>(outer_dims_elements);

  std::shared_ptr<Tensor> variance = Tensor::create(
      std::vector<int>(input_shape.begin(), input_shape.end() - 1));
  std::shared_ptr<std::vector<float>> variance_data =
      std::make_shared<std::vector<float>>(outer_dims_elements);

  const std::vector<float> &input_data = input->get_data();

  for (size_t i = 0; i < outer_dims_elements; ++i) {
    size_t start_idx = i * last_dim_size;
    float sum = 0.0f;
    for (size_t j = 0; j < last_dim_size; ++j) {
      sum += input_data[start_idx + j];
    }
    (*mean_data)[i] = sum / last_dim_size;

    float sum_sq_diff = 0.0f;
    for (size_t j = 0; j < last_dim_size; ++j) {
      float diff = input_data[start_idx + j] - (*mean_data)[i];
      sum_sq_diff += diff * diff;
    }
    (*variance_data)[i] = sum_sq_diff / last_dim_size;
  }

  mean_->set_data(mean_data);
  variance->set_data(variance_data);

  // Calculate inverse standard deviation
  inv_stddev_ = Tensor::create(
      std::vector<int>(input_shape.begin(), input_shape.end() - 1));
  std::shared_ptr<std::vector<float>> inv_stddev_data =
      std::make_shared<std::vector<float>>(outer_dims_elements);
  const std::vector<float> &variance_data_const = variance->get_data();

  for (size_t i = 0; i < outer_dims_elements; ++i) {
    (*inv_stddev_data)[i] =
        1.0f / std::sqrt((variance_data_const)[i] + epsilon_);
  }
  inv_stddev_->set_data(inv_stddev_data);

  // Normalize, scale, and shift
  std::shared_ptr<Tensor> output = Tensor::create(input_shape);
  std::vector<float> &output_data = output->data_ref();

  // Store centered input for backward pass
  centered_input_ = Tensor::create(input_shape);
  std::vector<float> &centered_input_data = centered_input_->data_ref();

  const std::vector<float> &gamma_data = gamma_->get_data();
  const std::vector<float> &beta_data = beta_->get_data();

  for (size_t i = 0; i < outer_dims_elements; ++i) {
    size_t start_idx = i * last_dim_size;
    float current_mean = (*mean_data)[i];
    float current_inv_stddev = (*inv_stddev_data)[i];

    for (size_t j = 0; j < last_dim_size; ++j) {
      centered_input_data[start_idx + j] =
          input_data[start_idx + j] - current_mean;
      output_data[start_idx + j] = centered_input_data[start_idx + j] *
                                       current_inv_stddev * gamma_data[j] +
                                   beta_data[j];
    }
  }

  output->set_creator_op(OperationType::LayerNorm);
  output->set_parents({input});
  output->layernorm_gamma_ = gamma_;
  output->layernorm_beta_ = beta_;
  output->layernorm_mean_ = mean_;
  output->layernorm_inv_stddev_ = inv_stddev_;
  output->layernorm_centered_input_ = centered_input_;
  output->layernorm_epsilon_ = epsilon_;

  return output;
}

std::shared_ptr<Tensor> LayerNorm::get_gamma() { return gamma_; }

std::shared_ptr<Tensor> LayerNorm::get_beta() { return beta_; }
