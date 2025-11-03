#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <utils/LossFunction.h>

void LossFunction::backward(std::shared_ptr<Tensor> &loss) {
  // The gradient of the loss with respect to itself is 1.0.
  if (loss->get_shape().size() != 1 || loss->get_shape()[0] != 1) {
    throw std::runtime_error(
        "Backward pass for loss must start from a scalar loss tensor.");
  }
  std::shared_ptr<Tensor> initial_grad_for_loss = Tensor::create(
      {1}, std::make_shared<std::vector<float>>(std::vector<float>{{1.0f}}));
  loss->backward(initial_grad_for_loss);
}

std::shared_ptr<Tensor>
MeanSquaredErrorLoss::compute_loss(std::shared_ptr<Tensor> &predictions,
                                   std::shared_ptr<Tensor> &targets) {
  if (predictions->get_shape() != targets->get_shape()) {
    throw std::runtime_error(
        "Prediction and target shapes mismatch in MeanSquaredErrorLoss.");
  }

  std::shared_ptr<Tensor> diff = *predictions - targets;
  std::shared_ptr<Tensor> squared_diff = *diff * diff;
  std::shared_ptr<Tensor> sum_squared_diff = squared_diff->sum();

  size_t num_elements = predictions->num_elements();
  if (num_elements == 0) {
    return Tensor::create(
        std::vector<int>{1},
        std::make_shared<std::vector<float>>(std::vector<float>{0.0f}));
  }

  std::shared_ptr<Tensor> num_elements_tensor = Tensor::create(
      {1}, std::make_shared<std::vector<float>>(
               std::vector<float>{{static_cast<float>(num_elements)}}));
  std::shared_ptr<Tensor> mean_loss = (*sum_squared_diff) / num_elements_tensor;

  return mean_loss;
}

// Helper function for LogSoftmax
std::shared_ptr<Tensor> log_softmax(const std::shared_ptr<Tensor> &input) {
  std::shared_ptr<Tensor> output = Tensor::create(input->get_shape());
  const std::vector<float> &input_data = input->get_data();
  std::vector<float> &output_data = output->data_ref();
  const std::vector<int> &shape = input->get_shape();
  size_t last_dim_size = shape.empty() ? 0 : shape.back();
  size_t num_elements = input->num_elements();

  if (last_dim_size == 0 || num_elements == 0) {
    return output;
  }

  size_t outer_dims_elements = num_elements / last_dim_size;

  for (size_t i = 0; i < outer_dims_elements; ++i) {
    size_t start_idx = i * last_dim_size;
    float max_val = -std::numeric_limits<float>::infinity();

    // Find the maximum value for numerical stability
    for (size_t j = 0; j < last_dim_size; ++j) {
      if (input_data[start_idx + j] > max_val) {
        max_val = input_data[start_idx + j];
      }
    }

    float log_sum_exp = 0.0f;
    for (size_t j = 0; j < last_dim_size; ++j) {
      log_sum_exp += std::exp(input_data[start_idx + j] - max_val);
    }
    log_sum_exp = max_val + std::log(log_sum_exp);

    // Calculate log softmax
    for (size_t j = 0; j < last_dim_size; ++j) {
      output_data[start_idx + j] = input_data[start_idx + j] - log_sum_exp;
    }
  }

  output->set_creator_op(OperationType::LogSoftmax);
  output->set_parents({input});

  return output;
}

// Helper function for Negative Log Likelihood Loss
std::shared_ptr<Tensor>
nll_loss(const std::shared_ptr<Tensor> &log_probabilities,
         const std::shared_ptr<Tensor> &targets) {
  std::shared_ptr<Tensor> loss = Tensor::create(
      std::vector<int>{1},
      std::make_shared<std::vector<float>>(std::vector<float>{0.0f}));
  const std::vector<float> &log_prob_data = log_probabilities->get_data();
  const std::vector<float> &target_data = targets->get_data();
  float total_loss = 0.0f;

  const std::vector<int> &log_prob_shape = log_probabilities->get_shape();
  size_t last_dim_size = log_prob_shape.empty() ? 0 : log_prob_shape.back();
  size_t num_elements_log_prob = log_probabilities->num_elements();

  if (last_dim_size == 0 || num_elements_log_prob == 0) {
    return loss;
  }

  if (log_prob_shape.size() < 1) {
    throw std::runtime_error(
        "Log probabilities tensor must have at least 1 dimension for NLLLoss.");
  }

  size_t outer_dims_elements = num_elements_log_prob / last_dim_size;

  if (targets->num_elements() != outer_dims_elements) {
    throw std::runtime_error("Target data size mismatch with log probabilities "
                             "outer dimensions in NLLLoss.");
  }

  for (size_t i = 0; i < outer_dims_elements; ++i) {
    size_t log_prob_start_idx = i * last_dim_size;
    size_t target_idx = i;

    float target_val = target_data[target_idx];
    if (target_val < 0 || target_val >= static_cast<float>(last_dim_size) ||
        std::fmod(target_val, 1.0f) != 0.0f) {
      throw std::runtime_error(
          "Target class index is out of bounds or not an integer in NLLLoss.");
    }
    int target_class = static_cast<int>(target_val);

    total_loss += -log_prob_data[log_prob_start_idx + target_class];
  }

  // Mean loss
  if (outer_dims_elements > 0) {
    loss->set(std::vector<int>{0}, total_loss / outer_dims_elements);
  } else {
    loss->set(std::vector<int>{0}, 0.0f);
  }

  loss->set_creator_op(OperationType::NegativeLogLikelihood);
  loss->set_parents({log_probabilities, targets});

  return loss;
}

std::shared_ptr<Tensor>
CrossEntropyLoss::compute_loss(std::shared_ptr<Tensor> &predictions,
                               std::shared_ptr<Tensor> &targets) {
  // Apply LogSoftmax to predictions
  std::shared_ptr<Tensor> log_probs = log_softmax(predictions);

  // Compute Negative Log Likelihood Loss
  std::shared_ptr<Tensor> loss = nll_loss(log_probs, targets);

  return loss;
}
