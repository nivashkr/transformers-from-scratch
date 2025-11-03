#ifndef TRANSFORMER_CPP_LAYERNORM_H
#define TRANSFORMER_CPP_LAYERNORM_H

#include "../utils/Tensor.h"
#include <memory>
#include <vector>

class LayerNorm {
public:
  // Constructor
  LayerNorm(int normalized_shape, float epsilon = 1e-5f);

  // Forward pass
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input);

  // Destructor
  ~LayerNorm() = default;

  // Getters for learnable parameters
  std::shared_ptr<Tensor> get_gamma();
  std::shared_ptr<Tensor> get_beta();

private:
  std::shared_ptr<Tensor> gamma_; // Learnable scale parameter (gain)
  std::shared_ptr<Tensor> beta_;  // Learnable shift parameter (bias)
  float epsilon_;

  int normalized_shape_;

  std::shared_ptr<Tensor> mean_;
  std::shared_ptr<Tensor> inv_stddev_;
  std::shared_ptr<Tensor> centered_input_;
};

#endif // TRANSFORMER_CPP_LAYERNORM_H