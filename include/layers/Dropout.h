#ifndef TRANSFORMER_CPP_DROPOUT_H
#define TRANSFORMER_CPP_DROPOUT_H

#include "../utils/Tensor.h"
#include <memory>
#include <random>

class Dropout {
public:
  // Constructor
  Dropout(float rate);

  // Forward pass
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input,
                                  bool is_training);

  // Destructor
  ~Dropout() = default;

private:
  float rate_; // Dropout rate

  // Random number generator and distribution for creating the mask
  std::shared_ptr<std::mt19937> generator_;
  std::shared_ptr<std::uniform_real_distribution<float>> distribution_;
};

#endif // TRANSFORMER_CPP_DROPOUT_H
