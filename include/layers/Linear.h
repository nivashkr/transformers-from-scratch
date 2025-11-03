#ifndef TRANSFORMER_CPP_LINEAR_H
#define TRANSFORMER_CPP_LINEAR_H

#include "../utils/Tensor.h"
#include <memory>
#include <vector>

class Linear {
public:
  Linear(int input_dim, int output_dim);

  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input);

  // Destructor
  ~Linear();

  std::shared_ptr<Tensor> get_weights();
  std::shared_ptr<Tensor> get_biases();

private:
  std::shared_ptr<Tensor> weights_; // Weight matrix (input_dim, output_dim)
  std::shared_ptr<Tensor> biases_;  // Bias vector (1, output_dim)

  int input_dim_;
  int output_dim_;
};

#endif // TRANSFORMER_CPP_LINEAR_H
