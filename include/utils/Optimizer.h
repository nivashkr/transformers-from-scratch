#ifndef TRANSFORMER_CPP_OPTIMIZER_H
#define TRANSFORMER_CPP_OPTIMIZER_H

#include "Tensor.h"
#include <memory>
#include <numeric>
#include <vector>

// Base class for all optimizers
class Optimizer {
public:
  // Constructor
  Optimizer() {}

  virtual ~Optimizer() = default;

  // Derived classes must implement this to update parameters based on gradients
  virtual void step() = 0;

  // Zeros the gradients of all parameters managed by this optimizer
  void zero_grad() {
    for (std::shared_ptr<Tensor> &param : parameters_) {
      if (param) {
        param->zero_grad();
      }
    }
  }

protected:
  // List of shared pointers to the parameters to update
  std::vector<std::shared_ptr<Tensor>> &parameters_ =
      Tensor::get_optimizable_tensors();
};

class SGD : public Optimizer {
public:
  // Constructor takes a learning rate
  SGD(float learning_rate) : learning_rate_(learning_rate) {}

  // Destructor
  ~SGD() override = default;

  // Implements the SGD optimization step
  void step() override;

private:
  float learning_rate_;
};

class Adam : public Optimizer {
public:
  // Constructor
  Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
       float epsilon = 1e-8f);

  // Destructor
  ~Adam() override = default;

  // Implements the Adam optimization step
  void step() override;

private:
  float learning_rate_;
  float beta1_;
  float beta2_;
  float epsilon_;
  int t_; // Timestep

  std::vector<std::shared_ptr<Tensor>> m_; // First moment
  std::vector<std::shared_ptr<Tensor>> v_; // Second moment
};

#endif // TRANSFORMER_CPP_OPTIMIZER_H
