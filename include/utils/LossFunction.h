#ifndef TRANSFORMER_CPP_LOSSFUNCTION_H
#define TRANSFORMER_CPP_LOSSFUNCTION_H

#include "Tensor.h"
#include <memory>

class LossFunction {
public:
  virtual ~LossFunction() = default;

  virtual std::shared_ptr<Tensor>
  compute_loss(std::shared_ptr<Tensor> &predictions,
               std::shared_ptr<Tensor> &targets) = 0;

  // This assumes compute_loss returns a shared_ptr to a Tensor that has its
  // computation graph set up.
  void backward(std::shared_ptr<Tensor> &loss);
};

class MeanSquaredErrorLoss : public LossFunction {
public:
  MeanSquaredErrorLoss() = default;

  ~MeanSquaredErrorLoss() override = default;

  // MSE = mean((predictions - targets)^2)
  std::shared_ptr<Tensor>
  compute_loss(std::shared_ptr<Tensor> &predictions,
               std::shared_ptr<Tensor> &targets) override;
};

class CrossEntropyLoss : public LossFunction {
public:
  CrossEntropyLoss() = default;
  ~CrossEntropyLoss() override = default;

  // Computes the Cross-Entropy Loss
  std::shared_ptr<Tensor>
  compute_loss(std::shared_ptr<Tensor> &predictions,
               std::shared_ptr<Tensor> &targets) override;
};

#endif // TRANSFORMER_CPP_LOSSFUNCTION_H
