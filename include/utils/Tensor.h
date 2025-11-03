#ifndef TRANSFORMER_CPP_TENSOR_H
#define TRANSFORMER_CPP_TENSOR_H

#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

enum class OperationType {
  None,
  Add,
  Sub,
  Mul,
  Dot,
  Transpose,
  Reshape,
  Sum,
  Div,
  ReLU,
  GELU,
  Sigmoid,
  Tanh,
  LogSoftmax,
  NegativeLogLikelihood,
  LayerNorm,
  Softmax,
  Dropout,
  EmbeddingLookup
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  // Constructor
  Tensor();
  // Constructor with shape
  Tensor(const std::vector<int> &shape, bool is_optimizable = false);
  // Constructor with shape and data (initializer list or vector)
  Tensor(const std::vector<int> &shape,
         const std::shared_ptr<std::vector<float>> &data,
         bool is_optimizable = false);

  // Destructor
  ~Tensor();

  // Intermediate values for different functions
  // Backward permutation
  std::vector<int> forward_permutation_;
  // Backward reshape
  std::vector<int> original_shape_before_reshape_;
  // Layernorm
  std::shared_ptr<Tensor> layernorm_gamma_;
  std::shared_ptr<Tensor> layernorm_beta_;
  std::shared_ptr<Tensor> layernorm_mean_;
  std::shared_ptr<Tensor> layernorm_inv_stddev_;
  std::shared_ptr<Tensor> layernorm_centered_input_;
  float layernorm_epsilon_;
  // Dropout
  std::shared_ptr<Tensor> dropout_mask_;
  float dropout_scale_;
  // Embedding
  std::shared_ptr<Tensor> embedding_indices_;

  // Getters for shape, data, and gradient
  const std::vector<int> &get_shape() const { return shape_; };
  const std::vector<float> &get_data() const { return *data_; };
  const std::vector<float> &get_grad() const { return *grad_; };
  std::vector<float> &data_ref() { return *data_; }
  std::vector<float> &grad_ref() { return *grad_; }

  // Setter for data
  void set_data(const std::shared_ptr<std::vector<float>> &data);
  // Setter of parents
  void set_parents(const std::vector<std::shared_ptr<Tensor>> &parents) {
    parents_ = parents;
  }
  // Setter of creator operation
  void set_creator_op(OperationType op) { creator_op_ = op; }

  // Get list of optimizable tensors
  static std::vector<std::shared_ptr<Tensor>> &get_optimizable_tensors() {
    return optimizable_tensors_;
  }
  // Get element by multi-dimensional index
  float get(const std::vector<int> &indices) const;
  // Set element by multi-dimensional index
  void set(const std::vector<int> &indices, float value);
  // Add a getter for the is_optimizable flag
  bool is_optimizable() const { return is_optimizable_; }

  // Basic tensor operations
  std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &other) const;
  std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &other) const;
  std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &other) const;
  std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &other) const;

  std::shared_ptr<Tensor> dot(const std::shared_ptr<Tensor> &other) const;
  std::shared_ptr<Tensor> sum() const;

  std::shared_ptr<Tensor> transpose(const std::vector<int> &permutation) const;
  std::shared_ptr<Tensor> reshape(const std::vector<int> &new_shape) const;

  std::shared_ptr<Tensor> softmax(int dim = -1) const;

  // Gradient handling
  void zero_grad();
  void backward(const std::shared_ptr<Tensor> &grad_output);

  // Helper to calculate total number of elements from shape
  size_t num_elements() const;

  const std::vector<std::shared_ptr<Tensor>> &get_parents() const {
    return parents_;
  }

  static std::shared_ptr<Tensor> create();
  static std::shared_ptr<Tensor> create(const std::vector<int> &shape,
                                        bool is_optimizable = false);
  static std::shared_ptr<Tensor>
  create(const std::vector<int> &shape,
         const std::shared_ptr<std::vector<float>> &data,
         bool is_optimizable = false);

private:
  static std::vector<std::shared_ptr<Tensor>> optimizable_tensors_;

  std::vector<int> shape_;
  std::shared_ptr<std::vector<float>> data_;
  std::shared_ptr<std::vector<float>> grad_;
  std::vector<std::shared_ptr<Tensor>> parents_;
  bool is_optimizable_ = false;

  OperationType creator_op_ = OperationType::None;

  size_t get_linear_index(const std::vector<int> &indices) const;
  std::vector<size_t> calculate_strides(const std::vector<int> &shape) const;

  static std::vector<int>
  calculate_broadcast_shape(const std::vector<int> &shape1,
                            const std::vector<int> &shape2);
  std::shared_ptr<Tensor> broadcast_to(const std::vector<int> &new_shape) const;
  static bool is_broadcastable(const std::vector<int> &shape1,
                               const std::vector<int> &shape2);
  bool are_shapes_compatible(const Tensor &other) const {
    return shape_ == other.shape_;
  }

  void reduce_gradient(const std::shared_ptr<Tensor> &grad_output,
                       std::shared_ptr<Tensor> &parent_grad,
                       const std::vector<int> &parent_shape);
  void backward_add(const std::shared_ptr<Tensor> &grad_output);
  void backward_sub(const std::shared_ptr<Tensor> &grad_output);
  void backward_mul(const std::shared_ptr<Tensor> &grad_output);
  void backward_div(const std::shared_ptr<Tensor> &grad_output);

  void backward_transpose(const std::shared_ptr<Tensor> &grad_output);
  void backward_reshape(const std::shared_ptr<Tensor> &grad_output);

  void backward_dot(const std::shared_ptr<Tensor> &grad_output);
  void backward_sum(const std::shared_ptr<Tensor> &grad_output);

  void backward_relu(const std::shared_ptr<Tensor> &grad_output);
  void backward_gelu(const std::shared_ptr<Tensor> &grad_output);
  void backward_sigmoid(const std::shared_ptr<Tensor> &grad_output);
  void backward_tanh(const std::shared_ptr<Tensor> &grad_output);
  void backward_logsoftmax(const std::shared_ptr<Tensor> &grad_output);
  void backward_nllloss(const std::shared_ptr<Tensor> &grad_output);
  void backward_layernorm(const std::shared_ptr<Tensor> &grad_output);
  void backward_softmax(const std::shared_ptr<Tensor> &grad_output);
  void backward_dropout(const std::shared_ptr<Tensor> &grad_output);
  void backward_embedding_lookup(const std::shared_ptr<Tensor> &grad_output);
};

#endif // TRANSFORMER_CPP_TENSOR_H