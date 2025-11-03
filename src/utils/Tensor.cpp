#include "utils/Tensor.h"
#include "utils/ConfigParser.h"
#include "utils/Helpers.h"
#include "utils/ThreadPool.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>

std::vector<std::shared_ptr<Tensor>> Tensor::optimizable_tensors_;

// Default constructor
Tensor::Tensor()
    : shape_{{}}, data_(std::make_shared<std::vector<float>>()),
      grad_(std::make_shared<std::vector<float>>()),
      creator_op_(OperationType::None), parents_{}, is_optimizable_{false} {}

// Constructor with shape
Tensor::Tensor(const std::vector<int> &shape, bool is_optimizable)
    : shape_{shape}, creator_op_(OperationType::None), parents_{},
      is_optimizable_{is_optimizable} {
  size_t total_elements = num_elements();
  data_ = std::make_shared<std::vector<float>>(total_elements, 0.0f);
  grad_ = std::make_shared<std::vector<float>>(total_elements, 0.0f);
}

// Constructor with shape and data
Tensor::Tensor(const std::vector<int> &shape,
               const std::shared_ptr<std::vector<float>> &data,
               bool is_optimizable)
    : shape_{shape}, creator_op_(OperationType::None), parents_{},
      is_optimizable_{is_optimizable} {
  size_t total_elements = num_elements();
  if (data->size() != total_elements) {
    throw std::runtime_error(
        "Data size does not match the specified shape in constructor.");
  }
  data_ = data;
  grad_ = std::make_shared<std::vector<float>>(total_elements, 0.0f);
}

// Destructor
Tensor::~Tensor() {}

// Default factory
std::shared_ptr<Tensor> Tensor::create() { return std::make_shared<Tensor>(); }

// Factory method with shape
std::shared_ptr<Tensor> Tensor::create(const std::vector<int> &shape,
                                       bool is_optimizable) {
  std::shared_ptr<Tensor> tensor =
      std::make_shared<Tensor>(shape, is_optimizable);
  if (is_optimizable) {
    optimizable_tensors_.push_back(tensor);
  }
  return tensor;
}

// Factory method with shape and data
std::shared_ptr<Tensor>
Tensor::create(const std::vector<int> &shape,
               const std::shared_ptr<std::vector<float>> &data,
               bool is_optimizable) {
  std::shared_ptr<Tensor> tensor =
      std::make_shared<Tensor>(shape, data, is_optimizable);
  if (is_optimizable) {
    optimizable_tensors_.push_back(tensor);
  }
  return tensor;
}

// Setter for data
void Tensor::set_data(const std::shared_ptr<std::vector<float>> &data) {
  size_t total_elements = num_elements();
  if (data->size() != total_elements) {
    throw std::runtime_error("Data size mismatch in set_data.");
  }
  data_ = data;
  grad_->assign(total_elements, 0.0f);
  // When data is explicitly set, this tensor is a leaf node, not derived from
  // an operation.
  creator_op_ = OperationType::None;
  parents_.clear();
}

// Get element by multi-dimensional index
float Tensor::get(const std::vector<int> &indices) const {
  if (!data_)
    throw std::runtime_error("Accessing data on a null shared_ptr.");
  return (*data_)[get_linear_index(indices)];
}

// Set element by multi-dimensional index
void Tensor::set(const std::vector<int> &indices, float value) {
  if (!data_)
    throw std::runtime_error("Accessing data on a null shared_ptr.");
  (*data_)[get_linear_index(indices)] = value;
}

size_t Tensor::num_elements() const {
  if (shape_.empty())
    return 0;
  return std::accumulate(shape_.begin(), shape_.end(), 1,
                         std::multiplies<size_t>());
}

// Helper to calculate the linear index from multi-dimensional indices
size_t Tensor::get_linear_index(const std::vector<int> &indices) const {
  if (indices.size() != shape_.size()) {
    throw std::runtime_error("Number of indices must match tensor dimensions.");
  }
  size_t linear_index = 0;
  const std::vector<size_t> &strides = calculate_strides(shape_);

  for (size_t i = 0; i < shape_.size(); ++i) {
    if (indices[i] < 0 || indices[i] >= shape_[i]) {
      throw std::runtime_error("Index out of bounds.");
    }
    linear_index += indices[i] * strides[i];
  }
  return linear_index;
}

std::vector<size_t>
Tensor::calculate_strides(const std::vector<int> &shape) const {
  std::vector<size_t> strides(shape.size());
  if (!shape.empty()) {
    strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }
  return strides;
}

bool Tensor::is_broadcastable(const std::vector<int> &shape1,
                              const std::vector<int> &shape2) {
  int max_dims = std::max(shape1.size(), shape2.size());
  std::vector<int> padded1(max_dims, 1);
  std::vector<int> padded2(max_dims, 1);

  // Pad shapes with 1s from the left
  std::copy(shape1.rbegin(), shape1.rend(), padded1.rbegin());
  std::copy(shape2.rbegin(), shape2.rend(), padded2.rbegin());

  for (int i = 0; i < max_dims; ++i) {
    if (padded1[i] != padded2[i] && padded1[i] != 1 && padded2[i] != 1) {
      return false;
    }
  }
  return true;
}

std::vector<int>
Tensor::calculate_broadcast_shape(const std::vector<int> &shape1,
                                  const std::vector<int> &shape2) {
  int max_dims = std::max(shape1.size(), shape2.size());
  std::vector<int> padded1(max_dims, 1);
  std::vector<int> padded2(max_dims, 1);
  std::vector<int> result_shape(max_dims);

  std::copy(shape1.rbegin(), shape1.rend(), padded1.rbegin());
  std::copy(shape2.rbegin(), shape2.rend(), padded2.rbegin());

  for (int i = 0; i < max_dims; ++i) {
    if (padded1[i] != padded2[i] && padded1[i] != 1 && padded2[i] != 1) {
      throw std::runtime_error("Shapes are not broadcastable");
    }
    result_shape[i] = std::max(padded1[i], padded2[i]);
  }

  return result_shape;
}

std::shared_ptr<Tensor>
Tensor::broadcast_to(const std::vector<int> &new_shape) const {
  if (!is_broadcastable(shape_, new_shape)) {
    throw std::runtime_error("Cannot broadcast to target shape");
  }

  std::shared_ptr<Tensor> result = Tensor::create(new_shape);
  std::vector<int> input_idx(shape_.size());
  std::vector<int> output_idx(new_shape.size());

  // Iterate through all elements in the result tensor
  size_t total_elements = result->num_elements();
  for (size_t i = 0; i < total_elements; ++i) {
    // Calculate output indices
    size_t temp = i;
    for (int d = new_shape.size() - 1; d >= 0; --d) {
      output_idx[d] = temp % new_shape[d];
      temp /= new_shape[d];
    }

    // Calculate corresponding input indices
    for (int d = 0; d < shape_.size(); ++d) {
      int offset = new_shape.size() - shape_.size();
      input_idx[d] = shape_[d] == 1 ? 0 : output_idx[d + offset];
    }

    result->set(output_idx, get(input_idx));
  }

  return result;
}

// Basic tensor operations
std::shared_ptr<Tensor>
Tensor::operator+(const std::shared_ptr<Tensor> &other) const {
  std::vector<int> result_shape =
      calculate_broadcast_shape(shape_, other->shape_);
  std::shared_ptr<Tensor> broadcasted_a = broadcast_to(result_shape);
  std::shared_ptr<Tensor> broadcasted_b = other->broadcast_to(result_shape);

  std::shared_ptr<Tensor> result = Tensor::create(result_shape);
  result->creator_op_ = OperationType::Add;
  result->parents_.push_back(
      std::const_pointer_cast<Tensor>(shared_from_this()));
  result->parents_.push_back(other);

  size_t total_elements = result->data_->size();
  size_t num_threads = ConfigParser::getInstance().getValue<int>("num_threads");
  if (total_elements < num_threads)
    num_threads = total_elements;
  size_t elements_per_thread = total_elements / num_threads;

  std::vector<std::function<void()>> tasks;
  for (size_t t = 0; t < num_threads; ++t) {
    size_t start_idx = t * elements_per_thread;
    size_t end_idx = (t == num_threads - 1) ? total_elements
                                            : start_idx + elements_per_thread;

    // A lambda task for each chunk of work
    tasks.emplace_back([&, start_idx, end_idx]() {
      for (size_t i = start_idx; i < end_idx; ++i) {
        (*result->data_)[i] =
            (*broadcasted_a->data_)[i] + (*broadcasted_b->data_)[i];
      }
    });
  }

  getThreadPool().run_batch(std::move(tasks));

  return result;
}

std::shared_ptr<Tensor>
Tensor::operator-(const std::shared_ptr<Tensor> &other) const {
  std::vector<int> result_shape =
      calculate_broadcast_shape(shape_, other->shape_);
  std::shared_ptr<Tensor> broadcasted_a = broadcast_to(result_shape);
  std::shared_ptr<Tensor> broadcasted_b = other->broadcast_to(result_shape);

  std::shared_ptr<Tensor> result = Tensor::create(result_shape);
  result->creator_op_ = OperationType::Sub;
  result->parents_.push_back(
      std::const_pointer_cast<Tensor>(shared_from_this()));
  result->parents_.push_back(other);

  size_t total_elements = result->data_->size();
  size_t num_threads = ConfigParser::getInstance().getValue<int>("num_threads");
  if (total_elements < num_threads)
    num_threads = total_elements;
  size_t elements_per_thread = total_elements / num_threads;

  std::vector<std::function<void()>> tasks;
  for (size_t t = 0; t < num_threads; ++t) {
    size_t start_idx = t * elements_per_thread;
    size_t end_idx = (t == num_threads - 1) ? total_elements
                                            : start_idx + elements_per_thread;

    tasks.emplace_back([&, start_idx, end_idx]() {
      for (size_t i = start_idx; i < end_idx; ++i) {
        (*result->data_)[i] =
            (*broadcasted_a->data_)[i] - (*broadcasted_b->data_)[i];
      }
    });
  }

  getThreadPool().run_batch(std::move(tasks));

  return result;
}

std::shared_ptr<Tensor>
Tensor::operator*(const std::shared_ptr<Tensor> &other) const {
  std::vector<int> result_shape =
      calculate_broadcast_shape(shape_, other->shape_);
  std::shared_ptr<Tensor> broadcasted_a = broadcast_to(result_shape);
  std::shared_ptr<Tensor> broadcasted_b = other->broadcast_to(result_shape);

  std::shared_ptr<Tensor> result = Tensor::create(result_shape);
  result->creator_op_ = OperationType::Mul;
  result->parents_.push_back(
      std::const_pointer_cast<Tensor>(shared_from_this()));
  result->parents_.push_back(other);

  size_t total_elements = result->data_->size();
  size_t num_threads = ConfigParser::getInstance().getValue<int>("num_threads");
  if (total_elements < num_threads)
    num_threads = total_elements;
  size_t elements_per_thread = total_elements / num_threads;

  std::vector<std::function<void()>> tasks;
  for (size_t t = 0; t < num_threads; ++t) {
    size_t start_idx = t * elements_per_thread;
    size_t end_idx = (t == num_threads - 1) ? total_elements
                                            : start_idx + elements_per_thread;

    tasks.emplace_back([&, start_idx, end_idx]() {
      for (size_t i = start_idx; i < end_idx; ++i) {
        (*result->data_)[i] =
            (*broadcasted_a->data_)[i] * (*broadcasted_b->data_)[i];
      }
    });
  }

  getThreadPool().run_batch(std::move(tasks));

  return result;
}

std::shared_ptr<Tensor>
Tensor::operator/(const std::shared_ptr<Tensor> &other) const {
  std::vector<int> result_shape =
      calculate_broadcast_shape(shape_, other->shape_);
  std::shared_ptr<Tensor> broadcasted_a = broadcast_to(result_shape);
  std::shared_ptr<Tensor> broadcasted_b = other->broadcast_to(result_shape);

  std::shared_ptr<Tensor> result = Tensor::create(result_shape);
  result->creator_op_ = OperationType::Div;
  result->parents_.push_back(
      std::const_pointer_cast<Tensor>(shared_from_this())); // numerator
  result->parents_.push_back(other);                        // denominator

  size_t total_elements = result->data_->size();
  size_t num_threads = ConfigParser::getInstance().getValue<int>("num_threads");
  if (total_elements < num_threads)
    num_threads = total_elements;
  size_t elements_per_thread = total_elements / num_threads;

  std::vector<std::function<void()>> tasks;
  for (size_t t = 0; t < num_threads; ++t) {
    size_t start_idx = t * elements_per_thread;
    size_t end_idx = (t == num_threads - 1) ? total_elements
                                            : start_idx + elements_per_thread;

    tasks.emplace_back([&, start_idx, end_idx]() {
      for (size_t i = start_idx; i < end_idx; ++i) {
        if ((*broadcasted_b->data_)[i] == 0.0f) {
          throw std::runtime_error(
              "Division by zero encountered in Tensor division.");
        }
        (*result->data_)[i] =
            (*broadcasted_a->data_)[i] / (*broadcasted_b->data_)[i];
      }
    });
  }

  getThreadPool().run_batch(std::move(tasks));

  return result;
}

std::shared_ptr<Tensor>
Tensor::transpose(const std::vector<int> &permutation) const {
  if (permutation.size() != shape_.size()) {
    throw std::runtime_error("Permutation size (" +
                             std::to_string(permutation.size()) +
                             ") must match tensor dimension (" +
                             std::to_string(shape_.size()) + ").");
  }
  // Check if permutation is valid
  std::vector<int> sorted_permutation = permutation;
  std::sort(sorted_permutation.begin(), sorted_permutation.end());
  bool valid_perm = true;
  if (shape_.size() > 0) {
    for (size_t i = 0; i < sorted_permutation.size(); ++i) {
      if (sorted_permutation[i] != static_cast<int>(i)) {
        valid_perm = false;
        break;
      }
    }
  } else if (!permutation.empty()) {
    valid_perm = false;
  }

  if (!valid_perm) {
    std::string perm_str = "[";
    for (size_t k = 0; k < permutation.size(); ++k)
      perm_str += std::to_string(permutation[k]) +
                  (k == permutation.size() - 1 ? "" : ", ");
    perm_str += "]";
    throw std::runtime_error("Invalid permutation " + perm_str +
                             " for transpose.");
  }

  std::vector<int> new_shape(shape_.size());
  for (size_t i = 0; i < shape_.size(); ++i) {
    new_shape[i] = shape_[permutation[i]];
  }

  std::shared_ptr<Tensor> result = Tensor::create(new_shape);
  result->creator_op_ = OperationType::Transpose;
  result->parents_.push_back(
      std::const_pointer_cast<Tensor>(shared_from_this()));
  result->forward_permutation_ = permutation;

  // Efficient Transposition
  if (num_elements() == 0)
    return result;

  std::vector<size_t> old_strides(shape_.size());
  std::vector<size_t> new_strides(new_shape.size());
  old_strides.back() = 1;
  new_strides.back() = 1;
  for (int i = shape_.size() - 2; i >= 0; --i)
    old_strides[i] = old_strides[i + 1] * shape_[i + 1];
  for (int i = new_shape.size() - 2; i >= 0; --i)
    new_strides[i] = new_strides[i + 1] * new_shape[i + 1];

  std::vector<int> current_indices(shape_.size());
  size_t total_elements = num_elements();

  for (size_t i = 0; i < total_elements; ++i) {
    // Calculate original multi-dimensional indices from linear index i
    size_t temp_linear_idx = i;
    for (size_t d = 0; d < shape_.size(); ++d) {
      if (old_strides[d] == 0)
        continue;
      current_indices[d] = temp_linear_idx / old_strides[d];
      temp_linear_idx %= old_strides[d];
    }

    // Calculate new linear index based on permuted indices
    size_t new_linear_idx = 0;
    for (size_t d = 0; d < new_shape.size(); ++d) {
      new_linear_idx += current_indices[permutation[d]] * new_strides[d];
    }

    // Bounds checks
    if (i >= data_->size() || new_linear_idx >= result->data_->size()) {
      throw std::runtime_error(
          "Index out of bounds during transpose calculation.");
    }
    (*result->data_)[new_linear_idx] = (*data_)[i];
  }

  return result;
}

std::shared_ptr<Tensor>
Tensor::reshape(const std::vector<int> &new_shape) const {
  size_t current_elements = num_elements();
  size_t new_num_elements = 1;
  bool has_neg_one = false;
  int neg_one_idx = -1;
  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == -1) {
      if (has_neg_one)
        throw std::runtime_error(
            "Reshape can only have one dimension specified as -1.");
      has_neg_one = true;
      neg_one_idx = i;
    } else if (new_shape[i] <= 0) {
      throw std::runtime_error("Reshape dimensions must be positive or -1.");
    } else {
      new_num_elements *= new_shape[i];
    }
  }

  std::vector<int> actual_new_shape = new_shape;
  if (has_neg_one) {
    if (current_elements == 0 && new_num_elements > 0) {
      throw std::runtime_error("Cannot infer dimension for -1 when reshaping "
                               "an empty tensor to a non-empty one.");
    }
    if (new_num_elements == 0) {
      throw std::runtime_error(
          "Internal error: product of positive dimensions is zero in reshape.");
    }
    if (current_elements % new_num_elements != 0) {
      throw std::runtime_error("Cannot infer dimension for -1: total elements "
                               "not divisible by product of other dimensions.");
    }
    actual_new_shape[neg_one_idx] = current_elements / new_num_elements;
    new_num_elements = current_elements;
  }

  if (current_elements != new_num_elements) {
    std::string old_shape_str = "[";
    for (size_t k = 0; k < shape_.size(); ++k)
      old_shape_str +=
          std::to_string(shape_[k]) + (k == shape_.size() - 1 ? "" : ", ");
    old_shape_str += "]";
    std::string new_shape_str = "[";
    for (size_t k = 0; k < new_shape.size(); ++k)
      new_shape_str += std::to_string(new_shape[k]) +
                       (k == new_shape.size() - 1 ? "" : ", ");
    new_shape_str += "]";
    throw std::runtime_error("Total number of elements must remain the same "
                             "during reshape. Cannot reshape " +
                             old_shape_str + " (" +
                             std::to_string(current_elements) +
                             " elements) to " + new_shape_str + " (" +
                             std::to_string(new_num_elements) + " elements).");
  }

  // For reshape, share the underlying data_ with the original tensor
  std::shared_ptr<Tensor> result = Tensor::create(actual_new_shape);
  result->data_ = shared_from_this()->data_;
  result->grad_ = std::make_shared<std::vector<float>>(new_num_elements, 0.0f);

  result->creator_op_ = OperationType::Reshape;
  result->parents_.push_back(
      std::const_pointer_cast<Tensor>(shared_from_this()));
  result->original_shape_before_reshape_ = shared_from_this()->shape_;

  return result;
}

std::shared_ptr<Tensor>
Tensor::dot(const std::shared_ptr<Tensor> &other) const {
  if (shape_.size() < 1 || other->shape_.size() < 1) {
    throw std::runtime_error(
        "Dot product requires tensors with at least 1 dimension.");
  }
  // Vector dot product
  if (shape_.size() == 1 && other->shape_.size() == 1) {
    if (shape_[0] != other->shape_[0]) {
      throw std::runtime_error(
          "Vector dot product requires vectors of the same size.");
    }
    // Result is a scalar (1D tensor with size 1)
    std::shared_ptr<Tensor> result = Tensor::create(
        std::vector<int>{1},
        std::make_shared<std::vector<float>>(std::vector<float>{0.0f}), false);
    result->creator_op_ = OperationType::Dot;
    result->parents_.push_back(
        std::const_pointer_cast<Tensor>(shared_from_this()));
    result->parents_.push_back(other);

    for (size_t i = 0; i < shape_[0]; ++i) {
      (*result->data_)[0] += (*data_)[i] * (*other->data_)[i];
    }
    return result;
  }
  // Matrix-vector product
  else if (shape_.size() >= 2 && other->shape_.size() == 1) {
    int cols_a = shape_.back();
    int vec_size = other->shape_[0];
    if (cols_a != vec_size) {
      throw std::runtime_error(
          "Matrix-vector product: Inner dimensions must match (" +
          std::to_string(cols_a) + " vs " + std::to_string(vec_size) + ").");
    }
    // Result shape is the matrix shape excluding the last dimension
    std::vector<int> result_shape(shape_.begin(), shape_.end() - 1);
    std::shared_ptr<Tensor> result = Tensor::create(result_shape);
    result->creator_op_ = OperationType::Dot;
    result->parents_.push_back(
        std::const_pointer_cast<Tensor>(shared_from_this()));
    result->parents_.push_back(other);

    size_t outer_elements = result->num_elements();
    size_t matrix_stride = cols_a;

    for (size_t i = 0; i < outer_elements; ++i) {
      float sum = 0.0f;
      size_t matrix_row_start_idx = i * matrix_stride;
      for (int k = 0; k < cols_a; ++k) {
        sum += (*data_)[matrix_row_start_idx + k] * (*other->data_)[k];
      }
      (*result->data_)[i] = sum;
    }
    return result;
  }
  // Matrix-matrix product (batched)
  else if (shape_.size() >= 2 && other->shape_.size() >= 2) {
    int rows_a = shape_[shape_.size() - 2];
    int cols_a = shape_[shape_.size() - 1];
    int rows_b = other->shape_[other->shape_.size() - 2];
    int cols_b = other->shape_[other->shape_.size() - 1];

    if (cols_a != rows_b) {
      throw std::runtime_error(
          "Matrix multiplication: Inner dimensions must match (" +
          std::to_string(cols_a) + " vs " + std::to_string(rows_b) + ").");
    }

    std::vector<int> batch_dims_a(shape_.begin(), shape_.end() - 2);
    std::vector<int> batch_dims_b(other->shape_.begin(),
                                  other->shape_.end() - 2);
    std::vector<int> batch_shape =
        calculate_broadcast_shape(batch_dims_a, batch_dims_b);

    std::vector<int> result_shape = batch_shape;
    result_shape.push_back(rows_a);
    result_shape.push_back(cols_b);

    std::shared_ptr<Tensor> result = Tensor::create(result_shape);
    result->creator_op_ = OperationType::Dot;
    result->parents_.push_back(
        std::const_pointer_cast<Tensor>(shared_from_this()));
    result->parents_.push_back(other);

    std::vector<size_t> stride_a = calculate_strides(shape_);
    std::vector<size_t> stride_b = calculate_strides(other->shape_);
    std::vector<size_t> stride_result = calculate_strides(result_shape);

    size_t total_batches =
        batch_shape.empty()
            ? 1
            : std::accumulate(batch_shape.begin(), batch_shape.end(), size_t{1},
                              std::multiplies<size_t>());
    std::vector<size_t> batch_strides_result = calculate_strides(batch_shape);

    size_t num_output_elements_per_batch = rows_a * cols_b;
    size_t total_output_elements =
        total_batches * num_output_elements_per_batch;

    size_t num_threads =
        ConfigParser::getInstance().getValue<int>("num_threads");
    if (total_output_elements < num_threads)
      num_threads = total_output_elements;
    size_t elements_per_thread = total_output_elements / num_threads;

    std::vector<std::function<void()>> tasks;
    for (size_t t = 0; t < num_threads; ++t) {
      size_t start_linear_idx = t * elements_per_thread;
      size_t end_linear_idx = (t == num_threads - 1)
                                  ? total_output_elements
                                  : start_linear_idx + elements_per_thread;

      tasks.emplace_back([&, start_linear_idx, end_linear_idx, total_batches,
                          num_output_elements_per_batch, rows_a, cols_b, cols_a,
                          stride_a, stride_b, stride_result, batch_shape,
                          batch_strides_result, batch_dims_a, batch_dims_b]() {
        std::vector<int> batch_idx(batch_shape.size());
        for (size_t linear_idx_result = start_linear_idx;
             linear_idx_result < end_linear_idx; ++linear_idx_result) {
          size_t batch_linear_idx =
              linear_idx_result / num_output_elements_per_batch;
          size_t matrix_linear_idx =
              linear_idx_result % num_output_elements_per_batch;

          int i = matrix_linear_idx / cols_b;
          int j = matrix_linear_idx % cols_b;

          size_t temp_batch_linear = batch_linear_idx;
          for (int d = batch_shape.size() - 1; d >= 0; --d) {
            if (batch_strides_result.size() > 0 && batch_strides_result[d] == 0)
              continue;
            if (batch_strides_result.empty()) {
              batch_idx.clear();
              break;
            }
            batch_idx[d] = temp_batch_linear / batch_strides_result[d];
            temp_batch_linear %= batch_strides_result[d];
          }

          size_t offset_a = 0, offset_b = 0, offset_result = 0;
          int batch_offset_a = batch_shape.size() - batch_dims_a.size();
          int batch_offset_b = batch_shape.size() - batch_dims_b.size();

          for (size_t d = 0; d < batch_shape.size(); ++d) {
            if (d >= batch_offset_a && d - batch_offset_a < stride_a.size()) {
              offset_a +=
                  (batch_dims_a[d - batch_offset_a] == 1 ? 0 : batch_idx[d]) *
                  stride_a[d - batch_offset_a];
            } else if (batch_dims_a.empty() && shape_.size() >= 2) {
              offset_a = 0;
            }

            if (d >= batch_offset_b && d - batch_offset_b < stride_b.size()) {
              offset_b +=
                  (batch_dims_b[d - batch_offset_b] == 1 ? 0 : batch_idx[d]) *
                  stride_b[d - batch_offset_b];
            } else if (batch_dims_b.empty() && other->shape_.size() >= 2) {
              offset_b = 0;
            }
            if (!batch_shape.empty() && d < stride_result.size()) {
              offset_result += batch_idx[d] * stride_result[d];
            } else if (batch_shape.empty()) {
              offset_result = 0;
            }
          }

          float sum = 0.0f;
          for (int k = 0; k < cols_a; ++k) {
            size_t idx_a = offset_a + i * stride_a[shape_.size() - 2] +
                           k * stride_a[shape_.size() - 1];
            size_t idx_b = offset_b + k * stride_b[other->shape_.size() - 2] +
                           j * stride_b[other->shape_.size() - 1];

            if (idx_a >= data_->size() || idx_b >= other->data_->size()) {
              throw std::runtime_error(
                  "Index out of bounds during dot product calculation.");
            }
            sum += (*data_)[idx_a] * (*other->data_)[idx_b];
          }
          size_t idx_result = offset_result +
                              i * stride_result[result_shape.size() - 2] +
                              j * stride_result[result_shape.size() - 1];
          if (idx_result >= result->data_->size()) {
            throw std::runtime_error(
                "Result index out of bounds during dot product calculation.");
          }
          (*result->data_)[idx_result] = sum;
        }
      });
    }

    getThreadPool().run_batch(std::move(tasks));

    return result;
  } else {
    throw std::runtime_error("Unsupported shapes for dot product.");
  }
}

std::shared_ptr<Tensor> Tensor::sum() const {
  std::shared_ptr<Tensor> result = Tensor::create(std::vector<int>{1});
  result->creator_op_ = OperationType::Sum;
  result->parents_.push_back(
      std::const_pointer_cast<Tensor>(shared_from_this()));

  float total_sum = 0.0f;
  if (data_ && !data_->empty()) {
    size_t num_elements = data_->size();
    size_t num_threads =
        ConfigParser::getInstance().getValue<int>("num_threads");
    if (num_elements < num_threads)
      num_threads = num_elements;
    size_t elements_per_thread = num_elements / num_threads;

    std::vector<float> partial_sums(num_threads, 0.0f);
    std::vector<std::function<void()>> tasks;

    for (size_t t = 0; t < num_threads; ++t) {
      size_t start_idx = t * elements_per_thread;
      size_t end_idx = (t == num_threads - 1) ? num_elements
                                              : start_idx + elements_per_thread;

      tasks.emplace_back([&, t, start_idx, end_idx]() {
        for (size_t i = start_idx; i < end_idx; ++i) {
          partial_sums[t] += (*data_)[i];
        }
      });
    }

    getThreadPool().run_batch(std::move(tasks));

    for (float partial_sum : partial_sums) {
      total_sum += partial_sum;
    }
  }
  if (result->data_ && !result->data_->empty()) {
    (*result->data_)[0] = total_sum;
  }

  return result;
}

std::shared_ptr<Tensor> Tensor::softmax(int dim) const {
  int actual_dim = (dim == -1) ? shape_.size() - 1 : dim;

  if (actual_dim < 0 || actual_dim >= static_cast<int>(shape_.size())) {
    throw std::runtime_error("Softmax dimension out of bounds.");
  }

  std::shared_ptr<Tensor> output = Tensor::create(shape_);
  const std::vector<float> &input_data = *data_;
  std::vector<float> &output_data = output->data_ref();

  size_t num_elements = shared_from_this()->num_elements();
  if (num_elements == 0)
    return output;

  size_t outer_size = 1;
  for (int i = 0; i < actual_dim; ++i) {
    outer_size *= shape_[i];
  }
  size_t inner_size = 1;
  for (size_t i = actual_dim + 1; i < shape_.size(); ++i) {
    inner_size *= shape_[i];
  }
  size_t dim_size = shape_[actual_dim];

  size_t num_threads = ConfigParser::getInstance().getValue<int>("num_threads");
  if (outer_size * inner_size < num_threads)
    num_threads = outer_size * inner_size;
  size_t work_per_thread = (outer_size * inner_size) / num_threads;

  std::vector<std::function<void()>> tasks;

  for (size_t t = 0; t < num_threads; ++t) {
    size_t start_work_idx = t * work_per_thread;
    size_t end_work_idx = (t == num_threads - 1)
                              ? (outer_size * inner_size)
                              : start_work_idx + work_per_thread;

    tasks.emplace_back([&, start_work_idx, end_work_idx, inner_size,
                        dim_size]() {
      for (size_t work_idx = start_work_idx; work_idx < end_work_idx;
           ++work_idx) {
        size_t i = work_idx / inner_size;
        size_t k = work_idx % inner_size;

        size_t start_idx = i * dim_size * inner_size + k;

        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < dim_size; ++j) {
          max_val = std::max(max_val, input_data[start_idx + j * inner_size]);
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < dim_size; ++j) {
          sum_exp += std::exp(input_data[start_idx + j * inner_size] - max_val);
        }

        for (size_t j = 0; j < dim_size; ++j) {
          output_data[start_idx + j * inner_size] =
              std::exp(input_data[start_idx + j * inner_size] - max_val) /
              sum_exp;
        }
      }
    });
  }

  getThreadPool().run_batch(std::move(tasks));

  output->creator_op_ = OperationType::Softmax;
  output->parents_.push_back(
      std::const_pointer_cast<Tensor>(shared_from_this()));

  return output;
}

// Gradient handling
void Tensor::zero_grad() {
  if (grad_) {
    grad_->assign(num_elements(), 0.0f);
  }
}

void Tensor::backward(const std::shared_ptr<Tensor> &grad_output) {
  if (shape_ != grad_output->get_shape()) {
    throw std::runtime_error("Gradient shape mismatch in backward.");
  }

  // Accumulate the incoming gradient
  if (grad_ && grad_output->data_) {
    for (size_t i = 0; i < grad_->size(); ++i) {
      (*grad_)[i] += (*grad_output->data_)[i];
    }
  }

  // If this tensor was the result of an operation, propagate the gradient to
  // its parents
  if (creator_op_ != OperationType::None && !parents_.empty()) {
    switch (creator_op_) {
    case OperationType::Add:
      backward_add(grad_output);
      break;
    case OperationType::Sub:
      backward_sub(grad_output);
      break;
    case OperationType::Mul:
      backward_mul(grad_output);
      break;
    case OperationType::Div:
      backward_div(grad_output);
      break;
    case OperationType::Transpose:
      backward_transpose(grad_output);
      break;
    case OperationType::Reshape:
      backward_reshape(grad_output);
      break;
    case OperationType::Dot:
      backward_dot(grad_output);
      break;
    case OperationType::Sum:
      backward_sum(grad_output);
      break;
    case OperationType::ReLU:
      backward_relu(grad_output);
      break;
    case OperationType::GELU:
      backward_gelu(grad_output);
      break;
    case OperationType::Sigmoid:
      backward_sigmoid(grad_output);
      break;
    case OperationType::Tanh:
      backward_tanh(grad_output);
      break;
    case OperationType::LogSoftmax:
      backward_logsoftmax(grad_output);
      break;
    case OperationType::NegativeLogLikelihood:
      backward_nllloss(grad_output);
      break;
    case OperationType::LayerNorm:
      backward_layernorm(grad_output);
      break;
    case OperationType::Softmax:
      backward_softmax(grad_output);
      break;
    case OperationType::EmbeddingLookup:
      backward_embedding_lookup(grad_output);
      break;
    }
  }
}

// Helper to reduce gradient for broadcasting
void Tensor::reduce_gradient(const std::shared_ptr<Tensor> &grad_output,
                             std::shared_ptr<Tensor> &parent_grad,
                             const std::vector<int> &parent_shape) {
  const std::vector<int> &grad_shape = grad_output->get_shape();

  if (grad_shape == parent_shape) {
    parent_grad = Tensor::create(parent_shape);
    parent_grad->set_data(grad_output->data_);
    return;
  }

  parent_grad = Tensor::create(parent_shape);

  std::vector<int> dims_to_sum;
  int max_dims = std::max(grad_shape.size(), parent_shape.size());
  int grad_dim_offset = max_dims - grad_shape.size();
  int parent_dim_offset = max_dims - parent_shape.size();

  for (int i = 0; i < max_dims; ++i) {
    int grad_dim = (i < grad_dim_offset) ? 1 : grad_shape[i - grad_dim_offset];
    int parent_dim =
        (i < parent_dim_offset) ? 1 : parent_shape[i - parent_dim_offset];

    if (parent_dim == 1 && grad_dim > 1) {
      dims_to_sum.push_back(i);
    } else if (parent_dim != grad_dim && parent_dim != 1) {
      throw std::runtime_error(
          "Inconsistent shapes found during gradient reduction.");
    }
  }

  size_t parent_total_elements = parent_grad->num_elements();
  if (parent_total_elements == 0)
    return;

  std::vector<size_t> parent_strides = calculate_strides(parent_shape);
  std::vector<size_t> grad_strides = calculate_strides(grad_shape);

  size_t num_threads = ConfigParser::getInstance().getValue<int>("num_threads");
  if (parent_total_elements < num_threads)
    num_threads = parent_total_elements;
  size_t elements_per_thread = parent_total_elements / num_threads;

  std::vector<std::function<void()>> tasks;

  for (size_t t = 0; t < num_threads; ++t) {
    size_t start_p_idx = t * elements_per_thread;
    size_t end_p_idx = (t == num_threads - 1)
                           ? parent_total_elements
                           : start_p_idx + elements_per_thread;

    tasks.emplace_back([&, start_p_idx, end_p_idx, grad_output, parent_grad,
                        parent_shape, grad_shape, parent_strides,
                        grad_strides]() {
      std::vector<int> parent_indices(parent_shape.size());
      size_t grad_total_elements = grad_output->num_elements();
      std::vector<int> current_grad_indices(grad_shape.size());
      int p_offset = grad_shape.size() - parent_shape.size();

      for (size_t p_idx = start_p_idx; p_idx < end_p_idx; ++p_idx) {
        size_t temp_p_idx = p_idx;
        for (int d = parent_shape.size() - 1; d >= 0; --d) {
          if (parent_strides.size() > 0 && parent_strides[d] == 0)
            continue;
          if (parent_strides.empty())
            break;
          parent_indices[d] = temp_p_idx / parent_strides[d];
          temp_p_idx %= parent_strides[d];
        }

        float sum_val = 0.0f;
        if (grad_output->data_) {
          for (size_t g_idx = 0; g_idx < grad_total_elements; ++g_idx) {
            size_t temp_g_idx = g_idx;
            for (int d = grad_shape.size() - 1; d >= 0; --d) {
              if (grad_strides.size() > 0 && grad_strides[d] == 0)
                continue;
              if (grad_strides.empty())
                break;
              current_grad_indices[d] = temp_g_idx / grad_strides[d];
              temp_g_idx %= grad_strides[d];
            }

            bool corresponds = true;
            for (size_t d = 0; d < parent_shape.size(); ++d) {
              if (parent_shape.size() > 0 && d < parent_indices.size() &&
                  p_offset + d < current_grad_indices.size()) {
                if (parent_shape[d] != 1 &&
                    parent_indices[d] != current_grad_indices[p_offset + d]) {
                  corresponds = false;
                  break;
                }
              } else if (parent_shape.size() > 0 && parent_shape[d] != 1) {
                corresponds = false;
                break;
              }
            }

            if (corresponds) {
              sum_val += (*grad_output->data_)[g_idx];
            }
          }
        }

        if (parent_grad->data_ && p_idx < parent_grad->data_->size()) {
          (*parent_grad->data_)[p_idx] = sum_val;
        }
      }
    });
  }

  getThreadPool().run_batch(std::move(tasks));
}

// Backward methods for specific operations
void Tensor::backward_add(const std::shared_ptr<Tensor> &grad_output) {
  /*
  Gradient of Z = A + B with respect to A is dZ/dA = 1.
  Gradient of Z = A + B with respect to B is dZ/dB = 1.
  Chain rule: dL/dA = dL/dZ * dZ/dA = grad_output * 1 = grad_output
  dL/dB = dL/dZ * dZ/dB = grad_output * 1 = grad_output
  */

  if (parents_.size() == 2) {
    std::shared_ptr<Tensor> parent_a = parents_[0];
    std::shared_ptr<Tensor> parent_b = parents_[1];

    bool a_needs_grad = parent_a->is_optimizable_ ||
                        parent_a->creator_op_ != OperationType::None;

    if (a_needs_grad) {
      std::shared_ptr<Tensor> grad_a_propagated = Tensor::create();
      reduce_gradient(grad_output, grad_a_propagated, parent_a->get_shape());
      parent_a->backward(grad_a_propagated);
    }

    bool b_needs_grad = parent_b->is_optimizable_ ||
                        parent_b->creator_op_ != OperationType::None;

    if (b_needs_grad) {
      std::shared_ptr<Tensor> grad_b_propagated = Tensor::create();
      reduce_gradient(grad_output, grad_b_propagated, parent_b->get_shape());
      parent_b->backward(grad_b_propagated);
    }
  } else {
    std::cerr << "Error: Add operation expected 2 parents, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_sub(const std::shared_ptr<Tensor> &grad_output) {
  /*
  Gradient of Z = A - B with respect to A is dZ/dA = 1.
  Gradient of Z = A - B with respect to B is dZ/dB = -1.
  Chain rule: dL/dA = dL/dZ * dZ/dA = grad_output * 1 = grad_output
  dL/dB = dL/dZ * dZ/dB = grad_output * -1 = -grad_output
  */

  if (parents_.size() == 2) {
    std::shared_ptr<Tensor> parent_a = parents_[0];
    std::shared_ptr<Tensor> parent_b = parents_[1];

    std::shared_ptr<Tensor> grad_a_propagated = Tensor::create();
    reduce_gradient(grad_output, grad_a_propagated, parent_a->get_shape());
    parent_a->backward(grad_a_propagated);

    // Need to propagate -grad_output
    std::shared_ptr<Tensor> neg_grad_output =
        Tensor::create(grad_output->get_shape());
    if (neg_grad_output->data_ && grad_output->data_) {
      std::shared_ptr<std::vector<float>> neg_data =
          std::make_shared<std::vector<float>>(grad_output->num_elements());
      const std::vector<float> &grad_output_data = grad_output->get_data();
      for (size_t i = 0; i < neg_data->size(); ++i) {
        (*neg_data)[i] = -grad_output_data[i];
      }
      neg_grad_output->set_data(neg_data);

      std::shared_ptr<Tensor> grad_b_propagated = Tensor::create();
      reduce_gradient(neg_grad_output, grad_b_propagated,
                      parent_b->get_shape());
      parent_b->backward(grad_b_propagated);
    } else {
      std::cerr
          << "Error: Data vector missing in backward_sub gradient calculation."
          << std::endl;
      // Consider throwing an exception or handling more robustly
    }
  } else {
    std::cerr << "Error: Sub operation expected 2 parents, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_mul(const std::shared_ptr<Tensor> &grad_output) {
  /*
  Gradient of Z = A * B (element-wise) with respect to A is dZ/dA = B.
  Gradient of Z = A * B (element-wise) with respect to B is dZ/dB = A.
  Chain rule: dL/dA = dL/dZ * dZ/dA = grad_output * B
  dL/dB = dL/dZ * dZ/dB = grad_output * A
  */
  if (parents_.size() == 2) {
    std::shared_ptr<Tensor> parent_a = parents_[0];
    std::shared_ptr<Tensor> parent_b = parents_[1];

    // grad_a = grad_output * parent_b
    std::vector<int> intermediate_shape_a = calculate_broadcast_shape(
        grad_output->get_shape(), parent_b->get_shape());
    std::shared_ptr<Tensor> grad_a_intermediate_tensor =
        Tensor::create(intermediate_shape_a);
    std::shared_ptr<Tensor> broadcasted_grad_a =
        grad_output->broadcast_to(intermediate_shape_a);
    std::shared_ptr<Tensor> broadcasted_parent_b =
        parent_b->broadcast_to(intermediate_shape_a);
    if (grad_a_intermediate_tensor->data_ && broadcasted_grad_a->data_ &&
        broadcasted_parent_b->data_) {
      for (size_t i = 0; i < grad_a_intermediate_tensor->data_->size(); ++i) {
        (*grad_a_intermediate_tensor->data_)[i] =
            (*broadcasted_grad_a->data_)[i] * (*broadcasted_parent_b->data_)[i];
      }
    } else { /* Handle missing data error */
      std::cerr << "Error: Missing data in backward_mul for parent A grad calc."
                << std::endl;
    }

    std::shared_ptr<Tensor> grad_a_propagated = Tensor::create();
    reduce_gradient(grad_a_intermediate_tensor, grad_a_propagated,
                    parent_a->get_shape());
    parent_a->backward(grad_a_propagated);

    // grad_b = grad_output * parent_a
    std::vector<int> intermediate_shape_b = calculate_broadcast_shape(
        grad_output->get_shape(), parent_a->get_shape());
    std::shared_ptr<Tensor> grad_b_intermediate_tensor =
        Tensor::create(intermediate_shape_b);
    std::shared_ptr<Tensor> broadcasted_grad_b =
        grad_output->broadcast_to(intermediate_shape_b);
    std::shared_ptr<Tensor> broadcasted_parent_a =
        parent_a->broadcast_to(intermediate_shape_b);
    if (grad_b_intermediate_tensor->data_ && broadcasted_grad_b->data_ &&
        broadcasted_parent_a->data_) {
      for (size_t i = 0; i < grad_b_intermediate_tensor->data_->size(); ++i) {
        (*grad_b_intermediate_tensor->data_)[i] =
            (*broadcasted_grad_b->data_)[i] * (*broadcasted_parent_a->data_)[i];
      }
    } else { /* Handle missing data error */
      std::cerr << "Error: Missing data in backward_mul for parent B grad calc."
                << std::endl;
    }

    std::shared_ptr<Tensor> grad_b_propagated = Tensor::create();
    reduce_gradient(grad_b_intermediate_tensor, grad_b_propagated,
                    parent_b->get_shape());
    parent_b->backward(grad_b_propagated);
  } else {
    std::cerr << "Error: Mul operation expected 2 parents, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_div(const std::shared_ptr<Tensor> &grad_output) {
  /*
  Backward pass for Z = A / B (element-wise):
  dL/dA = dL/dZ * dZ/dA = grad_output * (1 / B)
  dL/dB = dL/dZ * dZ/dB = grad_output * (-A / B^2)
  */

  if (parents_.size() == 2) {
    std::shared_ptr<Tensor> parent_a = parents_[0]; // Numerator
    std::shared_ptr<Tensor> parent_b = parents_[1]; // Denominator

    // grad_a = grad_output / parent_b
    std::vector<int> intermediate_shape_a = calculate_broadcast_shape(
        grad_output->get_shape(), parent_b->get_shape());
    std::shared_ptr<Tensor> grad_a_intermediate_tensor =
        Tensor::create(intermediate_shape_a);
    std::shared_ptr<Tensor> broadcasted_grad_a =
        grad_output->broadcast_to(intermediate_shape_a);
    std::shared_ptr<Tensor> broadcasted_parent_b_for_a =
        parent_b->broadcast_to(intermediate_shape_a);
    if (grad_a_intermediate_tensor->data_ && broadcasted_grad_a->data_ &&
        broadcasted_parent_b_for_a->data_) {
      for (size_t i = 0; i < grad_a_intermediate_tensor->data_->size(); ++i) {
        float denom = (*broadcasted_parent_b_for_a->data_)[i];
        (*grad_a_intermediate_tensor->data_)[i] =
            (denom == 0.0f) ? 0.0f : ((*broadcasted_grad_a->data_)[i] / denom);
      }
    } else {
      std::cerr << "Error: Missing data in backward_div for parent A grad calc."
                << std::endl;
    }

    std::shared_ptr<Tensor> grad_a_propagated = Tensor::create();
    reduce_gradient(grad_a_intermediate_tensor, grad_a_propagated,
                    parent_a->get_shape());
    parent_a->backward(grad_a_propagated);

    // grad_b = grad_output * (-parent_a / parent_b^2)
    std::vector<int> intermediate_shape_b = calculate_broadcast_shape(
        calculate_broadcast_shape(grad_output->get_shape(),
                                  parent_a->get_shape()),
        parent_b->get_shape());
    std::shared_ptr<Tensor> grad_b_intermediate_tensor =
        Tensor::create(intermediate_shape_b);
    std::shared_ptr<Tensor> broadcasted_grad_b =
        grad_output->broadcast_to(intermediate_shape_b);
    std::shared_ptr<Tensor> broadcasted_parent_a_for_b =
        parent_a->broadcast_to(intermediate_shape_b);
    std::shared_ptr<Tensor> broadcasted_parent_b_for_b =
        parent_b->broadcast_to(intermediate_shape_b);
    if (grad_b_intermediate_tensor->data_ && broadcasted_grad_b->data_ &&
        broadcasted_parent_a_for_b->data_ &&
        broadcasted_parent_b_for_b->data_) {
      for (size_t i = 0; i < grad_b_intermediate_tensor->data_->size(); ++i) {
        float numer = (*broadcasted_parent_a_for_b->data_)[i];
        float denom = (*broadcasted_parent_b_for_b->data_)[i];
        float denom_sq = denom * denom;
        (*grad_b_intermediate_tensor->data_)[i] =
            (denom_sq == 0.0f)
                ? 0.0f
                : ((*broadcasted_grad_b->data_)[i] * (-numer / denom_sq));
      }
    } else {
      std::cerr << "Error: Missing data in backward_div for parent B grad calc."
                << std::endl;
    }

    std::shared_ptr<Tensor> grad_b_propagated = Tensor::create();
    reduce_gradient(grad_b_intermediate_tensor, grad_b_propagated,
                    parent_b->get_shape());
    parent_b->backward(grad_b_propagated);
  } else {
    std::cerr << "Error: Division operation expected 2 parents, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_transpose(const std::shared_ptr<Tensor> &grad_output) {
  /*
  Backward pass for Y = X.transpose(p):
  dL/dX = dL/dY.transpose(p_inverse)
  We use the stored forward permutation to calculate the inverse permutation.
  */

  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> parent = parents_[0];

    // Calculate the inverse permutation from the stored forward permutation.
    std::vector<int> inverse_permutation(forward_permutation_.size());
    for (size_t i = 0; i < forward_permutation_.size(); ++i) {
      inverse_permutation[forward_permutation_[i]] = i;
    }

    // Transpose the incoming gradient using the inverse permutation.
    std::shared_ptr<Tensor> grad_input =
        grad_output->transpose(inverse_permutation);

    std::shared_ptr<Tensor> grad_input_tensor = Tensor::create(
        parent->get_shape(),
        std::make_shared<std::vector<float>>(grad_input->get_data()));
    parent->backward(grad_input_tensor);
  } else {
    std::cerr << "Error: Transpose operation expected 1 parent, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_reshape(const std::shared_ptr<Tensor> &grad_output) {
  /*
  Backward pass for Y = X.reshape(new_shape):
  dL/dX = dL/dY.reshape(original_shape_of_X)
  We use the stored original shape before reshape to reshape the gradient.
  */

  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> parent = parents_[0];

    // Ensure the incoming gradient has the same number of elements as the
    // parent's original shape
    size_t expected_elements = std::accumulate(
        original_shape_before_reshape_.begin(),
        original_shape_before_reshape_.end(), 1, std::multiplies<size_t>());
    if (grad_output->num_elements() != expected_elements) {
      throw std::runtime_error(
          "Gradient element count mismatch during reshape backward pass.");
    }

    std::shared_ptr<Tensor> grad_input_reshaped_tensor = Tensor::create(
        original_shape_before_reshape_,
        std::make_shared<std::vector<float>>(grad_output->get_data()));

    parent->backward(grad_input_reshaped_tensor);
  } else {
    std::cerr << "Error: Reshape operation expected 1 parent, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_dot(const std::shared_ptr<Tensor> &grad_output) {
  /*
  Backward pass for Z = A.dot(B):
  dL/dA = dL/dZ . B^T
  dL/dB = A^T . dL/dZ
  This requires implementing matrix multiplication for gradients and handling
  batch dimensions and broadcasting correctly.
  */
  if (parents_.size() == 2) {
    std::shared_ptr<Tensor> parent_a = parents_[0];
    std::shared_ptr<Tensor> parent_b = parents_[1];

    // Calculate dL/dA = dL/dZ . B^T
    std::vector<int> b_transpose_perm(parent_b->get_shape().size());
    std::iota(b_transpose_perm.begin(), b_transpose_perm.end(), 0);
    if (b_transpose_perm.size() >= 2) {
      std::swap(b_transpose_perm[b_transpose_perm.size() - 1],
                b_transpose_perm[b_transpose_perm.size() - 2]);
    }
    std::shared_ptr<Tensor> parent_b_transposed =
        parent_b->transpose(b_transpose_perm);

    // Optimized matrix multiplication for grad_output . B^T
    std::shared_ptr<Tensor> grad_a_intermediate =
        grad_output->dot(parent_b_transposed);

    // Reduce gradient for parent A
    std::shared_ptr<Tensor> grad_a_propagated = Tensor::create();
    reduce_gradient(grad_a_intermediate, grad_a_propagated,
                    parent_a->get_shape());
    parent_a->backward(grad_a_propagated);

    // Calculate dL/dB = A^T . dL/dZ
    std::vector<int> a_transpose_perm(parent_a->get_shape().size());
    std::iota(a_transpose_perm.begin(), a_transpose_perm.end(), 0);
    if (a_transpose_perm.size() >= 2) {
      std::swap(a_transpose_perm[a_transpose_perm.size() - 1],
                a_transpose_perm[a_transpose_perm.size() - 2]);
    }
    std::shared_ptr<Tensor> parent_a_transposed =
        parent_a->transpose(a_transpose_perm);

    // Optimized matrix multiplication for A^T . grad_output
    std::shared_ptr<Tensor> grad_b_intermediate =
        parent_a_transposed->dot(grad_output);

    // Reduce gradient for parent B
    std::shared_ptr<Tensor> grad_b_propagated = Tensor::create();
    reduce_gradient(grad_b_intermediate, grad_b_propagated,
                    parent_b->get_shape());
    parent_b->backward(grad_b_propagated);
  } else {
    std::cerr << "Error: Dot operation expected 2 parents, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_sum(const std::shared_ptr<Tensor> &grad_output) {
  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> parent = parents_[0];

    if (grad_output->get_shape().size() != 1 ||
        grad_output->get_shape()[0] != 1) {
      throw std::runtime_error("Gradient for sum operation must be a scalar.");
    }
    float grad_value = grad_output->get_data()[0];

    std::shared_ptr<Tensor> grad_input_tensor =
        Tensor::create(parent->get_shape());

    size_t total_elements = grad_input_tensor->num_elements();
    size_t num_threads =
        ConfigParser::getInstance().getValue<int>("num_threads");
    if (total_elements < num_threads)
      num_threads = total_elements;
    size_t elements_per_thread = total_elements / num_threads;

    std::vector<std::function<void()>> tasks;
    for (size_t t = 0; t < num_threads; ++t) {
      size_t start_idx = t * elements_per_thread;
      size_t end_idx = (t == num_threads - 1) ? total_elements
                                              : start_idx + elements_per_thread;

      tasks.emplace_back([&, grad_value, start_idx, end_idx]() {
        std::fill(grad_input_tensor->data_->begin() + start_idx,
                  grad_input_tensor->data_->begin() + end_idx, grad_value);
      });
    }

    getThreadPool().run_batch(std::move(tasks));

    parent->backward(grad_input_tensor);
  } else {
    std::cerr << "Error: Sum operation expected 1 parent, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_relu(const std::shared_ptr<Tensor> &grad_output) {
  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> parent = parents_[0];
    std::shared_ptr<Tensor> grad_input = Tensor::create(parent->get_shape());
    const std::vector<float> &parent_data = parent->get_data();
    const std::vector<float> &grad_output_data = grad_output->get_data();
    std::vector<float> &grad_input_data = grad_input->data_ref();

    size_t total_elements = parent_data.size();
    size_t num_threads =
        ConfigParser::getInstance().getValue<int>("num_threads");
    if (total_elements < num_threads)
      num_threads = total_elements;
    size_t elements_per_thread = total_elements / num_threads;

    std::vector<std::function<void()>> tasks;
    for (size_t t = 0; t < num_threads; ++t) {
      size_t start_idx = t * elements_per_thread;
      size_t end_idx = (t == num_threads - 1) ? total_elements
                                              : start_idx + elements_per_thread;

      tasks.emplace_back([&, start_idx, end_idx]() {
        for (size_t i = start_idx; i < end_idx; ++i) {
          if (parent_data[i] > 0) {
            grad_input_data[i] = grad_output_data[i];
          } else {
            grad_input_data[i] = 0.0f;
          }
        }
      });
    }

    getThreadPool().run_batch(std::move(tasks));
    parent->backward(grad_input);
  } else {
    std::cerr << "Error: ReLU operation expected 1 parent, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_gelu(const std::shared_ptr<Tensor> &grad_output) {
  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> parent = parents_[0];
    std::shared_ptr<Tensor> grad_input = Tensor::create(parent->get_shape());
    const std::vector<float> &parent_data = parent->get_data();
    const std::vector<float> &grad_output_data = grad_output->get_data();
    std::vector<float> &grad_input_data = grad_input->data_ref();

    const float M_SQRT2_OVER_PI = 0.7978845608028654f; // sqrt(2 / PI)
    const float GELU_CONSTANT = 0.044715f;

    size_t total_elements = parent_data.size();
    size_t num_threads =
        ConfigParser::getInstance().getValue<int>("num_threads");
    if (total_elements < num_threads)
      num_threads = total_elements;
    size_t elements_per_thread = total_elements / num_threads;

    std::vector<std::function<void()>> tasks;
    for (size_t t = 0; t < num_threads; ++t) {
      size_t start_idx = t * elements_per_thread;
      size_t end_idx = (t == num_threads - 1) ? total_elements
                                              : start_idx + elements_per_thread;

      tasks.emplace_back([&, start_idx, end_idx]() {
        for (size_t i = start_idx; i < end_idx; ++i) {
          float x = parent_data[i];
          float x3 = x * x * x;
          float tanh_arg = M_SQRT2_OVER_PI * (x + GELU_CONSTANT * x3);
          float tanh_val = std::tanh(tanh_arg);
          float sech_sq = 1.0f - tanh_val * tanh_val;
          float derivative = 0.5f * (1.0f + tanh_val) +
                             0.5f * x * sech_sq * M_SQRT2_OVER_PI *
                                 (1.0f + 3.0f * GELU_CONSTANT * x * x);
          grad_input_data[i] = grad_output_data[i] * derivative;
        }
      });
    }

    getThreadPool().run_batch(std::move(tasks));

    parent->backward(grad_input);
  } else {
    std::cerr << "Error: GELU operation expected 1 parent, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_sigmoid(const std::shared_ptr<Tensor> &grad_output) {
  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> parent = parents_[0];
    std::shared_ptr<Tensor> grad_input = Tensor::create(parent->get_shape());
    const std::vector<float> &parent_data = parent->get_data();
    const std::vector<float> &grad_output_data = grad_output->get_data();
    std::vector<float> &grad_input_data = grad_input->data_ref();

    size_t total_elements = parent_data.size();
    size_t num_threads =
        ConfigParser::getInstance().getValue<int>("num_threads");
    if (total_elements < num_threads)
      num_threads = total_elements;
    size_t elements_per_thread = total_elements / num_threads;

    std::vector<std::function<void()>> tasks;
    for (size_t t = 0; t < num_threads; ++t) {
      size_t start_idx = t * elements_per_thread;
      size_t end_idx = (t == num_threads - 1) ? total_elements
                                              : start_idx + elements_per_thread;

      tasks.emplace_back([&, start_idx, end_idx]() {
        for (size_t i = start_idx; i < end_idx; ++i) {
          float sigmoid_x = 1.0f / (1.0f + std::exp(-parent_data[i]));
          grad_input_data[i] =
              grad_output_data[i] * (sigmoid_x * (1.0f - sigmoid_x));
        }
      });
    }

    getThreadPool().run_batch(std::move(tasks));

    parent->backward(grad_input);
  } else {
    std::cerr << "Error: Sigmoid operation expected 1 parent, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_tanh(const std::shared_ptr<Tensor> &grad_output) {
  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> parent = parents_[0];
    std::shared_ptr<Tensor> grad_input = Tensor::create(parent->get_shape());
    const std::vector<float> &parent_data = parent->get_data();
    const std::vector<float> &grad_output_data = grad_output->get_data();
    std::vector<float> &grad_input_data = grad_input->data_ref();

    size_t total_elements = parent_data.size();
    size_t num_threads =
        ConfigParser::getInstance().getValue<int>("num_threads");
    if (total_elements < num_threads)
      num_threads = total_elements;
    size_t elements_per_thread = total_elements / num_threads;

    std::vector<std::function<void()>> tasks;
    for (size_t t = 0; t < num_threads; ++t) {
      size_t start_idx = t * elements_per_thread;
      size_t end_idx = (t == num_threads - 1) ? total_elements
                                              : start_idx + elements_per_thread;

      tasks.emplace_back([&, start_idx, end_idx]() {
        for (size_t i = start_idx; i < end_idx; ++i) {
          float tanh_x = std::tanh(parent_data[i]);
          grad_input_data[i] = grad_output_data[i] * (1.0f - tanh_x * tanh_x);
        }
      });
    }

    getThreadPool().run_batch(std::move(tasks));

    parent->backward(grad_input);
  } else {
    std::cerr << "Error: Tanh operation expected 1 parent, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_logsoftmax(const std::shared_ptr<Tensor> &grad_output) {
  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> parent = parents_[0];
    std::shared_ptr<Tensor> grad_input_intermediate =
        Tensor::create(shared_from_this()->get_shape());
    const std::vector<float> &output_data = shared_from_this()->get_data();
    const std::vector<float> &grad_output_data = grad_output->get_data();
    std::vector<float> &grad_input_intermediate_data = grad_input_intermediate->data_ref();

    const std::vector<int> &shape = shared_from_this()->get_shape();
    size_t last_dim_size = shape.empty() ? 0 : shape.back();
    size_t num_elements = shared_from_this()->num_elements();

    if (last_dim_size == 0 || num_elements == 0) {
      std::shared_ptr<Tensor> grad_input_propagated = Tensor::create();
      reduce_gradient(grad_input_intermediate, grad_input_propagated,
                      parent->get_shape());
      parent->backward(grad_input_propagated);
      return;
    }

    size_t outer_dims_elements = num_elements / last_dim_size;

    size_t num_threads =
        ConfigParser::getInstance().getValue<int>("num_threads");
    if (outer_dims_elements < num_threads)
      num_threads = outer_dims_elements;
    size_t slices_per_thread = outer_dims_elements / num_threads;

    std::vector<std::function<void()>> tasks;

    for (size_t t = 0; t < num_threads; ++t) {
      size_t start_slice_idx = t * slices_per_thread;
      size_t end_slice_idx = (t == num_threads - 1)
                                 ? outer_dims_elements
                                 : start_slice_idx + slices_per_thread;

      tasks.emplace_back([&, start_slice_idx, end_slice_idx, last_dim_size]() {
        for (size_t i = start_slice_idx; i < end_slice_idx; ++i) {
          size_t start_idx = i * last_dim_size;
          float sum_of_grads = 0.0f;

          for (size_t j = 0; j < last_dim_size; ++j) {
            sum_of_grads += grad_output_data[start_idx + j];
          }

          for (size_t j = 0; j < last_dim_size; ++j) {
            grad_input_intermediate_data[start_idx + j] =
                grad_output_data[start_idx + j] -
                std::exp(output_data[start_idx + j]) * sum_of_grads;
          }
        }
      });
    }

    getThreadPool().run_batch(std::move(tasks));

    std::shared_ptr<Tensor> grad_input_propagated = Tensor::create();
    reduce_gradient(grad_input_intermediate, grad_input_propagated,
                    parent->get_shape());
    parent->backward(grad_input_propagated);
  } else {
    std::cerr << "Error: LogSoftmax operation expected 1 parent, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_nllloss(const std::shared_ptr<Tensor> &grad_output) {
  if (parents_.size() == 2) {
    std::shared_ptr<Tensor> log_probs =
        parents_[0]; // Input to NLLLoss (log probabilities)
    std::shared_ptr<Tensor> targets =
        parents_[1]; // Targets (used for indexing in forward, not for gradient)

    std::shared_ptr<Tensor> grad_input_intermediate =
        Tensor::create(log_probs->get_shape());
    const std::vector<float> &target_data = parents_[1]->get_data();
    const std::vector<float> &grad_output_data = grad_output->get_data();
    std::vector<float> &grad_input_intermediate_data = grad_input_intermediate->data_ref();

    if (grad_output->num_elements() != 1) {
      throw std::runtime_error("Gradient for NLLLoss must be a scalar.");
    }
    float loss_grad_value = grad_output_data[0];

    const std::vector<int> &log_prob_shape = log_probs->get_shape();
    size_t last_dim_size = log_prob_shape.empty() ? 0 : log_prob_shape.back();
    size_t num_elements = log_probs->num_elements();

    if (last_dim_size == 0 || num_elements == 0) {
      std::shared_ptr<Tensor> grad_input_propagated = Tensor::create();
      reduce_gradient(grad_input_intermediate, grad_input_propagated,
                      log_probs->get_shape());
      log_probs->backward(grad_input_propagated);
      return;
    }

    size_t outer_dims_elements = num_elements / last_dim_size;

    if (target_data.size() != outer_dims_elements) {
      throw std::runtime_error(
          "Target data size mismatch with log probabilities outer dimensions "
          "in NLLLoss backward.");
    }

    for (size_t i = 0; i < outer_dims_elements; ++i) {
      size_t log_prob_start_idx = i * last_dim_size;
      int target_class = static_cast<int>(target_data[i]);

      if (target_class < 0 || target_class >= static_cast<int>(last_dim_size)) {
        throw std::runtime_error(
            "Target class index out of bounds in NLLLoss backward.");
      }

      // The gradient is -1 at the target class index and 0 otherwise, scaled by
      // the loss gradient and divided by the number of instances (for the mean
      // loss)
      for (size_t j = 0; j < last_dim_size; ++j) {
        if (static_cast<int>(j) == target_class) {
          grad_input_intermediate_data[log_prob_start_idx + j] =
              -loss_grad_value / outer_dims_elements;
        } else {
          grad_input_intermediate_data[log_prob_start_idx + j] = 0.0f;
        }
      }
    }
    std::shared_ptr<Tensor> grad_input_propagated = Tensor::create();
    reduce_gradient(grad_input_intermediate, grad_input_propagated,
                    log_probs->get_shape());
    log_probs->backward(grad_input_propagated);
  } else {
    std::cerr << "Error: NegativeLogLikelihood operation expected 2 parents, "
                 "but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_layernorm(const std::shared_ptr<Tensor> &grad_output) {
  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> input_parent = parents_[0];
    const std::vector<float> &grad_output_data = grad_output->get_data();

    std::shared_ptr<Tensor> gamma = layernorm_gamma_;
    std::shared_ptr<Tensor> beta = layernorm_beta_;
    std::shared_ptr<Tensor> mean = layernorm_mean_;
    std::shared_ptr<Tensor> inv_stddev = layernorm_inv_stddev_;
    std::shared_ptr<Tensor> centered_input = layernorm_centered_input_;
    float epsilon = layernorm_epsilon_;

    const std::vector<float> &gamma_data = gamma->get_data();
    const std::vector<float> &inv_stddev_data = inv_stddev->get_data();
    const std::vector<float> &centered_input_data = centered_input->get_data();

    const std::vector<int> &input_shape = input_parent->get_shape();
    size_t last_dim_size = input_shape.empty() ? 0 : input_shape.back();
    size_t num_elements = input_parent->num_elements();
    size_t outer_dims_elements = num_elements / last_dim_size;

    if (last_dim_size == 0 || num_elements == 0) {
      std::shared_ptr<Tensor> grad_input_propagated =
          Tensor::create(input_parent->get_shape());
      input_parent->backward(grad_input_propagated);
      return;
    }

    if (gamma->grad_ && beta->grad_) {
      std::vector<float> &gamma_grad_data = gamma->grad_ref();
      std::vector<float> &beta_grad_data = beta->grad_ref();

      if (gamma_grad_data.size() != last_dim_size ||
          beta_grad_data.size() != last_dim_size) {
        throw std::runtime_error(
            "Gamma or Beta gradient size mismatch in LayerNorm backward.");
      }

      size_t num_threads =
          ConfigParser::getInstance().getValue<int>("num_threads");
      if (outer_dims_elements < num_threads)
        num_threads = outer_dims_elements;
      size_t slices_per_thread = outer_dims_elements / num_threads;

      std::vector<std::vector<float>> thread_gamma_grads(
          num_threads, std::vector<float>(last_dim_size, 0.0f));
      std::vector<std::vector<float>> thread_beta_grads(
          num_threads, std::vector<float>(last_dim_size, 0.0f));
      std::vector<std::function<void()>> tasks_gamma_beta;

      for (size_t t = 0; t < num_threads; ++t) {
        size_t start_slice_idx = t * slices_per_thread;
        size_t end_slice_idx = (t == num_threads - 1)
                                   ? outer_dims_elements
                                   : start_slice_idx + slices_per_thread;

        tasks_gamma_beta.emplace_back(
            [&, t, start_slice_idx, end_slice_idx, last_dim_size]() {
              for (size_t i = start_slice_idx; i < end_slice_idx; ++i) {
                size_t start_idx = i * last_dim_size;
                for (size_t j = 0; j < last_dim_size; ++j) {
                  size_t idx = start_idx + j;
                  thread_gamma_grads[t][j] += grad_output_data[idx] *
                                              centered_input_data[idx] *
                                              inv_stddev_data[i];
                  thread_beta_grads[t][j] += grad_output_data[idx];
                }
              }
            });
      }

      getThreadPool().run_batch(std::move(tasks_gamma_beta));

      for (size_t t = 0; t < num_threads; ++t) {
        for (size_t j = 0; j < last_dim_size; ++j) {
          gamma_grad_data[j] += thread_gamma_grads[t][j];
          beta_grad_data[j] += thread_beta_grads[t][j];
        }
      }
    }

    std::shared_ptr<Tensor> grad_input_intermediate =
        Tensor::create(input_shape);
    std::vector<float> &grad_input_intermediate_data = grad_input_intermediate->data_ref();

    const std::vector<float> &mean_data = mean->get_data();
    const std::vector<float> &variance_data = mean->get_data();

    size_t num_threads_input = std::thread::hardware_concurrency();
    if (outer_dims_elements < num_threads_input)
      num_threads_input = outer_dims_elements;
    size_t slices_per_thread_input = outer_dims_elements / num_threads_input;

    std::vector<std::function<void()>> tasks_input;

    for (size_t t = 0; t < num_threads_input; ++t) {
      size_t start_slice_idx = t * slices_per_thread_input;
      size_t end_slice_idx = (t == num_threads_input - 1)
                                 ? outer_dims_elements
                                 : start_slice_idx + slices_per_thread_input;

      tasks_input.emplace_back(
          [&, start_slice_idx, end_slice_idx, last_dim_size, epsilon]() {
            for (size_t i = start_slice_idx; i < end_slice_idx; ++i) {
              size_t start_idx = i * last_dim_size;
              float current_mean = mean_data[i];
              float current_inv_stddev = inv_stddev_data[i];
              float current_variance =
                  (1.0f / (current_inv_stddev * current_inv_stddev)) - epsilon;

              float sum_term2 = 0.0f;
              for (size_t k = 0; k < last_dim_size; ++k) {
                sum_term2 += grad_output_data[start_idx + k] *
                             centered_input_data[start_idx + k];
              }

              float sum_grad_out = 0.0f;
              for (size_t k = 0; k < last_dim_size; ++k) {
                sum_grad_out += grad_output_data[start_idx + k];
              }

              for (size_t j = 0; j < last_dim_size; ++j) {
                float grad_out = grad_output_data[start_idx + j];
                float x_mu = centered_input_data[start_idx + j];

                float term1 = gamma_data[j] * current_inv_stddev;
                float term2 = gamma_data[j] * x_mu *
                              std::pow(current_inv_stddev, 3) / last_dim_size;
                term2 *= sum_term2;
                float term3 = sum_grad_out / last_dim_size;

                grad_input_intermediate_data[start_idx + j] =
                    grad_out * term1 - term2 - term3 * term1;
              }
            }
          });
    }

    getThreadPool().run_batch(std::move(tasks_input));

    std::shared_ptr<Tensor> grad_input_propagated = Tensor::create();
    reduce_gradient(grad_input_intermediate, grad_input_propagated,
                    input_parent->get_shape());
    input_parent->backward(grad_input_propagated);
  } else {
    std::cerr
        << "Error: LayerNorm operation expected 1 input parent, but found "
        << parents_.size() << std::endl;
  }
}

void Tensor::backward_softmax(const std::shared_ptr<Tensor> &grad_output) {
  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> parent = parents_[0];
    std::shared_ptr<Tensor> grad_input_intermediate =
        Tensor::create(this->get_shape());
    const std::vector<float> &output_data = this->get_data();
    const std::vector<float> &grad_output_data = grad_output->get_data();
    std::vector<float> &grad_input_intermediate_data = grad_input_intermediate->data_ref();

    const std::vector<int> &shape = this->get_shape();
    int dim = -1; // Need to figure out the dimension softmax was applied on.
                  // Assuming last dimension for now.
    if (!shape.empty()) {
      dim = shape.size() - 1;
    } else {
      // Handle scalar or empty tensor case if necessary.
      std::shared_ptr<Tensor> grad_input_propagated = Tensor::create();
      reduce_gradient(grad_input_intermediate, grad_input_propagated,
                      parent->get_shape());
      parent->backward(grad_input_propagated);
      return;
    }

    size_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
      outer_size *= shape[i];
    }
    size_t inner_size = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
      inner_size *= shape[i];
    }
    size_t dim_size = shape[dim];

    size_t num_slices_to_process = outer_size * inner_size;
    if (num_slices_to_process == 0) {
      std::shared_ptr<Tensor> grad_input_propagated = Tensor::create();
      reduce_gradient(grad_input_intermediate, grad_input_propagated,
                      parent->get_shape());
      parent->backward(grad_input_propagated);
      return;
    }

    size_t num_threads =
        ConfigParser::getInstance().getValue<int>("num_threads");
    if (num_slices_to_process < num_threads)
      num_threads = num_slices_to_process;
    size_t slices_per_thread = num_slices_to_process / num_threads;

    std::vector<std::function<void()>> tasks;

    for (size_t t = 0; t < num_threads; ++t) {
      size_t start_slice_work_idx = t * slices_per_thread;
      size_t end_slice_work_idx =
          (t == num_threads - 1) ? num_slices_to_process
                                 : start_slice_work_idx + slices_per_thread;

      tasks.emplace_back([&, start_slice_work_idx, end_slice_work_idx,
                          inner_size, dim_size]() {
        for (size_t work_idx = start_slice_work_idx;
             work_idx < end_slice_work_idx; ++work_idx) {
          size_t i = work_idx / inner_size; // outer index
          size_t k = work_idx % inner_size; // inner index

          size_t start_idx = i * dim_size * inner_size + k;

          // Compute the Jacobian product for this slice along the softmax
          // dimension
          for (size_t j = 0; j < dim_size; ++j) {
            size_t current_idx = start_idx + j * inner_size;
            float grad_out_val = grad_output_data[current_idx];
            float softmax_out_val = output_data[current_idx];

            float sum_term = 0.0f;
            for (size_t l = 0; l < dim_size; ++l) {
              sum_term += grad_output_data[start_idx + l * inner_size] *
                          output_data[start_idx + l * inner_size];
            }
            grad_input_intermediate_data[current_idx] =
                softmax_out_val * (grad_out_val - sum_term);
          }
        }
      });
    }

    getThreadPool().run_batch(std::move(tasks));

    std::shared_ptr<Tensor> grad_input_propagated = Tensor::create();
    reduce_gradient(grad_input_intermediate, grad_input_propagated,
                    parent->get_shape());
    parent->backward(grad_input_propagated);
  } else {
    std::cerr << "Error: Softmax operation expected 1 parent, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_dropout(const std::shared_ptr<Tensor> &grad_output) {
  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> parent = parents_[0];
    std::shared_ptr<Tensor> grad_input_intermediate =
        Tensor::create(this->get_shape());
    const std::vector<float> &grad_output_data = grad_output->get_data();
    std::vector<float> &grad_input_intermediate_data = grad_input_intermediate->data_ref();

    // Retrieve the mask and scale factor stored during the forward pass
    std::shared_ptr<Tensor> mask = this->dropout_mask_;
    float scale = this->dropout_scale_;

    if (!mask || mask->get_shape() != this->get_shape()) {
      throw std::runtime_error(
          "Dropout mask is missing or shape mismatch in backward.");
    }

    const std::vector<float> &mask_data = mask->get_data();

    // The gradient is passed through only for the elements that were kept in
    // the forward pass.
    for (size_t i = 0; i < grad_output_data.size(); ++i) {
      grad_input_intermediate_data[i] =
          grad_output_data[i] * mask_data[i] * scale;
    }

    // Propagate gradient to the parent
    std::shared_ptr<Tensor> grad_input_propagated = Tensor::create();
    reduce_gradient(grad_input_intermediate, grad_input_propagated,
                    parent->get_shape());
    parent->backward(grad_input_propagated);
  } else {
    std::cerr << "Error: Dropout operation expected 1 parent, but found "
              << parents_.size() << std::endl;
  }
}

void Tensor::backward_embedding_lookup(
    const std::shared_ptr<Tensor> &grad_output) {
  if (parents_.size() == 1) {
    std::shared_ptr<Tensor> weights_parent = parents_[0];
    const std::vector<float> &grad_output_data = grad_output->get_data();

    std::shared_ptr<Tensor> input_ids = this->embedding_indices_;

    if (!input_ids || input_ids->get_shape().size() != 2) {
      throw std::runtime_error("Embedding indices tensor is missing or has "
                               "incorrect shape in backward.");
    }

    const std::vector<float> &input_ids_data = input_ids->get_data();
    const std::vector<int> &input_ids_shape = input_ids->get_shape();

    size_t batch_size = input_ids_shape[0];
    size_t sequence_length = input_ids_shape[1];
    size_t embed_dim = grad_output->get_shape().back();

    if (grad_output->get_shape() != std::vector<int>{(int)batch_size,
                                                     (int)sequence_length,
                                                     (int)embed_dim}) {
      throw std::runtime_error(
          "Gradient output shape mismatch in EmbeddingLookup backward.");
    }

    if (weights_parent->grad_) {
      std::vector<float> &weights_grad_data = weights_parent->grad_ref();
      size_t vocab_size = weights_parent->get_shape()[0];

      size_t total_indices = batch_size * sequence_length;
      size_t num_threads =
          ConfigParser::getInstance().getValue<int>("num_threads");
      if (total_indices < num_threads)
        num_threads = total_indices;
      size_t indices_per_thread = total_indices / num_threads;

      std::vector<std::vector<float>> thread_local_grads(
          num_threads, std::vector<float>(vocab_size * embed_dim, 0.0f));
      std::vector<std::function<void()>> tasks;

      for (size_t t = 0; t < num_threads; ++t) {
        size_t start_idx = t * indices_per_thread;
        size_t end_idx = (t == num_threads - 1)
                             ? total_indices
                             : start_idx + indices_per_thread;

        tasks.emplace_back([&, t, start_idx, end_idx, vocab_size, embed_dim]() {
          for (size_t i = start_idx; i < end_idx; ++i) {
            float token_id_float = input_ids_data[i];
            if (token_id_float < 0 || token_id_float >= vocab_size ||
                std::fmod(token_id_float, 1.0f) != 0.0f) {
              continue;
            }
            int token_id = static_cast<int>(token_id_float);

            size_t grad_output_start_idx = i * embed_dim;
            size_t weights_grad_start_idx = token_id * embed_dim;

            for (size_t j = 0; j < embed_dim; ++j) {
              thread_local_grads[t][weights_grad_start_idx + j] +=
                  grad_output_data[grad_output_start_idx + j];
            }
          }
        });
      }

      getThreadPool().run_batch(std::move(tasks));

      for (size_t t = 0; t < num_threads; ++t) {
        for (size_t i = 0; i < vocab_size * embed_dim; ++i) {
          weights_grad_data[i] += thread_local_grads[t][i];
        }
      }
    }
  } else {
    std::cerr << "Error: EmbeddingLookup operation expected 1 parent "
                 "(weights), but found "
              << parents_.size() << std::endl;
  }
}
