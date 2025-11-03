#ifndef TRANSFORMER_CPP_HELPER_H
#define TRANSFORMER_CPP_HELPER_H

#include "Tensor.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

inline void print_tensor(const std::shared_ptr<Tensor> &t,
                         const std::string &name) {
  if (!t)
    return;
  std::cout << "--- Tensor: " << name << " ---" << std::endl;
  std::cout << "Shape: [";
  for (size_t i = 0; i < t->get_shape().size(); ++i) {
    std::cout << t->get_shape()[i]
              << (i == t->get_shape().size() - 1 ? "" : ", ");
  }
  std::cout << "]" << std::endl;

  std::cout << "Data: [";
  const auto &data = t->get_data();
  size_t print_limit = std::min((size_t)20, data.size());
  for (size_t i = 0; i < print_limit; ++i) {
    std::cout << data[i] << (i == print_limit - 1 ? "" : ", ");
  }
  if (data.size() > print_limit) {
    std::cout << ", ...";
  }
  std::cout << "]" << std::endl;

  std::cout << "Grad: [";
  const auto &grad = t->get_grad();
  print_limit = std::min((size_t)20, grad.size());
  for (size_t i = 0; i < print_limit; ++i) {
    std::cout << grad[i] << (i == print_limit - 1 ? "" : ", ");
  }
  if (grad.size() > print_limit) {
    std::cout << ", ...";
  }
  std::cout << "]" << std::endl;
  std::cout << "--------------------------" << std::endl;
}

inline bool are_tensors_equal(const std::shared_ptr<Tensor> &t1,
                              const std::shared_ptr<Tensor> &t2,
                              float tolerance = 1e-9) {
  if (!t1 || !t2)
    return false;
  if (t1->get_shape() != t2->get_shape()) {
    std::cerr << "Shape mismatch!" << std::endl;
    return false;
  }
  const auto &data1 = t1->get_data();
  const auto &data2 = t2->get_data();
  if (data1.size() != data2.size()) {
    std::cerr << "Data size mismatch!" << std::endl;
    return false;
  }

  for (size_t i = 0; i < data1.size(); ++i) {
    if (std::abs(data1[i] - data2[i]) > tolerance) {
      std::cerr << "Data mismatch at index " << i << ": " << data1[i] << " vs "
                << data2[i] << std::endl;
      return false;
    }
  }
  return true;
}

template <typename T>
inline std::string vector_to_string(const std::vector<T> &vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i != vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

#endif // TRANSFORMER_CPP_HELPER_H
