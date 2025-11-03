#include "layers/Embedding.h"
#include <stdexcept>

Embedding::Embedding(int vocab_size, int embed_dim)
    : vocab_size_(vocab_size), embed_dim_(embed_dim) {
  // Embedding matrix shape: (vocab_size, embed_dim)
  weights_ = Tensor::create(std::vector<int>{vocab_size_, embed_dim_}, true);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  std::shared_ptr<std::vector<float>> weights_data =
      std::make_shared<std::vector<float>>(vocab_size_ * embed_dim_);
  for (size_t i = 0; i < weights_data->size(); ++i) {
    (*weights_data)[i] = dist(gen);
  }
  weights_->set_data(weights_data);
}

std::shared_ptr<Tensor>
Embedding::forward(const std::shared_ptr<Tensor> &input_ids) {
  // input_ids shape: (batch_size, sequence_length)
  // Output shape: (batch_size, sequence_length, embed_dim)

  const std::vector<int> &input_shape = input_ids->get_shape();
  if (input_shape.size() != 2) {
    throw std::runtime_error("Embedding layer input must be a 2D tensor "
                             "(batch_size, sequence_length).");
  }

  size_t batch_size = input_shape[0];
  size_t sequence_length = input_shape[1];

  std::shared_ptr<Tensor> output =
      Tensor::create({(int)batch_size, (int)sequence_length, embed_dim_});
  std::vector<float> &output_data = output->data_ref();

  const std::vector<float> &input_ids_data = input_ids->get_data();
  const std::vector<float> &weights_data = weights_->get_data();

  for (size_t i = 0; i < batch_size * sequence_length; ++i) {
    float token_id_float = input_ids_data[i];
    if (token_id_float < 0 || token_id_float >= vocab_size_ ||
        std::fmod(token_id_float, 1.0f) != 0.0f) {
      throw std::runtime_error("Input token ID out of vocabulary bounds or not "
                               "an integer in Embedding forward.");
    }
    int token_id = static_cast<int>(token_id_float);

    size_t output_start_idx = i * embed_dim_;
    size_t weights_start_idx = token_id * embed_dim_;

    for (size_t j = 0; j < embed_dim_; ++j) {
      output_data[output_start_idx + j] = weights_data[weights_start_idx + j];
    }
  }

  output->set_creator_op(OperationType::EmbeddingLookup);
  output->set_parents({weights_});
  output->embedding_indices_ = input_ids;

  return output;
}

std::shared_ptr<Tensor> Embedding::get_weights() { return weights_; }
