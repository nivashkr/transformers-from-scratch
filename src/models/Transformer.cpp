#include "models/Transformer.h"
#include "utils/Helpers.h"
#include "utils/Tensor.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

Transformer::Transformer(int input_vocab_size, int target_vocab_size,
                         int embed_dim, int max_sequence_length, int num_layers,
                         int num_heads, int ff_hidden_dim, float dropout_rate,
                         float pad_token_id)
    : input_vocab_size_(input_vocab_size),
      target_vocab_size_(target_vocab_size), embed_dim_(embed_dim),
      max_sequence_length_(max_sequence_length), num_layers_(num_layers),
      num_heads_(num_heads), ff_hidden_dim_(ff_hidden_dim),
      dropout_rate_(dropout_rate), pad_token_id_(pad_token_id),
      encoder_embedding_(input_vocab_size, embed_dim),
      encoder_positional_encoding_(max_sequence_length, embed_dim),
      encoder_(num_layers, embed_dim, num_heads, ff_hidden_dim, dropout_rate),
      decoder_embedding_(target_vocab_size, embed_dim),
      decoder_positional_encoding_(max_sequence_length, embed_dim),
      decoder_(num_layers, embed_dim, num_heads, ff_hidden_dim, dropout_rate),
      final_linear_(embed_dim, target_vocab_size) {
  if (embed_dim_ % num_heads_ != 0) {
    throw std::runtime_error(
        "Embedding dimension must be divisible by the number of heads.");
  }
}

std::shared_ptr<Tensor> Transformer::create_encoder_padding_mask(
    const std::shared_ptr<Tensor> &encoder_input_ids) {
  return create_padding_mask(encoder_input_ids, pad_token_id_);
}

std::shared_ptr<Tensor> Transformer::create_decoder_self_attention_mask(
    const std::shared_ptr<Tensor> &decoder_input_ids) {
  const auto &decoder_shape = decoder_input_ids->get_shape();
  if (decoder_shape.size() != 2) {
    throw std::runtime_error(
        "Decoder input for self-attention mask must be 2D (batch, seq_len).");
  }
  int batch_size = decoder_shape[0];
  int sequence_length = decoder_shape[1];

  std::shared_ptr<Tensor> look_ahead_mask =
      create_look_ahead_mask(sequence_length);
  std::shared_ptr<Tensor> padding_mask =
      create_padding_mask(decoder_input_ids, pad_token_id_);

  const auto &look_ahead_shape = look_ahead_mask->get_shape();
  const auto &padding_shape = padding_mask->get_shape();

  if (look_ahead_shape !=
          std::vector<int>{1, 1, sequence_length, sequence_length} ||
      padding_shape != std::vector<int>{batch_size, 1, 1, sequence_length}) {
    throw std::runtime_error("Unexpected shapes for look-ahead or padding "
                             "masks during combination.");
  }

  std::vector<int> combined_mask_shape = {batch_size, 1, sequence_length,
                                          sequence_length};
  std::shared_ptr<Tensor> combined_mask = Tensor::create(combined_mask_shape);
  std::vector<float> &combined_mask_data = combined_mask->data_ref();

  const std::vector<float> &look_ahead_data = look_ahead_mask->get_data();
  const std::vector<float> &padding_data = padding_mask->get_data();

  for (int b = 0; b < batch_size; ++b) {
    for (int tq = 0; tq < sequence_length; ++tq) {
      for (int tk = 0; tk < sequence_length; ++tk) {
        size_t look_ahead_idx = tq * sequence_length + tk;
        size_t padding_idx = b * sequence_length + tk;
        size_t combined_idx =
            b * sequence_length * sequence_length + tq * sequence_length + tk;
        combined_mask_data[combined_idx] = std::max(
            look_ahead_data[look_ahead_idx], padding_data[padding_idx]);
      }
    }
  }

  return combined_mask;
}

std::shared_ptr<Tensor> Transformer::create_decoder_cross_attention_mask(
    const std::shared_ptr<Tensor> &encoder_input_ids) {
  return create_padding_mask(encoder_input_ids, pad_token_id_);
}

std::shared_ptr<Tensor>
Transformer::forward(const std::shared_ptr<Tensor> &encoder_input_ids,
                     const std::shared_ptr<Tensor> &decoder_input_ids,
                     bool is_training) {
  const std::vector<int> &enc_input_shape = encoder_input_ids->get_shape();
  const std::vector<int> &dec_input_shape = decoder_input_ids->get_shape();

  if (enc_input_shape.size() != 2 || dec_input_shape.size() != 2) {
    throw std::runtime_error("Encoder and decoder input IDs must be 2D tensors "
                             "(batch_size, sequence_length).");
  }

  std::shared_ptr<Tensor> encoder_padding_mask =
      create_encoder_padding_mask(encoder_input_ids);
  std::shared_ptr<Tensor> decoder_self_attention_mask =
      create_decoder_self_attention_mask(decoder_input_ids);
  std::shared_ptr<Tensor> decoder_cross_attention_mask =
      create_decoder_cross_attention_mask(encoder_input_ids);

  std::shared_ptr<Tensor> encoder_input_embeddings =
      encoder_embedding_.forward(encoder_input_ids);
  std::shared_ptr<Tensor> encoder_input_with_pos =
      encoder_positional_encoding_.forward(encoder_input_embeddings);

  std::shared_ptr<Tensor> encoder_input_dropped = encoder_input_with_pos;

  std::shared_ptr<Tensor> encoder_output = encoder_.forward(
      encoder_input_dropped, encoder_padding_mask, is_training);

  std::shared_ptr<Tensor> decoder_input_embeddings =
      decoder_embedding_.forward(decoder_input_ids);
  std::shared_ptr<Tensor> decoder_input_with_pos =
      decoder_positional_encoding_.forward(decoder_input_embeddings);

  std::shared_ptr<Tensor> decoder_input_dropped = decoder_input_with_pos;

  std::shared_ptr<Tensor> decoder_output = decoder_.forward(
      decoder_input_dropped, encoder_output, decoder_self_attention_mask,
      decoder_cross_attention_mask, is_training);

  std::shared_ptr<Tensor> logits = final_linear_.forward(decoder_output);

  return logits;
}

void Transformer::save_weights(const std::string &filename) const {
  std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);
  if (!outfile.is_open()) {
    throw std::runtime_error("Failed to open file for saving weights: " +
                             filename);
  }

  const auto &optimizable_tensors = Tensor::get_optimizable_tensors();
  uint32_t num_tensors = optimizable_tensors.size();

  // Write the number of tensors
  outfile.write(reinterpret_cast<const char *>(&num_tensors),
                sizeof(num_tensors));

  std::cout << "Saving " << num_tensors << " optimizable tensors to "
            << filename << "..." << std::endl;

  for (const auto &tensor_ptr : optimizable_tensors) {
    if (!tensor_ptr) {
      std::cerr
          << "Warning: Encountered null tensor pointer while saving weights."
          << std::endl;
      continue;
    }
    const std::vector<int> &shape = tensor_ptr->get_shape();
    const std::vector<float> &data = tensor_ptr->get_data();

    // Write shape information
    uint32_t rank = shape.size();
    outfile.write(reinterpret_cast<const char *>(&rank), sizeof(rank));
    for (int dim_size : shape) {
      uint32_t size = static_cast<uint32_t>(dim_size);
      outfile.write(reinterpret_cast<const char *>(&size), sizeof(size));
    }

    // Write data
    uint64_t num_elements = data.size();
    outfile.write(reinterpret_cast<const char *>(&num_elements),
                  sizeof(num_elements));
    if (num_elements > 0) {
      outfile.write(reinterpret_cast<const char *>(data.data()),
                    num_elements * sizeof(float));
    }
    if (!outfile) {
      throw std::runtime_error("Error writing tensor data to file: " +
                               filename);
    }
  }
  std::cout << "Weights saved successfully." << std::endl;
  outfile.close();
}

void Transformer::load_weights(const std::string &filename) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open file for loading weights: " +
                             filename);
  }

  uint32_t num_tensors_in_file;
  infile.read(reinterpret_cast<char *>(&num_tensors_in_file),
              sizeof(num_tensors_in_file));
  if (!infile) {
    throw std::runtime_error("Failed to read number of tensors from file: " +
                             filename);
  }

  auto &optimizable_tensors = Tensor::get_optimizable_tensors();
  if (num_tensors_in_file != optimizable_tensors.size()) {
    throw std::runtime_error("Mismatch between number of tensors in file (" +
                             std::to_string(num_tensors_in_file) +
                             ") and model (" +
                             std::to_string(optimizable_tensors.size()) +
                             "). Model architecture may have changed.");
  }

  std::cout << "Loading " << num_tensors_in_file << " optimizable tensors from "
            << filename << "..." << std::endl;

  for (size_t i = 0; i < num_tensors_in_file; ++i) {
    std::shared_ptr<Tensor> &tensor_ptr = optimizable_tensors[i];
    if (!tensor_ptr) {
      std::cerr << "Warning: Encountered null tensor pointer in model while "
                   "loading weights for index "
                << i << "." << std::endl;
      uint32_t rank;
      infile.read(reinterpret_cast<char *>(&rank), sizeof(rank));
      for (uint32_t r = 0; r < rank; ++r) {
        uint32_t dim_size;
        infile.read(reinterpret_cast<char *>(&dim_size), sizeof(dim_size));
      }
      uint64_t num_elements;
      infile.read(reinterpret_cast<char *>(&num_elements),
                  sizeof(num_elements));
      infile.seekg(num_elements * sizeof(float), std::ios::cur);
      continue;
    }

    // Read shape information from file
    uint32_t rank;
    infile.read(reinterpret_cast<char *>(&rank), sizeof(rank));
    if (!infile)
      throw std::runtime_error("Failed to read tensor rank from file.");

    std::vector<int> file_shape(rank);
    for (uint32_t r = 0; r < rank; ++r) {
      uint32_t dim_size;
      infile.read(reinterpret_cast<char *>(&dim_size), sizeof(dim_size));
      if (!infile)
        throw std::runtime_error(
            "Failed to read tensor dimension size from file.");
      file_shape[r] = static_cast<int>(dim_size);
    }

    // Read data size from file
    uint64_t file_num_elements;
    infile.read(reinterpret_cast<char *>(&file_num_elements),
                sizeof(file_num_elements));
    if (!infile)
      throw std::runtime_error(
          "Failed to read tensor element count from file.");

    // Verify shape and size match the tensor in the model
    const std::vector<int> &model_shape = tensor_ptr->get_shape();
    size_t model_num_elements = tensor_ptr->num_elements();

    if (file_shape != model_shape || file_num_elements != model_num_elements) {
      infile.close();
      throw std::runtime_error(
          "Mismatch in shape or size for tensor index " + std::to_string(i) +
          ". File shape: " + vector_to_string(file_shape) +
          ", Model shape: " + vector_to_string(model_shape) +
          ". File elements: " + std::to_string(file_num_elements) +
          ", Model elements: " + std::to_string(model_num_elements));
    }

    // Read data into a temporary buffer, then set it in the tensor
    if (model_num_elements > 0) {
      auto data_buffer =
          std::make_shared<std::vector<float>>(model_num_elements);
      infile.read(reinterpret_cast<char *>(data_buffer->data()),
                  model_num_elements * sizeof(float));
      if (!infile) {
        throw std::runtime_error(
            "Error reading tensor data from file for tensor index " +
            std::to_string(i));
      }
      tensor_ptr->set_data(data_buffer);
    } else {
      tensor_ptr->set_data(std::make_shared<std::vector<float>>());
    }
  }
  std::cout << "Weights loaded successfully." << std::endl;
  infile.close();
}