#include "utils/ConfigParser.h"

template <>
std::string ConfigParser::getValue<std::string>(const std::string &key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = config_map_.find(key);
  if (it == config_map_.end()) {
    throw std::runtime_error("Config key not found: " + key);
  }
  return it->second;
}