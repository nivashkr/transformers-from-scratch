#ifndef CONFIGPARSER_H
#define CONFIGPARSER_H

#include <algorithm>
#include <cctype>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

class ConfigParser {
public:
  static ConfigParser &getInstance(const std::string &filename = "") {
    static ConfigParser instance(filename);
    return instance;
  }

  // Delete copy constructor and assignment operator to prevent cloning
  ConfigParser(const ConfigParser &) = delete;
  ConfigParser &operator=(const ConfigParser &) = delete;

  // Method to load the configuration
  void loadConfig(const std::string &filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_map_.clear();

    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open config file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
      // Remove leading/trailing whitespace
      line.erase(line.begin(),
                 std::find_if(line.begin(), line.end(), [](unsigned char ch) {
                   return !std::isspace(ch);
                 }));
      line.erase(
          std::find_if(line.rbegin(), line.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          line.end());

      // Skip empty lines and comments
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::size_t equals_pos = line.find('=');
      if (equals_pos != std::string::npos) {
        std::string key = line.substr(0, equals_pos);
        std::string value = line.substr(equals_pos + 1);

        // Trim whitespace from key and value
        key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
        value.erase(value.begin(), std::find_if(value.begin(), value.end(),
                                                [](unsigned char ch) {
                                                  return !std::isspace(ch);
                                                }));
        value.erase(
            std::find_if(value.rbegin(), value.rend(),
                         [](unsigned char ch) { return !std::isspace(ch); })
                .base(),
            value.end());

        config_map_[key] = value;
      }
    }
  }

  template <typename T> T getValue(const std::string &key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = config_map_.find(key);
    if (it == config_map_.end()) {
      throw std::runtime_error("Config key not found: " + key);
    }

    if constexpr (std::is_same_v<T, bool>) {
      std::string value = it->second;
      std::transform(value.begin(), value.end(), value.begin(), ::tolower);
      if (value == "true")
        return true;
      if (value == "false")
        return false;
      throw std::runtime_error("Failed to parse boolean value for key: " + key);
    } else {
      std::stringstream ss(it->second);
      T value;
      ss >> value;
      if (ss.fail() || !ss.eof()) {
        throw std::runtime_error("Failed to parse config value for key: " +
                                 key);
      }
      return value;
    }
  }

private:
  ConfigParser(const std::string &filename = "") {
    if (!filename.empty()) {
      loadConfig(filename);
    }
  }

  std::unordered_map<std::string, std::string> config_map_;
  mutable std::mutex mutex_;
};

template <>
std::string ConfigParser::getValue<std::string>(const std::string &key) const;

#endif // CONFIGPARSER_H
