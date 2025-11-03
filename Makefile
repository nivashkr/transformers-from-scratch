# Makefile for TransformerCPP
# Builds object files in ./build directory and executables in main directory

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -flto -DNDEBUG -I./include
LDFLAGS = -flto

# Directories
BUILD_DIR = ./build
SRC_DIR = ./src
TEST_DIR = ./tests
INCLUDE_DIR = ./include

# Create build subdirectories
BUILD_LAYERS_DIR = $(BUILD_DIR)/layers
BUILD_MODELS_DIR = $(BUILD_DIR)/models
BUILD_UTILS_DIR = $(BUILD_DIR)/utils

# Source files
MAIN_SRC = $(SRC_DIR)/main.cpp

LAYERS_SRCS = $(SRC_DIR)/layers/Activations.cpp \
              $(SRC_DIR)/layers/Dropout.cpp \
              $(SRC_DIR)/layers/Embedding.cpp \
              $(SRC_DIR)/layers/LayerNorm.cpp \
              $(SRC_DIR)/layers/Linear.cpp \
              $(SRC_DIR)/layers/MultiHeadAttention.cpp \
              $(SRC_DIR)/layers/PositionalEncoding.cpp \
              $(SRC_DIR)/layers/PositionwiseFeedForward.cpp

MODELS_SRCS = $(SRC_DIR)/models/Decoder.cpp \
              $(SRC_DIR)/models/DecoderLayer.cpp \
              $(SRC_DIR)/models/Encoder.cpp \
              $(SRC_DIR)/models/EncoderLayer.cpp \
              $(SRC_DIR)/models/Transformer.cpp

UTILS_SRCS = $(SRC_DIR)/utils/ConfigParser.cpp \
             $(SRC_DIR)/utils/DataLoader.cpp \
             $(SRC_DIR)/utils/LossFunction.cpp \
             $(SRC_DIR)/utils/Masking.cpp \
             $(SRC_DIR)/utils/Optimizer.cpp \
             $(SRC_DIR)/utils/Tensor.cpp \
             $(SRC_DIR)/utils/ThreadPool.cpp

TEST_SRCS = $(TEST_DIR)/test_tensor.cpp

# Object files (will be created in build directory)
MAIN_OBJ = $(BUILD_DIR)/main.o

LAYERS_OBJS = $(BUILD_LAYERS_DIR)/Activations.o \
              $(BUILD_LAYERS_DIR)/Dropout.o \
              $(BUILD_LAYERS_DIR)/Embedding.o \
              $(BUILD_LAYERS_DIR)/LayerNorm.o \
              $(BUILD_LAYERS_DIR)/Linear.o \
              $(BUILD_LAYERS_DIR)/MultiHeadAttention.o \
              $(BUILD_LAYERS_DIR)/PositionalEncoding.o \
              $(BUILD_LAYERS_DIR)/PositionwiseFeedForward.o

MODELS_OBJS = $(BUILD_MODELS_DIR)/Decoder.o \
              $(BUILD_MODELS_DIR)/DecoderLayer.o \
              $(BUILD_MODELS_DIR)/Encoder.o \
              $(BUILD_MODELS_DIR)/EncoderLayer.o \
              $(BUILD_MODELS_DIR)/Transformer.o

UTILS_OBJS = $(BUILD_UTILS_DIR)/ConfigParser.o \
             $(BUILD_UTILS_DIR)/DataLoader.o \
             $(BUILD_UTILS_DIR)/LossFunction.o \
             $(BUILD_UTILS_DIR)/Masking.o \
             $(BUILD_UTILS_DIR)/Optimizer.o \
             $(BUILD_UTILS_DIR)/Tensor.o \
             $(BUILD_UTILS_DIR)/ThreadPool.o

TEST_OBJS = $(BUILD_DIR)/test_tensor.o

# All object files
ALL_OBJS = $(MAIN_OBJ) $(LAYERS_OBJS) $(MODELS_OBJS) $(UTILS_OBJS)
TEST_OBJS_ALL = $(TEST_OBJS) $(UTILS_OBJS) $(BUILD_LAYERS_DIR)/Linear.o

# Executables (in main directory)
MAIN_EXE = neural_network
TEST_EXE = test_tensor

# Default target
all: $(MAIN_EXE) $(TEST_EXE)

# Create build directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_LAYERS_DIR)
	mkdir -p $(BUILD_MODELS_DIR)
	mkdir -p $(BUILD_UTILS_DIR)

# Main executable
$(MAIN_EXE): $(ALL_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ -lstdc++fs

# Test executable
$(TEST_EXE): $(TEST_OBJS_ALL)
	$(CXX) $(LDFLAGS) -o $@ $^ -lstdc++fs

# Main object file
$(BUILD_DIR)/main.o: $(MAIN_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Test object file
$(BUILD_DIR)/test_tensor.o: $(TEST_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Layer object files
$(BUILD_LAYERS_DIR)/%.o: $(SRC_DIR)/layers/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Model object files
$(BUILD_MODELS_DIR)/%.o: $(SRC_DIR)/models/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Utility object files
$(BUILD_UTILS_DIR)/%.o: $(SRC_DIR)/utils/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(MAIN_EXE) $(TEST_EXE)

# Install target (optional)
install: $(MAIN_EXE) $(TEST_EXE)
	@echo "Build completed successfully!"
	@echo "Executables created: $(MAIN_EXE), $(TEST_EXE)"

# Phony targets
.PHONY: all clean install 