# Sindarin Tensor Package - Makefile

.PHONY: all test hooks build-libs install-libs clean help

# Disable implicit rules for .sn.c files (compiled by the Sindarin compiler)
%.sn: %.sn.c
	@:

#------------------------------------------------------------------------------
# Platform Detection
#------------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
    PLATFORM := windows
    EXE_EXT  := .exe
    MKDIR    := mkdir
else
    UNAME_S := $(shell uname -s 2>/dev/null || echo Unknown)
    ifeq ($(UNAME_S),Darwin)
        PLATFORM := darwin
    else
        PLATFORM := linux
    endif
    EXE_EXT :=
    MKDIR   := mkdir -p
endif

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
BIN_DIR := bin
SN      ?= sn

SRC_SOURCES := $(wildcard src/*.sn) $(wildcard src/*.sn.c)

TEST_SRCS := $(wildcard tests/test_*.sn)
TEST_BINS := $(patsubst tests/%.sn,$(BIN_DIR)/%$(EXE_EXT),$(TEST_SRCS))

#------------------------------------------------------------------------------
# ggml build configuration
#------------------------------------------------------------------------------
GGML_SRC   := /tmp/ggml-src
GGML_BUILD := /tmp/ggml-build
GGML_INST  := /tmp/ggml-install

# Set GGML_CUDA=ON to enable CUDA backend
GGML_CUDA ?= OFF

#------------------------------------------------------------------------------
# Targets
#------------------------------------------------------------------------------
all: test

test: hooks $(TEST_BINS)
	@echo "Running tests..."
	@failed=0; \
	for t in $(TEST_BINS); do \
	    printf "  %-50s" "$$t"; \
	    if $$t; then \
	        echo "PASS"; \
	    else \
	        echo "FAIL"; \
	        failed=1; \
	    fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
	    echo "All tests passed."; \
	else \
	    echo "Some tests failed."; \
	    exit 1; \
	fi

$(BIN_DIR):
	@$(MKDIR) $(BIN_DIR)

$(BIN_DIR)/%$(EXE_EXT): tests/%.sn $(SRC_SOURCES) | $(BIN_DIR)
	@$(SN) $< -o $@ -l 1

build-libs:
	@echo "Building ggml from source for $(PLATFORM)..."
	@if [ ! -d "$(GGML_SRC)" ]; then \
	    git clone --depth=1 https://github.com/ggml-org/ggml.git $(GGML_SRC); \
	fi
	@cmake -S $(GGML_SRC) -B $(GGML_BUILD) \
	    -G Ninja \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_INSTALL_PREFIX=$(GGML_INST) \
	    -DBUILD_SHARED_LIBS=OFF \
	    -DGGML_STATIC=ON \
	    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	    -DGGML_CUDA=$(GGML_CUDA)
	@cmake --build $(GGML_BUILD) -j
	@cmake --install $(GGML_BUILD)
	@cmake -S . -B build/package \
	    -DGGML_INSTALL_PREFIX=$(GGML_INST)
	@cmake --build build/package
	@echo "Libraries built in libs/$(PLATFORM)/"

install-libs:
	@bash scripts/install.sh

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BIN_DIR) .sn
	@echo "Clean complete."

#------------------------------------------------------------------------------
# hooks - Configure git to use tracked pre-commit hooks
#------------------------------------------------------------------------------
hooks:
	@git config core.hooksPath .githooks 2>/dev/null || true

help:
	@echo "Sindarin Tensor Package (ggml backend)"
	@echo ""
	@echo "Targets:"
	@echo "  make test              Build and run all tests"
	@echo "  make build-libs        Build ggml libraries from source"
	@echo "  make build-libs GGML_CUDA=ON   Build with CUDA support"
	@echo "  make install-libs      Download pre-built libraries from GitHub releases"
	@echo "  make clean             Remove build artifacts"
	@echo "  make help              Show this help"
	@echo ""
	@echo "Platform: $(PLATFORM)"
