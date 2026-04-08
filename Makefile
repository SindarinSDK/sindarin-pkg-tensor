# Sindarin Tensor Package - Makefile

.PHONY: all test hooks install-libs clean help

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

SRC_SOURCES := $(wildcard src/*.sn) $(wildcard src/*.sn.c) $(wildcard src/native/*.sn.c) $(wildcard src/native/*.h)

TEST_SRCS := $(wildcard tests/test_*.sn)
TEST_BINS := $(patsubst tests/%.sn,$(BIN_DIR)/%$(EXE_EXT),$(TEST_SRCS))

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

GGML_INCLUDE := $(wildcard libs/$(PLATFORM)/include)
GGML_LIB     := $(wildcard libs/$(PLATFORM)/lib)

$(BIN_DIR)/%$(EXE_EXT): tests/%.sn $(SRC_SOURCES) | $(BIN_DIR)
	@SN_CFLAGS="-I$(CURDIR)/libs/$(PLATFORM)/include $(SN_CFLAGS)" \
	 SN_LDFLAGS="-L$(CURDIR)/libs/$(PLATFORM)/lib $(SN_LDFLAGS)" \
	 $(SN) $< -o $@ -l 1

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
	@echo "  make install-libs      Download pre-built libraries from GitHub releases"
	@echo "  make clean             Remove build artifacts"
	@echo "  make help              Show this help"
	@echo ""
	@echo "Platform: $(PLATFORM)"
