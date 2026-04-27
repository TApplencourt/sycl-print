SHELL    := /bin/bash
CXXFLAGS := -std=c++20 -Wall -Werror

ifdef USE_ACPP
  export PATH := $(HOME)/projet/p26.02/install/bin:$(PATH)
  CXX             := acpp
  SYCLFLAGS       := --acpp-targets=generic
  OPT_LEVELS      := O0 O2
else
  CXX             := icpx
  SYCLFLAGS       := -fsycl
  OPT_LEVELS      := O0 O1 O2 O3
  BUFFER_PATH     :=
  # Work around DPC++ bug (string literal through pointer segfaults at O0)
  WA_O0           := -DFMT_SYCL_WA_STR
endif

# ── Source files ────────────────────────────────────────────
TEST_DIR    := test
TEST_NAMES  := integers floats strings layout misc
ifdef USE_ACPP
  TEST_NAMES += buffer_path escape_percent
endif
TEST_HDRS   := $(wildcard $(TEST_DIR)/*.hpp $(TEST_DIR)/*.inc)

# Derived binary names (all in build/)
TEST_BINS := $(foreach t,$(TEST_NAMES),$(foreach o,$(OPT_LEVELS),build/test_$(t)_$(o)))
FUZZ_BINS := $(addprefix build/fuzz_,$(OPT_LEVELS))
FUZZ_FM   := $(addprefix build/fuzz_ffast_,$(OPT_LEVELS))
ifdef USE_ACPP
  FUZZ_PCT := $(addprefix build/fuzz_escape_percent_,$(OPT_LEVELS))
endif

ALL_BINS := $(TEST_BINS) $(FUZZ_BINS) $(FUZZ_FM) $(FUZZ_PCT)

.PHONY: all build test test-format test-fuzz test-fuzz-pct test-ffast \
        readme-examples clean

all: test

# ── Build directory ─────────────────────────────────────────

build/:
	mkdir -p build

# ── Test binaries (one binary per test × opt level) ─────────

define TEST_template
build/test_$(1)_$(2): $(TEST_DIR)/test_$(1).cpp $(TEST_HDRS) sycl_khr_print.hpp | build/
	@TIMEFORMAT="  compile test_$(1)_$(2): %Rs"; time \
	$$(CXX) $$(CXXFLAGS) $$(SYCLFLAGS) -$(2) $$(BUFFER_PATH) $$(WA_$(2)) $$< -o $$@
endef

$(foreach t,$(TEST_NAMES),$(foreach o,$(OPT_LEVELS),$(eval $(call TEST_template,$(t),$(o)))))

# ── Fuzz targets (single binary per opt level) ──────────────

build/fuzz_%: $(TEST_DIR)/fuzz.cpp $(TEST_DIR)/capture.hpp sycl_khr_print.hpp | build/
	@TIMEFORMAT="  compile fuzz_$*: %Rs"; time \
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -$* $(BUFFER_PATH) $(WA_$*) $< -o $@

build/fuzz_ffast_%: $(TEST_DIR)/fuzz.cpp $(TEST_DIR)/capture.hpp sycl_khr_print.hpp | build/
	@TIMEFORMAT="  compile fuzz_ffast_$*: %Rs"; time \
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -$* -ffast-math $(BUFFER_PATH) $(WA_$*) $< -o $@

build/fuzz_escape_percent_%: $(TEST_DIR)/fuzz_escape_percent.cpp $(TEST_DIR)/capture.hpp sycl_khr_print.hpp | build/
	@TIMEFORMAT="  compile fuzz_escape_percent_$*: %Rs"; time \
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -$* $(WA_$*) $< -o $@

# README examples
build/example_readme%: example_readme%.cpp sycl_khr_print.hpp | build/
	@TIMEFORMAT="  compile example_readme$*: %Rs"; time \
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $< -o $@

readme-examples: build/example_readme1 build/example_readme2
	@t0=$$(date +%s%N); \
	./build/example_readme1 >/dev/null && ./build/example_readme2 >/dev/null; rc=$$?; \
	ms=$$(( ($$(date +%s%N) - t0) / 1000000 )); \
	if [ $$rc -eq 0 ]; then echo "readme-examples: PASS ($${ms}ms)"; \
	else echo "readme-examples: FAIL ($${ms}ms)"; false; fi

# ── Test targets ─────────────────────────────────────────────

build: $(ALL_BINS)

test: test-format test-fuzz test-ffast readme-examples
	@echo "==============================="
	@echo "All tests passed."
	@echo "==============================="

test-format: $(TEST_BINS)
	@fail=0; \
	for t in $(TEST_NAMES); do \
	  for opt in $(OPT_LEVELS); do \
	    t0=$$(date +%s%N); \
	    ./build/test_$${t}_$$opt; rc=$$?; \
	    ms=$$(( ($$(date +%s%N) - t0) / 1000000 )); \
	    if [ $$rc -eq 0 ]; then echo "test_$$t -$$opt: PASS ($${ms}ms)"; \
	    else echo "test_$$t -$$opt: FAIL ($${ms}ms)"; fail=1; fi; \
	  done; \
	done; \
	exit $$fail

test-fuzz: $(FUZZ_BINS)
	@fail=0; \
	for opt in $(OPT_LEVELS); do \
	  t0=$$(date +%s%N); \
	  ./build/fuzz_$$opt; rc=$$?; \
	  ms=$$(( ($$(date +%s%N) - t0) / 1000000 )); \
	  if [ $$rc -eq 0 ]; then echo "fuzz -$$opt: PASS ($${ms}ms)"; \
	  else echo "fuzz -$$opt: FAIL ($${ms}ms)"; fail=1; fi; \
	done; \
	exit $$fail

ifdef USE_ACPP
test-fuzz-pct: $(FUZZ_PCT)
	@fail=0; \
	for opt in $(OPT_LEVELS); do \
	  t0=$$(date +%s%N); \
	  ./build/fuzz_escape_percent_$$opt; rc=$$?; \
	  ms=$$(( ($$(date +%s%N) - t0) / 1000000 )); \
	  if [ $$rc -eq 0 ]; then echo "fuzz_escape_percent -$$opt: PASS ($${ms}ms)"; \
	  else echo "fuzz_escape_percent -$$opt: FAIL ($${ms}ms)"; fail=1; fi; \
	done; \
	exit $$fail
endif

test-ffast: $(FUZZ_FM)
	@fail=0; \
	for opt in $(OPT_LEVELS); do \
	  t0=$$(date +%s%N); \
	  ./build/fuzz_ffast_$$opt; rc=$$?; \
	  ms=$$(( ($$(date +%s%N) - t0) / 1000000 )); \
	  if [ $$rc -eq 0 ]; then echo "fuzz -ffast-math -$$opt: PASS ($${ms}ms)"; \
	  else echo "fuzz -ffast-math -$$opt: FAIL ($${ms}ms)"; fail=1; fi; \
	done; \
	exit $$fail

clean:
	rm -rf build/
