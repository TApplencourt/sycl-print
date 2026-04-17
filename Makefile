SHELL    := /bin/bash
CXXFLAGS := -std=c++20 -Wall -Werror

ifdef USE_ACPP
  export PATH := $(HOME)/projet/p26.02/install/bin:$(PATH)
  CXX             := acpp
  SYCLFLAGS       := --acpp-targets=generic
  OPT_LEVELS      := O0
  BUFFER_PATH     := -DFMT_SYCL_BUFFER_PATH_ONLY
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
TEST_SRCS   := test_main test_integers test_floats test_strings test_layout test_misc
TEST_HDRS   := $(wildcard $(TEST_DIR)/*.hpp $(TEST_DIR)/*.inc)

# Derived binary names (all in build/)
TEST_BINS := $(addprefix build/test_,$(OPT_LEVELS))
FUZZ_BINS := $(addprefix build/fuzz_,$(OPT_LEVELS))
FUZZ_FM   := $(addprefix build/fuzz_ffast_,$(OPT_LEVELS))

ALL_BINS := $(TEST_BINS) $(FUZZ_BINS) $(FUZZ_FM)

.PHONY: all build test test-format test-fuzz test-ffast \
        readme-examples clean

all: test

# ── Build directory ─────────────────────────────────────────

build/:
	mkdir -p build

# ── Test binaries (single binary per opt level) ─────────────

define TEST_template
$(foreach s,$(TEST_SRCS),$(eval \
build/$(s)_$(1).o: $(TEST_DIR)/$(s).cpp $(TEST_HDRS) sycl_khr_print.hpp | build/; \
	@TIMEFORMAT="  compile $(s)_$(1).o: %Rs"; time \
	$$(CXX) $$(CXXFLAGS) $$(SYCLFLAGS) -$(1) $$(BUFFER_PATH) $$(WA_$(1)) -c $$< -o $$@))

build/test_$(1): $(foreach s,$(TEST_SRCS),build/$(s)_$(1).o)
	@TIMEFORMAT="  link test_$(1): %Rs"; time \
	$$(CXX) $$(CXXFLAGS) $$(SYCLFLAGS) $$^ -o $$@
endef

$(foreach o,$(OPT_LEVELS),$(eval $(call TEST_template,$(o))))

# ── Fuzz targets (single binary per opt level) ──────────────

build/fuzz_%: $(TEST_DIR)/fuzz.cpp $(TEST_DIR)/capture.hpp sycl_khr_print.hpp | build/
	@TIMEFORMAT="  compile fuzz_$*: %Rs"; time \
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -$* $(BUFFER_PATH) $(WA_$*) $< -o $@

build/fuzz_ffast_%: $(TEST_DIR)/fuzz.cpp $(TEST_DIR)/capture.hpp sycl_khr_print.hpp | build/
	@TIMEFORMAT="  compile fuzz_ffast_$*: %Rs"; time \
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -$* -ffast-math $(BUFFER_PATH) $(WA_$*) $< -o $@

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
	for opt in $(OPT_LEVELS); do \
	  t0=$$(date +%s%N); \
	  ./build/test_$$opt; rc=$$?; \
	  ms=$$(( ($$(date +%s%N) - t0) / 1000000 )); \
	  if [ $$rc -eq 0 ]; then echo "test -$$opt: PASS ($${ms}ms)"; \
	  else echo "test -$$opt: FAIL ($${ms}ms)"; fail=1; fi; \
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
