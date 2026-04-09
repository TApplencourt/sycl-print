SHELL    := /bin/bash
CXXFLAGS := -std=c++20 -Wall -Werror

ifdef USE_ACPP
  export PATH := $(HOME)/projet/p26.02/install/bin:$(PATH)
  CXX             := acpp
  SYCLFLAGS       := --acpp-targets=generic
  OPT_LEVELS      := O2
  BUFFER_PATH     := -DFMT_SYCL_BUFFER_PATH_ONLY
else
  CXX             := icpx
  SYCLFLAGS       := -fsycl
  OPT_LEVELS      := O0 O1 O2 O3
  BUFFER_PATH     :=
endif

# -DWA at O0: work around DPC++ bug (string literal through pointer segfaults)
WA_O0 := -DFMT_SYCL_WA_STR

# Derived binary names
TEST_STD      := $(addprefix test_std_,$(OPT_LEVELS))
TEST_SYCL     := $(addprefix test_sycl_,$(OPT_LEVELS))
FUZZ_STD      := $(addprefix fuzz_std_,$(OPT_LEVELS))
FUZZ_SYCL     := $(addprefix fuzz_sycl_,$(OPT_LEVELS))
FUZZ_SYCL_FM  := $(addprefix fuzz_sycl_ffast_,$(OPT_LEVELS))

ALL_BINS := $(TEST_STD) $(TEST_SYCL) $(FUZZ_STD) $(FUZZ_SYCL) $(FUZZ_SYCL_FM)

.PHONY: all build test test-format test-fuzz test-ffast \
        readme-examples clean

all: test

# ── Build rules ──────────────────────────────────────────────

build: $(ALL_BINS)

test_std_%: tests.cpp
	$(CXX) $(CXXFLAGS) -$* -DUSE_STD $(BUFFER_PATH) $(WA_$*) $< -o $@

test_sycl_%: tests.cpp sycl_khr_print.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -$* $(BUFFER_PATH) $(WA_$*) $< -o $@

fuzz_std_%: fuzz.cpp
	$(CXX) $(CXXFLAGS) -$* -DUSE_STD $(BUFFER_PATH) $(WA_$*) $< -o $@

fuzz_sycl_%: fuzz.cpp sycl_khr_print.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -$* $(BUFFER_PATH) $(WA_$*) $< -o $@

fuzz_sycl_ffast_%: fuzz.cpp sycl_khr_print.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -$* -ffast-math $(BUFFER_PATH) $(WA_$*) $< -o $@

# README examples
example_readme%: example_readme%.cpp sycl_khr_print.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $< -o $@

readme-examples: example_readme1 example_readme2
	@echo "README examples compiled OK."

# ── Test targets ─────────────────────────────────────────────

test: test-format test-fuzz test-ffast
	@echo "==============================="
	@echo "All tests passed."
	@echo "==============================="

test-format: $(TEST_STD) $(TEST_SYCL)
	@fail=0; \
	for opt in $(OPT_LEVELS); do \
	  echo "--- test -$$opt ---"; \
	  diff <(./test_std_$$opt) <(./test_sycl_$$opt 2>/dev/null) \
	    && echo "  PASS" \
	    || { echo "  FAIL"; fail=1; }; \
	done; \
	exit $$fail

test-fuzz: $(FUZZ_STD) $(FUZZ_SYCL)
	@fail=0; \
	for opt in $(OPT_LEVELS); do \
	  echo "--- fuzz -$$opt ---"; \
	  diff <(./fuzz_std_$$opt) <(./fuzz_sycl_$$opt 2>/dev/null) \
	    && echo "  PASS" \
	    || { echo "  FAIL"; fail=1; }; \
	done; \
	exit $$fail

test-ffast: $(FUZZ_STD) $(FUZZ_SYCL_FM)
	@fail=0; \
	for opt in $(OPT_LEVELS); do \
	  echo "--- fuzz -ffast-math -$$opt ---"; \
	  diff <(./fuzz_std_$$opt) <(./fuzz_sycl_ffast_$$opt 2>/dev/null) \
	    && echo "  PASS" \
	    || { echo "  FAIL"; fail=1; }; \
	done; \
	exit $$fail

clean:
	rm -f $(ALL_BINS) repro_O0 repro_O1
