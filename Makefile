SHELL    := /bin/bash
CXX      := icpx
CXXFLAGS := -std=c++20 -Wall -Werror
SYCLFLAGS := -fsycl

OPT_LEVELS := O0 O1 O2 O3

# -DWA at O0: work around DPC++ bug (string literal through pointer segfaults)
WA_O0 := -DFMT_SYCL_WA_STR

# Derived binary names
EXAMPLES_STD  := $(addprefix examples_std_,$(OPT_LEVELS))
EXAMPLES_SYCL := $(addprefix examples_sycl_,$(OPT_LEVELS))
FUZZ_STD      := $(addprefix fuzz_std_,$(OPT_LEVELS))
FUZZ_SYCL     := $(addprefix fuzz_sycl_,$(OPT_LEVELS))
FUZZ_SYCL_FM  := $(addprefix fuzz_sycl_ffast_,$(OPT_LEVELS))

ALL_BINS := $(EXAMPLES_STD) $(EXAMPLES_SYCL) $(FUZZ_STD) $(FUZZ_SYCL) $(FUZZ_SYCL_FM)

.PHONY: all build test test-examples test-fuzz test-ffast clean

all: test

# ── Build rules ──────────────────────────────────────────────

build: $(ALL_BINS)

examples_std_%: examples.cpp
	$(CXX) $(CXXFLAGS) -$* -DUSE_STD $(WA_$*) $< -o $@

examples_sycl_%: examples.cpp fmt_sycl.hpp dragonbox.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -$* $(WA_$*) $< -o $@

fuzz_std_%: fuzz.cpp
	$(CXX) $(CXXFLAGS) -$* -DUSE_STD $(WA_$*) $< -o $@

fuzz_sycl_%: fuzz.cpp fmt_sycl.hpp dragonbox.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -$* $(WA_$*) $< -o $@

fuzz_sycl_ffast_%: fuzz.cpp fmt_sycl.hpp dragonbox.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -$* -ffast-math $(WA_$*) $< -o $@

# ── Test targets ─────────────────────────────────────────────

test: test-examples test-fuzz test-ffast
	@echo "==============================="
	@echo "All tests passed."
	@echo "==============================="

test-examples: $(EXAMPLES_STD) $(EXAMPLES_SYCL)
	@fail=0; \
	for opt in $(OPT_LEVELS); do \
	  echo "--- examples -$$opt ---"; \
	  diff <(./examples_std_$$opt) <(./examples_sycl_$$opt 2>/dev/null) \
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
