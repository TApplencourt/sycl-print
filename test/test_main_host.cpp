#include "capture.hpp"

bool test_integers();
bool test_floats();
bool test_strings();
bool test_layout();
bool test_misc();
bool test_fuzz();

int main() {
  bool ok = true;
  ok &= test_integers();
  ok &= test_floats();
  ok &= test_strings();
  ok &= test_layout();
  ok &= test_misc();
  ok &= test_fuzz();
  return ok ? 0 : 1;
}
