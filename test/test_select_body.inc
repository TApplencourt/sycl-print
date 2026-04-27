#if defined(FMT_SYCL_HOST) || defined(FMT_SYCL_HOST_ACPP)
#include "test_body_host.inc"
#else
#include "test_body.inc"
#endif
