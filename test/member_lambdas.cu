int main(void)
{
#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif
  auto d_lambda = [] __device__ { };
  auto hd_lambda = [] __host__ __device__ { };
  static_assert(__nv_is_extended_device_lambda_closure_type(
                  decltype(d_lambda)), "");
  static_assert(__nv_is_extended_host_device_lambda_closure_type(
                  decltype(hd_lambda)), "");
}
