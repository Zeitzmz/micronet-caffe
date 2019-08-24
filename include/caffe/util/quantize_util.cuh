#ifndef CAFFE_UTIL_QUANTIZE_UTIL_H_
#define CAFFE_UTIL_QUANTIZE_UTIL_H_

#include <cfloat>

namespace caffe {

template <typename Dtype>
inline __device__ Dtype MaxData4(const Dtype* data, unsigned int index, bool abs);

template <>
inline __device__
float MaxData4(const float* data, unsigned int index, bool abs) {
  float4 val = ((float4*)data)[index];
  if (abs)
    return max(max(max(fabsf(val.x), fabsf(val.y)), fabsf(val.z)), fabsf(val.w));
  else
    return max(max(max(val.x, val.y), val.z), val.w);
}

template <>
inline __device__
double MaxData4(const double* data, unsigned int index, bool abs) {
  double4 val = ((double4*)data)[index];
  if (abs)
    return max(max(max(fabs(val.x), fabs(val.y)), fabs(val.z)), fabs(val.w));
  else
    return max(max(max(val.x, val.y), val.z), val.w);
}

template <typename Dtype>
inline __device__ void ReduceMax(Dtype* data) {
  for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      data[threadIdx.x] = max(data[threadIdx.x], data[threadIdx.x + i]);
    }
    __syncthreads();
  }
}

template <typename Dtype>
inline __device__ void ReduceSum(Dtype* data) {
  for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      data[threadIdx.x] += data[threadIdx.x + i];
    }
    __syncthreads();
  }
}

template <typename Dtype>
inline __device__ void ThreadMax(const Dtype* in, Dtype* out, 
    unsigned int count, unsigned int start, unsigned int stride,
    bool abs) {
  Dtype tmp = abs? 0 : -FLT_MAX;
  for (unsigned int i = start; i < count / 4; i += stride) {
    tmp = max(tmp, MaxData4(in, i, abs));
  }
  // process remaining elements
  for (unsigned int i = start + count / 4 * 4; i < count; i += 4) {
    if (abs)
      tmp = max(tmp, fabs(in[i]));
    else
      tmp = max(tmp, in[i]);
  }
  out[threadIdx.x] = tmp; 
  __syncthreads();
}

static __device__ unsigned int block_counter = 0;

template <typename Dtype> 
inline __device__ void IncBlock(const Dtype* in, Dtype* out, bool* flag) {
  if (threadIdx.x == 0) {
    *out = *in;
    __threadfence();
    unsigned int value = atomicInc(&block_counter, gridDim.x); // accumulate how many blocks have down
    bool last_block = (value == (gridDim.x - 1));
    *flag = last_block;
    if (last_block) block_counter = 0;
  }
  __syncthreads();
}

template <typename Dtype>
__global__ void GetAbsMax(const Dtype* data, int count, 
    Dtype* tmp_storage, Dtype* abs_max) {
  __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];
  __shared__ bool is_lastblock_done;
  unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
  ThreadMax(data, buffer, count, gid, blockDim.x * gridDim.x, true);
  // Reduce max on a single thread block in shared memory.
  ReduceMax(buffer);
  // Use last block to conduct max reduce across blocks.
  IncBlock(buffer, tmp_storage + blockIdx.x, &is_lastblock_done);
  if (is_lastblock_done) {
    ThreadMax(tmp_storage, buffer, gridDim.x, threadIdx.x, blockDim.x, false);
    ReduceMax(buffer);
    if (threadIdx.x == 0) {
      *abs_max = buffer[0];
    }
  }
}

template<typename Dtype>
inline __device__ Dtype quantize_op(const Dtype* input, Dtype step,
    Dtype min_val, Dtype max_val) {
  return min(max(round(*input / step) * step, max_val), min_val);
}

template<typename Dtype>
__global__ void Quantize(const Dtype* input, int count, 
    Dtype step, Dtype min_val, Dtype max_val, Dtype* output) {
  CUDA_KERNEL_LOOP(index, count) {
    output[index] = quantize_op(input + index, step, min_val, max_val);
  }
}

} // namespace caffe

#endif
