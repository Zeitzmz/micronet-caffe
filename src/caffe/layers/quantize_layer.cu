/*
 * Quantize Layer
 *
 * Created on: Aug 19, 2019
 * Author: hujie (Momenta)
 */

#include <cfloat>

#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/quantize_layer.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

#define STATS_BINS 2048

__device__ unsigned int counter = 0;

template <typename Dtype>
inline __device__ Dtype MaxData4(const Dtype* data, unsigned int index, bool abs);

template <>
inline __device__
float MaxData4(const float* data, unsigned int index, bool abs) {
  float4 val = ((float4*)data)[index];
  if (abs)
    return max(max(max(fabs(val.x), fabs(val.y)), fabs(val.z)), fabs(val.w));
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
      data[threadIdx.x] += data[threadIdx.x] + data[threadIdx.x + i];
    }
    __syncthreads();
  }
}

template <typename Dtype>
inline __device__ void ThreadMax(const Dtype* in, Dtype* out, 
    unsigned int count, unsigned int start, unsigned int stride, bool abs) {
  Dtype tmp = 0;
  for (unsigned int i = start; i < count / 4; i += stride) {
    tmp = max(tmp, MaxData4(in, i, abs));
  }
  // process remaining elements
  for (unsigned int i = start + count / 4 * 4; i < count; i += 4) {
    tmp = max(tmp, in[i]);
  }
  out[threadIdx.x] = tmp; 
  __syncthreads();
}

template <typename Dtype> 
inline __device__ void IncBlock(const Dtype* in, Dtype* out, 
    unsigned int* counter, bool* flag) {
  if (threadIdx.x == 0) {
    *out = *in;
    __threadfence();
    unsigned int value = atomicInc(counter, gridDim.x); // accumulate how many blocks have down
    *flag = (value == (gridDim.x - 1));
  }
  __syncthreads();
}

template <typename Dtype>
static __global__ void QuantizeGetAbsMax(const Dtype* data, int count, 
    Dtype* tmp_storage, Dtype* abs_max) {
  __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];
  __shared__ bool is_lastblock_done;
  unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
  ThreadMax(data, buffer, count, gid, blockDim.x * gridDim.x, true);
  // Reduce max on a single thread block in shared memory.
  ReduceMax(buffer);
  // Use last block to conduct max reduce across blocks.
  IncBlock(buffer, tmp_storage + blockIdx.x, &counter, &is_lastblock_done);
  if (is_lastblock_done) {
    ThreadMax(tmp_storage, buffer, gridDim.x, threadIdx.x, blockDim.x, false);
    ReduceMax(buffer);
    if (threadIdx.x == 0) {
      *abs_max = buffer[0];
    }
  }
}

template <typename Dtype>
static __global__ void QuantizeGetHist(const Dtype* data, int count,
    int num_bins, Dtype* abs_max, Dtype* hist) {
  // Generate data distribution
  Dtype step = *abs_max / num_bins; 
  __shared__ Dtype hist_buffer[STATS_BINS];
  for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    hist_buffer[i] = 0;
  }
  
  // generate hist within block     
  for (unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
      gid < count; gid += blockDim.x * gridDim.x) {
    Dtype abs_data = fabs(data[gid]);
    if (abs_data > Dtype(0.0001)) {
      int bin_index = max(abs_data / step, Dtype(num_bins - 1));
      caffe_gpu_atomic_add(Dtype(1), hist_buffer + bin_index);
    }
  }
  __syncthreads();

  // accumulate hist across blocks
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    caffe_gpu_atomic_add(hist_buffer[i], hist+i);
  }
}

template <typename Dtype>
static __global__ void QuantizeMinimizeKLDivs(const Dtype* hist, int num_bins, 
    int num_quant_bins, int num_kl_divs, Dtype tolerance, Dtype* kl_divs, 
    const Dtype* abs_max, Dtype* final_step) {
  __shared__ Dtype kl_buffer[CAFFE_CUDA_NUM_THREADS];
  __shared__ bool is_lastblock_done; // flag refer to whether the last block
  kl_buffer[threadIdx.x] = 0;

  for (int i = threadIdx.x; i < num_quant_bins; i += blockDim.x) {
    int gid = blockIdx.x * blockDim.x + i;
    if (gid >= num_kl_divs * num_quant_bins) 
      break;
    int kl_index = gid / num_quant_bins;
    int bin_index = gid % num_quant_bins;
    
    Dtype num_merged_bins = Dtype(num_quant_bins + kl_index) / num_quant_bins;
    Dtype start_not_rounded = num_merged_bins * bin_index;
    Dtype end_not_rounded = start_not_rounded + num_merged_bins;

    int start = floor(start_not_rounded);
    int end = ceil(end_not_rounded);
    start = min(start, num_bins - 1);
    end = min(end, num_bins);
    if (bin_index == num_quant_bins - 1) {
      end_not_rounded = num_bins;
      end = num_bins;
    }

    float non_zero_len = 0; // Use decimal fraction to precise equipartition
    float sum = 0;
    for (int j = start; j < end; ++j) {
      Dtype fraction = 1;
      if (j == start) {
        fraction = start + 1 - start_not_rounded;
      } else if (j == end - 1) {
        fraction = end_not_rounded - (end - 1);
      }
      if (hist[j] != Dtype(0)) {
        sum += hist[j] * fraction;
        non_zero_len += fraction;
      }
    }
    
    for (int j = start; j < end; ++j) {
      Dtype fraction = 1;
      if (j == start) {
        fraction = start + 1 - start_not_rounded;
      } else if (j == end - 1) {
        fraction = end_not_rounded - (end - 1);
      }
      if (hist[j] != Dtype(0)) {
        // Attention: here store -kl, than find the max kl
        kl_buffer[threadIdx.x] -= hist[j] * fraction 
            * log(hist[j] / (sum / non_zero_len));
      }
    }
  }
  __syncthreads();
  
  ReduceSum(kl_buffer);
 
  IncBlock(kl_buffer, kl_divs + blockIdx.x, &counter, &is_lastblock_done);

  // Use last block to conduct max reduce across kl_divs.
  if (is_lastblock_done) {
    ThreadMax(kl_divs, kl_buffer, gridDim.x, threadIdx.x, blockDim.x, false);
    ReduceMax(kl_buffer);
    
    // find the max index whose KL <= min_kl * tolerance
    Dtype loose_kl = kl_buffer[0] * tolerance;
    __shared__ unsigned int max_index[CAFFE_CUDA_NUM_THREADS];
    max_index[threadIdx.x] = 0;
    for (unsigned int index = threadIdx.x; index < gridDim.x; index += blockDim.x) {
      if (kl_divs[index] >= loose_kl && index > max_index[threadIdx.x]) {
        max_index[threadIdx.x] = index;
      }
    }
    ReduceMax(max_index);
    if (threadIdx.x == 0) {
      Dtype src_step = *abs_max / num_bins;
      *final_step = (num_quant_bins + max_index[0] + 0.5) * src_step / num_quant_bins;
    }
  }
}

template<typename Dtype>
static __global__ void Quantize(const Dtype* input, int count, 
    Dtype step, Dtype min_val, Dtype max_val, Dtype* output) {
  CUDA_KERNEL_LOOP(index, count) {
    output[index] = min(max(round(input[index] / step) * step, max_val), min_val);
  }
}

template <typename Dtype>
void QuantizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = bottom[0]->count();
  
  if (this->phase_ == TRAIN && !frozen_) {
    workspace_.Reshape(vector<int>(1, CAFFE_GET_BLOCKS(count)));
    hist_.Reshape(vector<int>(1, STATS_BINS));

    Dtype* workspace_data = workspace_.mutable_gpu_data();
    Dtype* hist_data = hist_.mutable_gpu_data();
    // use blobs[0] to storage temporary absmax value.
    Dtype* step_data = this->blobs_[0]->mutable_gpu_data(); 

    // Find the max abs data
    QuantizeGetAbsMax<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_data, count, workspace_data, step_data);
    CUDA_POST_KERNEL_CHECK;
    DLOG(INFO) << "Abs Max value is " << this->blobs_[0]->cpu_data()[0];
   
    // Gererate histgram from input data
    QuantizeGetHist<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_data, count, STATS_BINS, step_data, hist_data);
    CUDA_POST_KERNEL_CHECK;

    // Find the optimal threshold by minimizing KL_divergence.
    int num_quant_bins = positive_?(1 << precision_):(1 << (precision_ - 1));
    kl_divs_.Reshape(vector<int>(1, STATS_BINS - num_quant_bins));
    QuantizeMinimizeKLDivs<Dtype><<<STATS_BINS - num_quant_bins, CAFFE_CUDA_NUM_THREADS>>>(
        hist_data, STATS_BINS, num_quant_bins, STATS_BINS - num_quant_bins, 
        tolerance_, kl_divs_.mutable_gpu_data(), step_data, 
        this->blobs_[0]->mutable_gpu_data());
    DLOG(INFO) << "Step = " << this->blobs_[0]->cpu_data()[0];
    CUDA_POST_KERNEL_CHECK;
  } 

  // Calculate lower and upper bound
  Dtype step = this->blobs_[0]->mutable_cpu_data()[0];
  if (positive_) {
    min_ = Dtype(0);
    max_ = ((1 << precision_) - 1) * step;
  } else {
    min_ = -(1 << (precision_ - 1)) * step;
    max_ = ((1 << (precision_ - 1)) - 1) * step;
  }

  Quantize<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->gpu_data(), count, step, min_, max_, 
      top[0]->mutable_gpu_data());
}

template <typename Dtype>
static __global__ void QuantizeClipGradients(const int n, Dtype min, Dtype max,
    const Dtype* in_data, const Dtype* in_diff, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(i, n) {
    if (in_data[i] == min || in_data[i] == max) {
      out_diff[i] = Dtype(0);
    } else {
      out_diff[i] = in_diff[i];
    }
  }
}

template <typename Dtype>
void QuantizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    int count = top[0]->count();
    QuantizeClipGradients<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, min_, max_, 
        top[0]->gpu_data(), 
        top[0]->gpu_diff(), 
        bottom[0]->mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(QuantizeLayer);

}
