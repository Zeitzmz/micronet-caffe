/*
 * Quantize Layer
 *
 * Created on: Aug 19, 2019
 * Author: hujie (Momenta)
 */

#include <cfloat>

#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/quantize_layer.hpp"
#include "caffe/util/quantize_util.cuh"

namespace caffe {

#define STATS_BINS 2048

template <typename Dtype>
static __global__ void GetHist(const Dtype* data, int count,
    int num_bins, Dtype* abs_max, unsigned int* hist) {
  // Generate data distribution
  Dtype step = *abs_max / num_bins; 
  __shared__ unsigned int hist_buffer[STATS_BINS];
  for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    hist_buffer[i] = 0;
  }
  
  // generate hist within block     
  for (unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
      gid < count; gid += blockDim.x * gridDim.x) {
    Dtype abs_data = fabs(data[gid]);
    if (abs_data > Dtype(0.0001)) {
      int bin_index = min(int(abs_data / step), num_bins - 1);
      atomicAdd(hist_buffer + bin_index, 1);
    }
  }
  __syncthreads();

  // accumulate hist across blocks
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    atomicAdd(hist+i, hist_buffer[i]);
  }
}

template <typename Dtype>
static __global__ void MinimizeKLDivs(const unsigned int* hist, int num_bins, 
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
 
  IncBlock(kl_buffer, kl_divs + blockIdx.x, &is_lastblock_done);

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
    unsigned int* hist_data = hist_.mutable_gpu_data();
    // use blobs[0] to storage temporary absmax value.
    Dtype* step_data = this->blobs_[0]->mutable_gpu_data(); 

    // Find the max abs data
    GetAbsMax<Dtype><<<CAFFE_GET_BLOCKS(count/4), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_data, count, workspace_data, step_data);
    CUDA_POST_KERNEL_CHECK;
    DLOG(INFO) << "Abs Max value is " << this->blobs_[0]->cpu_data()[0];
   
    // Gererate histgram from input data
    GetHist<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_data, count, STATS_BINS, step_data, hist_data);
    CUDA_POST_KERNEL_CHECK;

    // Find the optimal threshold by minimizing KL_divergence.
    int num_quant_bins = positive_?(1 << precision_):(1 << (precision_ - 1));
    kl_divs_.Reshape(vector<int>(1, STATS_BINS - num_quant_bins));
    MinimizeKLDivs<Dtype><<<STATS_BINS - num_quant_bins, CAFFE_CUDA_NUM_THREADS>>>(
        hist_data, STATS_BINS, num_quant_bins, STATS_BINS - num_quant_bins, 
        tolerance_, kl_divs_.mutable_gpu_data(), step_data, 
        this->blobs_[0]->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    DLOG(INFO) << "Step = " << this->blobs_[0]->cpu_data()[0];
  } 

  // Calculate lower and upper bound
  Dtype step = this->blobs_[0]->cpu_data()[0];
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
  CUDA_POST_KERNEL_CHECK;
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
