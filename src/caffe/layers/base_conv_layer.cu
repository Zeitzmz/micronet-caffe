#include <algorithm>
#include <vector>

#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/quantize_util.cuh"

namespace caffe {

template <typename Dtype>
static __global__ void QuantizeChannel(const Dtype* in, Dtype* out, 
    int dim, int precision) {
  __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];
  Dtype tmp = FLT_MIN;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    int gid = blockIdx.x * dim + i;
    tmp = max(tmp, fabs(in[gid]));
  }
  buffer[threadIdx.x] = tmp;
  __syncthreads();
  ReduceMax(buffer); 
  Dtype step = buffer[0] / ((1 << (precision - 1)) - 1);
  Dtype min_val = -(1 << (precision - 1)) * step; 
  Dtype max_val = ((1 << (precision - 1)) - 1) * step;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    int gid = blockIdx.x * dim + i;
    out[gid] = quantize_op(in + gid, step, min_val, max_val); 
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::QuantizeWeights_gpu(Dtype *buffer) {
  bool frozen = this->layer_param_.quantize_param().frozen();
  if ((this->phase_ == TRAIN && !frozen) || 
      (this->phase_ == TEST && !quantize_setup_)) {
    int precision = this->layer_param_.quantize_param().precision();
    bool channel_shared = this->layer_param_.quantize_param().channel_shared();
    int count = this->blobs_[0]->count();
    Dtype* weights = this->blobs_[0]->mutable_gpu_data();

    if (channel_shared) {
      CHECK_NOTNULL(buffer);
      GetAbsMax<Dtype><<<CAFFE_GET_BLOCKS(count/4), CAFFE_CUDA_NUM_THREADS>>>(
          weights, count, buffer, buffer);
      CUDA_POST_KERNEL_CHECK;
      
      Dtype step;
      caffe_copy(1, buffer, &step);
      step /= (1 << (precision - 1)) - 1;
      Dtype min_val = -(1 << (precision - 1)) * step;  
      Dtype max_val = ((1 << (precision - 1)) - 1) * step;

      Quantize<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          weights, count, step, min_val, max_val, weights);
      CUDA_POST_KERNEL_CHECK;
    } else {
      int dim = this->blobs_[0]->count(1);
      int n = this->blobs_[0]->shape(0);
      QuantizeChannel<<<n, CAFFE_CUDA_NUM_THREADS>>>(
          weights, weights, dim, precision); 
      CUDA_POST_KERNEL_CHECK;
    }
    quantize_setup_ = true;
  }
}

template void BaseConvolutionLayer<float>::QuantizeWeights_gpu(float* buffer);
template void BaseConvolutionLayer<double>::QuantizeWeights_gpu(double* buffer);

}  // namespace caffe
