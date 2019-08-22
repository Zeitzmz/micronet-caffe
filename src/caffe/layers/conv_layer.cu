#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/quantize_util.cuh"

namespace caffe {

template <typename Dtype>
static __global__ void QuantizeChannel(const Dtype* in, Dtype* out, 
    int count, int dim, int precision) {
  __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];
  Dtype tmp = 0;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    int gid = blockIdx.x * dim + i;
    if (gid < count) {
      tmp = max(tmp, in[gid]);
    }
  }
  buffer[threadIdx.x] = tmp;
  __syncthreads();
  ReduceMax(buffer); 
  Dtype step = buffer[0] / ((1 << (precision - 1)) - 1);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    int gid = blockIdx.x * dim + i;
    if (gid < count) {
      out[gid] = round(in[gid] / step) * step;
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::QuantizeWeights_gpu(Dtype *buffer) {
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
    Dtype min_val = -(1 << (precision - 1)) * step;  
    Dtype max_val = ((1 << (precision - 1)) - 1) * step;

    Quantize<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        weights, count, step, min_val, max_val, weights);
  } else {
    int dim = this->blobs_[0]->count(1);
    int n = this->blobs_[0]->shape(0);
    QuantizeChannel<<<n, CAFFE_CUDA_NUM_THREADS>>>(
        weights, weights, count, dim, precision); 
  }  
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.has_quantize_param()) {
    // use top data to store temporary buffer.
    QuantizeWeights_gpu(top[0]->mutable_gpu_data());
  }
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

template void ConvolutionLayer<float>::QuantizeWeights_gpu(float* buffer);
template void ConvolutionLayer<double>::QuantizeWeights_gpu(double* buffer);
}  // namespace caffe
