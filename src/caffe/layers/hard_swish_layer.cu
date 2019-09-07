#include <algorithm>
#include <vector>

#include "caffe/layers/hard_swish_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void HardSwishForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(i, n) {
    Dtype tmp = in[i] + 3;  // x + 3
    tmp = tmp > 6 ? Dtype(6) : tmp;
    tmp = tmp < 0 ? Dtype(0) : tmp; // ReLU6(x+3)
    out[i] = in[i] * tmp / Dtype(6);  // x * ReLU6(x+3) / 6
  }
}

template <typename Dtype>
void HardSwishLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  HardSwishForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void HardSwishBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(i, n) {
    Dtype tmp = (in_data[i] * 2 + 3) / Dtype(6);  // (x * 2 + 3) / 6
    tmp = in_data[i] < -3 ? Dtype(0) : tmp;  // 0 if x < -3
    tmp = in_data[i] > 3 ? Dtype(1) : tmp;  // 1 if x > 3
    out_diff[i] = in_diff[i] * tmp;
  }
}

template <typename Dtype>
void HardSwishLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();

    // For in-place computation
    if (top[0] == bottom[0]) {
      bottom_data = bottom_memory_.gpu_data();
    }

    // NOLINT_NEXT_LINE(whitespace/operators)
    HardSwishBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(HardSwishLayer);


}  // namespace caffe
