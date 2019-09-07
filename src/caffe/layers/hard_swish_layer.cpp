#include <algorithm>
#include <vector>

#include "caffe/layers/hard_swish_layer.hpp"

namespace caffe {

template <typename Dtype>
void HardSwishLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  for (int i = 0; i < count; ++i) {
    Dtype tmp = bottom_data[i] + 3;  // x + 3
    tmp = std::min(std::max(tmp, Dtype(0)), Dtype(6));  // ReLU6(x+3)
    top_data[i] = bottom_data[i] * tmp / Dtype(6);  // x * ReLU6(x+3) / 6
  }
}

template <typename Dtype>
void HardSwishLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void HardSwishLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();

    // For in-place computation
    if (top[0] == bottom[0]) {
      bottom_data = bottom_memory_.cpu_data();
    }

    for (int i = 0; i < count; ++i) {
      Dtype tmp = (bottom_data[i] * 2 + 3) / Dtype(6);  // (x * 2 + 3) / 6
      tmp = bottom_data[i] < -3 ? Dtype(0) : tmp;  // 0 if x < -3
      tmp = bottom_data[i] > 3 ? Dtype(1) : tmp;  // 1 if x > 3
      bottom_diff[i] = top_diff[i] * tmp;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(HardSwishLayer);
#endif

INSTANTIATE_CLASS(HardSwishLayer);
REGISTER_LAYER_CLASS(HardSwish);

}  // namespace caffe
