/*
 * Quantize Layer
 *
 * Created on: Aug 19, 2019
 * Author: hujie (Momenta)
 */

#include <algorithm>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/quantize_layer.hpp"
// #include "caffe/util/fixed_point.hpp"

namespace caffe {

template <typename Dtype>
void QuantizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  precision_ = this->layer_param_.quantize_param().precision();
  frozen_ = this->layer_param_.quantize_param().frozen();
  positive_ = this->layer_param_.quantize_param().positive();
  channel_shared_ = this->layer_param_.quantize_param().channel_shared();
  tolerance_ = this->layer_param_.quantize_param().tolerance();

  if (this->phase_ == TRAIN) {
    CHECK_EQ(this->layer_param_.param_size(), 1);
    CHECK_EQ(this->layer_param_.param(0).lr_mult(), 0);
    CHECK_EQ(this->layer_param_.param(0).decay_mult(), 0);
  }
  this->param_propagate_down_.resize(1);
  this->param_propagate_down_[0] = false;
   
  this->blobs_.resize(1);
  if (channel_shared_) {
    this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, 1)));
  } else {
    NOT_IMPLEMENTED;
   // this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, bottom[0]->channels())));
  }
}

template <typename Dtype>
void QuantizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void QuantizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void QuantizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    NOT_IMPLEMENTED;
  }
}

#ifdef CPU_ONLY
STUB_GPU(QuantizeLayer);
#endif

INSTANTIATE_CLASS(QuantizeLayer);
REGISTER_LAYER_CLASS(Quantize);

}  // namespace caffe
