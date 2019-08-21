/*
 * Quantize Layer
 *
 * Created on: Aug 19, 2019
 * Author: hujie (Momenta)
 */

#ifndef CAFFE_QUANTIZE_LAYER_HPP_
#define CAFFE_QUANTIZE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Minimize Kullback–Leibler divergence to find the optimal threshold used for quantization.
 *  Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
 *        
 */
template <typename Dtype>
class QuantizeLayer : public NeuronLayer<Dtype> {
 public:
  explicit QuantizeLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Quantize"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int precision_;
  bool frozen_;
  bool positive_;
  bool channel_shared_;
  float tolerance_;
  
  Blob<Dtype> hist_;
  Blob<Dtype> kl_divs_;
  Blob<Dtype> workspace_;

  Dtype min_;
  Dtype max_;
};

}  // namespace caffe

#endif  // CAFFE_QUANTIZE_LAYER_HPP_
