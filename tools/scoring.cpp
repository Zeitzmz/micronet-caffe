#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <gflags/gflags.h>

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Blob;
using std::vector;
using std::string;
using boost::shared_ptr;

#define BASELINE_FLOPS 1170
#define BASELINE_PARAM 6.9

#define BASE_BITS 16 // use fp16 accumulation

DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(sparsity, "",
    "The sparsity of each layer.");  

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model.size() == 0) {
    LOG(INFO) << "Usage: scoreing [OPTIONS ...] \n\n" 
        << "  --model             PATH: The model definition protocol buffer text file.\n"
        << "  --sparsity          FILE: Format as line \"layer_name, sparsity.\n";
    return -1;
  }
 
  // Init caffe
#ifndef CPU_ONLY
    Caffe::SetDevice(1);
    Caffe::set_mode(Caffe::GPU);
#else
    Caffe::set_mode(Caffe::CPU);
#endif 

  unsigned long total_storage_bits = 0;
  unsigned long total_mul_bitops = 0;
  unsigned long total_add_bitops = 0;

  // Instantiate the caffe net.
  Net<float> net(FLAGS_model, caffe::TEST);
  const vector<shared_ptr<Layer<float> > >& layers = net.layers();
  const vector<string>& layer_names = net.layer_names();
  const vector<vector<Blob<float>*> >& bottom_vecs = net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = net.top_vecs();
 
  // Parse sparsity configuration
  std::map<std::string, float> sparsity_info;
  if (FLAGS_sparsity.size()) {
    std::ifstream fin(FLAGS_sparsity.c_str());
    std::string layer_name;
    float sparsity;
    while (fin >> layer_name >> sparsity) {
      sparsity_info.insert({layer_name, sparsity});    
    }
  }

  std::map<int,int> top_layer_id;
  for (int i=0; i < layers.size(); ++i) {
    for (int j=0; j < net.top_ids(i).size(); ++j) {
      top_layer_id[net.top_ids(i)[j]] = i;
    }
  }

  // Parse fake quantization blob
  std::map<int,int> quantized_blobs;
  for (int i=0; i < layers.size(); ++i) {
    if (!strcmp(layers[i]->type(), "Quantize")) {
      for (int j=0 ;j<net.top_ids(i).size(); ++j) {
        int top_id = net.top_ids(i)[j];
        int top_precision = layers[i]->layer_param().quantize_param().precision();
        quantized_blobs.insert({top_id, top_precision});
      }
    }
  }

  for (int i=0; i < layers.size(); ++i) {
    if (!strcmp(layers[i]->type(), "Split") || !strcmp(layers[i]->type(), "Slice") 
        || !strcmp(layers[i]->type(), "Reshape")) {
      for (int j=0 ;j<net.top_ids(i).size(); ++j) {
        int top_id = net.top_ids(i)[j];
        if (quantized_blobs.find(net.bottom_ids(i)[0])!=quantized_blobs.end()) {
          int precision = quantized_blobs[net.bottom_ids(i)[0]];
          quantized_blobs.insert({top_id, precision});
        }
      }
    } else if (!strcmp(layers[i]->type(), "Concat")) {
      int top_id = net.top_ids(i)[0];
      if (quantized_blobs.find(top_id) != quantized_blobs.end()) {
        int precision = quantized_blobs[top_id];
        for (int j=0; j < net.bottom_ids(i).size(); ++j) {
          int bottom_id = net.bottom_ids(i)[j];
          quantized_blobs.insert({bottom_id, precision});
        }
      }
    }
  }

  // count for each layer
  for (int i = 0; i < layers.size(); ++i) {
    bool verbose = true;

    const std::string layer_name = layer_names[i];
    const vector<caffe::shared_ptr<Blob<float> > >& params = net.layers()[i]->blobs();
    const vector<Blob<float>*> &bottoms = bottom_vecs[i];
    const vector<Blob<float>*> &tops = top_vecs[i];

    float add_bitops = 0;
    float mul_bitops = 0;
    float storage_bits = 0;

    float layer_sparsity = 0.0;
    string layer_sparsity_str = "-";

    int param_bits = BASE_BITS;
    int add_bits = BASE_BITS;
    int mul_bits = BASE_BITS;
    
    // set sparsity
    auto sparsity_iter = sparsity_info.find(layer_name);
    if (sparsity_iter != sparsity_info.end()) {
      layer_sparsity = sparsity_iter->second;
      layer_sparsity_str = std::to_string(layer_sparsity);
    }
   
    // get input bits 
    const vector<int> &layer_bottom_ids = net.bottom_ids(i);
    vector<int> bottoms_bits(layer_bottom_ids.size(), BASE_BITS); 
    for (int j=0; j<layer_bottom_ids.size(); ++j) {
      if (quantized_blobs.find(layer_bottom_ids[j]) != quantized_blobs.end()) {
        bottoms_bits[j] = quantized_blobs[layer_bottom_ids[j]];
      }
    }
    if (layer_bottom_ids.size()) {
      mul_bits = bottoms_bits[0];
    }

    // set_param_bits
    if (layers[i]->layer_param().has_quantize_param()) {
      param_bits = layers[i]->layer_param().quantize_param().precision();
    }

    if (strcmp(layers[i]->type(), "Convolution") == 0 ||
        strcmp(layers[i]->type(), "DepthwiseConvolution") == 0 ||
        strcmp(layers[i]->type(), "InnerProduct") == 0) {
      CHECK_GE(params.size(), 1);
      CHECK_LE(params.size(), 2);

      unsigned long op_each_output = params[0]->count();

      // count param storage
      storage_bits = op_each_output * param_bits * (1 - layer_sparsity); // sparse param storage
      storage_bits += layer_sparsity > 0 ? op_each_output : 0; // mask storage
      if (params.size() == 2) {
        storage_bits += params[1]->count() * add_bits;
      }
      
      // count ops
      int vector_length = params[0]->count(1) * (1 - layer_sparsity);
      mul_bits = std::max(bottoms_bits[0], param_bits);
      mul_bitops += float(vector_length) * tops[0]->count(1) * mul_bits;
      add_bitops += float(vector_length - 1) * tops[0]->count(1) * add_bits;
      if (params.size()==2) {
        add_bitops += tops[0]->count(1) * add_bits;
      }
    } else if (strcmp(layers[i]->type(), "Scale") == 0) {
      CHECK_EQ(bottoms.size(), 2);
      mul_bits = std::max(bottoms_bits[0], bottoms_bits[1]);
      mul_bitops += tops[0]->count(1) * mul_bits;
    } else if (strcmp(layers[i]->type(), "Pooling") == 0) {
      int ksize = layers[i]->layer_param().pooling_param().kernel_size();
      int pool_method = layers[i]->layer_param().pooling_param().pool();
      if (ksize == 0) { // global pooling
        ksize = bottoms[0]->shape(2);
      }
      if (pool_method == 0) {
        LOG(FATAL) << "Not Implemented for MAX Pooling!";
      }
      else if(pool_method == 1) { // ave pooling
        mul_bitops += tops[0]->count(1) * mul_bits;
        add_bitops += (ksize * ksize - 1) * tops[0]->count(1) * add_bits;
      }
    } else if (!strcmp(layers[i]->type(), "ReLU")) {
      mul_bitops += (tops[0]->count(1)) * mul_bits;
    } else if (!strcmp(layers[i]->type(), "Sigmoid")) {
      mul_bitops += 2 * (tops[0]->count(1)) * mul_bits;
      add_bitops += (tops[0]->count(1)) * mul_bits;
    } else if (!strcmp(layers[i]->type(), "HardSwish")) {
      mul_bitops += 3 * (tops[0]->count(1)) * mul_bits;
    } else if (!strcmp(layers[i]->type(), "Eltwise")) {
      if ((layers[i]->layer_param().eltwise_param().operation() 
          == caffe::EltwiseParameter_EltwiseOp_SUM)) {
        add_bitops += (tops[0]->count(1)) * add_bits;
      } else {
        LOG(FATAL) << "Not Implemented!"; 
      }
    } else {
      verbose = false;
      DLOG(INFO) << "Skip layer " << layers[i]->layer_param().name();
    }

    if (verbose) { 
      LOG(INFO) << layer_name << ", "
                << layers[i]->type() << ", "
                << bottoms_bits[0] << ", "
                << (bottoms_bits.size() > 1 ? std::to_string(bottoms_bits[1]) : "-") << ", " 
                << (params.size() == 0 ? "-" : std::to_string(param_bits)) << ", "
                << layer_sparsity_str << ", "
                << (mul_bitops == 0 ? "-" : std::to_string(mul_bits)) << ", " 
                << (add_bitops == 0 ? "-" : std::to_string(add_bits)) << ", "
                << (mul_bitops == 0 ? "-" : std::to_string(int(mul_bitops/ 32))) << ", " 
                << (add_bitops == 0 ? "-" : std::to_string(int(add_bitops / 32))) << ", "
                << (params.size() == 0 ? "-" : std::to_string(int(storage_bits / 32))) << std::endl;
    }

    total_mul_bitops += mul_bitops;
    total_add_bitops += add_bitops;
    total_storage_bits += storage_bits;
  }

  float param_score = total_storage_bits / 1e6 / 32 / BASELINE_PARAM;
  float ops_score = (total_add_bitops + total_mul_bitops) / 1e6 / 32 / BASELINE_FLOPS;
  LOG(INFO) << "*********** summary ****************";
  LOG(INFO) << "Model: " << FLAGS_model;
  LOG(INFO) << "Model name: " << net.name();
  LOG(INFO) << "Total mul operations: " << total_mul_bitops / float(1e6) / 32 << " M";
  LOG(INFO) << "Total add operations: " << total_add_bitops / float(1e6) / 32 << " M";
  LOG(INFO) << "Total model size    : " << total_storage_bits / float(1e6) / 32 << " M";
  LOG(INFO) << "Total param score   : " << param_score; 
  LOG(INFO) << "Total bitops score  : " << ops_score; 
  LOG(INFO) << "Total score         : " << param_score + ops_score; 
  LOG(INFO) << "***********************************\n";
  return 0;
}

