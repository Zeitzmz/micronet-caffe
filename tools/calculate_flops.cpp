#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <gflags/gflags.h>

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
//#include "caffe/util/gpu_memory.hpp"

using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Blob;
using std::vector;
using std::string;
using boost::shared_ptr;
#define BIAS_BITS 16
#define ADD_BITS 16
#define BASELINE_FLOPS 1170
#define BASELINE_PARAM 6.9

DEFINE_string(model, "",
    "The model definition protocol buffer text file.");

DEFINE_string(config_file, "",
    "The sparsity and bit information of each layer.");  //layer_name, sparsity, add_bits, mul_bits,
DEFINE_bool(ignore_batchsize, true, 
    "Whether ignore batchsize.");

int main(int argc, char** argv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = false; 
  FLAGS_minloglevel = 2;

  if (FLAGS_model.size() == 0) {
    std::cerr << "Usage: calculate_flops [OPTIONS ...] \n\n" 
        << "  --model             The model definition protocol buffer text file.\n"
        << "  --config_file       Format as line \"layer_name, sparsity, param_bits, add_bits, mul_bits\".\n" 
        << "  --ignore_batchsize  Must be true (Default) or false.\n\n";
    return -1;
  };
  
  // Init caffe
#ifndef CPU_ONLY
    Caffe::SetDevice(1);
    Caffe::set_mode(Caffe::GPU);
    //std::cout<<"set GPU 1"<<std::endl;  
#else
    Caffe::set_mode(Caffe::CPU);
#endif 

  // Instantiate the caffe net.
  //std::cout<<"Before Build Net"<<std::endl;
  Net<float> net(FLAGS_model, caffe::TEST);
  //std::cout<<"End    Build Net"<<std::endl;
  unsigned long total_model_size = 0;
  unsigned long total_mul_flops = 0;
  unsigned long total_add_flops = 0;

  const vector<shared_ptr<Layer<float> > >& layers = net.layers();
  const vector<string>& layer_names = net.layer_names();
  //const vector<vector<Blob<float>*> >& bottom_vecs = net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = net.top_vecs();
  int index = FLAGS_ignore_batchsize ? 1 : 0;
  //bool multiadd_mode = (FLAGS_mode == "multi-add");
  
  std::map<std::string, vector<float>> config_info;
  if(FLAGS_config_file.size()){
    std::ifstream config_file(FLAGS_config_file.c_str());
    std::string line_string;
    while(std::getline(config_file, line_string)){
      int length = 0;
      int start = 0;
      string layer_name;
      vector<float> layer_config;
      while(start<line_string.size()){
        while(start+length<=line_string.size() && line_string[start+length]!=' '){
          length++;
        }
        if(start == 0){
          layer_name = line_string.substr(start, length);
        }
        else{
          //std::cout<<atof(line_string.substr(start,length).c_str())<<std::endl;
          layer_config.push_back(float(atof(line_string.substr(start,length).c_str())));
        }
        start+=length;
        length=1;
      }
      config_info.insert({layer_name, layer_config});    
    }
  }
  /*
  for(auto i : config_info){
    std::cout<<"layer_name:"<<i.first<<" ";
    for(auto j : i.second){
      std::cout<<j<<" ";
    }
    std::cout<<std::endl;
  }
  */

  std::map<int,int> bottom_quantize_info;
  for (int i=0; i < layers.size(); ++i){
    if(strcmp(layers[i]->type(), "Quantize") == 0){
      int top_id = net.top_ids(i)[0];
      int top_precision = layers[i]->layer_param().quantize_param().precision();
      bottom_quantize_info.insert({top_id, top_precision});
    }
  }


  for (int i = 0; i < layers.size(); ++i) {
    const vector<caffe::shared_ptr<Blob<float> > >& params=net.layers()[i]->blobs();
    float add_bit_flops = 0;
    float mul_bit_flops = 0;
    unsigned long layer_size = 0;

    float layer_sparsity = 0.0;
    int param_bits = 32;
    //int add_bits = 32;
    int add_bits = ADD_BITS;
    int mul_bits = 32;

    std::string layer_name = layer_names[i];
   
    //set sparsity
    auto config_iter = config_info.find(layer_name);
    if(config_iter!=config_info.end()){
      layer_sparsity = config_iter->second[0];
      //param_bits = config_iter->second[1];
      //add_bits = config_iter->second[2];
      //mul_bits = config_iter->second[3];
    }

    // set mul_bits (input_bits)
    vector<int> layer_bottom_ids = net.bottom_ids(i);
    int tmp_mul_bits = -1;
    for(auto i  : layer_bottom_ids){
      if(bottom_quantize_info.find(i)!=bottom_quantize_info.end()){
        tmp_mul_bits = std::max(tmp_mul_bits, bottom_quantize_info.find(i)->second);
      }
    } 
    mul_bits = tmp_mul_bits > 0 ? std::min(tmp_mul_bits, mul_bits) : mul_bits; 

    // set_param_bits
    if(layers[i]->layer_param().has_quantize_param()){
      param_bits = layers[i]->layer_param().quantize_param().precision();
    }

    if (strcmp(layers[i]->type(), "Convolution") == 0 ||
        strcmp(layers[i]->type(), "DepthwiseConvolution") == 0 ||
        strcmp(layers[i]->type(), "InnerProduct") == 0){
      CHECK_GE(params.size(), 1);
      CHECK_LE(params.size(), 2);
      unsigned long op_each_output = params[0]->count();

      // model size count 
      layer_size = op_each_output * param_bits * (1 - layer_sparsity);
      layer_size += layer_sparsity > 0 ? op_each_output : 0;
      
      // bit * flops count
      int vector_length = params[0]->count(1) * (1-layer_sparsity);

      mul_bit_flops += float(vector_length) * int(top_vecs[i][0]->count(index)) * int(std::max(mul_bits, param_bits));
      add_bit_flops += float(vector_length - 1) * int(top_vecs[i][0]->count(index)) * int(std::max(add_bits, param_bits));
      
      // use bias 
      if(params.size()==2){
        layer_size += params[1]->count() * BIAS_BITS; // bias use 32bits
        add_bit_flops += top_vecs[i][0]->count(index) * std::max(BIAS_BITS, add_bits);
      }

    } else if (strcmp(layers[i]->type(), "Scale") == 0) {
      mul_bit_flops += top_vecs[i][0]->count(index) * mul_bits;
    } else if (strcmp(layers[i]->type(), "Pooling") == 0){
      int ksize = layers[i]->layer_param().pooling_param().kernel_size();
      //int stride = layers[i]->layer_param().pooling_param().stride();
      int pool_method = layers[i]->layer_param().pooling_param().pool();
      if(pool_method==0){
        std::cout<<"Not Implemented for MAX Pooling"<<std::endl;
        //exit();
      }
      else if(pool_method==1){
        mul_bit_flops += top_vecs[i][0]->shape()[1] * mul_bits;
        add_bit_flops += ( ksize * ksize - 1 ) * top_vecs[i][0]->shape()[1] * std::max(add_bits, param_bits);
      }
    } else if (strcmp(layers[i]->type(), "ReLU") == 0) {
        mul_bit_flops += (top_vecs[i][0]->count(index)) * mul_bits;
        //mul_bit_flops += (top_vecs[i][0]->count(index)) * 8;
    } 
    else if (strcmp(layers[i]->type(), "Eltwise") == 0) {
      if ((layers[i]->layer_param().eltwise_param().operation() 
          == caffe::EltwiseParameter_EltwiseOp_SUM)) {
        add_bit_flops += (top_vecs[i][0]->count(index)) * std::max(add_bits, param_bits);
      }
    }

    //std::cout<<"Layer: "<<layer_names[i]<<" : param_bits: "<< param_bits<<", layer size: "<< layer_size/32<<std::endl;
    /*if (add_bit_flops || mul_bit_flops) {
      std::cout << "Layer: " << layer_names[i] << ", type: " << layers[i]->type() 
          << ", operations add: "<< add_bit_flops / float(1e6) / 32 << " M "
          << ", operations mul: "<< mul_bit_flops / float(1e6) / 32 << " M "
          << ", layer param : "<< layer_size/32 <<"   "<<total_mul_flops/float(1e6)/32<<std::endl;
    }*/
    int bottom0_bits=32;
    int bottom1_bits=32;
     
    for(int i=0; i<layer_bottom_ids.size(); ++i){
      if(bottom_quantize_info.find(layer_bottom_ids[i])!=bottom_quantize_info.end()){
        if(i==0)
          bottom0_bits = bottom_quantize_info.find(layer_bottom_ids[i])->second;
        if(i==1)
          bottom1_bits = bottom_quantize_info.find(layer_bottom_ids[i])->second;
      }
    }
 
    
    std::cout<<layer_names[i]<<", "<<layers[i]->type()<<", "<<bottom0_bits<<", "<<(layer_bottom_ids.size()>1 ? std::to_string(bottom1_bits) : "_")<<", "<<layer_sparsity<<", "<<add_bits<<", "<<std::max(param_bits,mul_bits)<<", "<<param_bits<<", "<<BIAS_BITS<<std::endl;    

    total_mul_flops += mul_bit_flops;
    total_add_flops += add_bit_flops;
    total_model_size += layer_size;
  }
  std::cout << "\n*********** summary ****************" << std::endl;
  std::cout << "Model: " << FLAGS_model << std::endl;
  std::cout << "Model name: " << net.name() << std::endl;
  std::cout << "Ignore batchsize: " << (FLAGS_ignore_batchsize ? "True":"False") << std::endl;
  std::cout << "Total mul operations: " << total_mul_flops / float(1e6) / 32 << " M"<< std::endl;
  std::cout << "Total add operations: " << total_add_flops / float(1e6) / 32 << " M"<< std::endl;
  std::cout << "Total model size    : " << total_model_size / float(1e6) / 32 << " M"<< std::endl;
  std::cout << "Total score         : " << total_model_size / float(1e6) / 32 / float(BASELINE_PARAM) + (total_add_flops + total_mul_flops) / float(1e6) / 32 / float(BASELINE_FLOPS) << std::endl;
  std::cout << "***********************************\n" << std::endl;
  return 0;
}

