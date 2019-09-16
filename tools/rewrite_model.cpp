#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device ID");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "TEST",
    "Optional; network phase (TRAIN or TEST)");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_string(output, "",
    "Oytput path.");

// Parse phase from flags
caffe::Phase get_phase_from_flags(caffe::Phase default_value) {
  if (FLAGS_phase == "")
    return default_value;
  if (FLAGS_phase == "TRAIN")
    return caffe::TRAIN;
  if (FLAGS_phase == "TEST")
    return caffe::TEST;
  LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
  return caffe::TRAIN;  // Avoid warning
}

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Net<float>* net, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    net->CopyTrainedLayersFrom(model_names[i]);
  }
}

int main(int argc, char** argv) {
  caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);

  CHECK_GT(FLAGS_model.size(), 0) << "Need a model prototxt as input.";
  CHECK_GT(FLAGS_output.size(), 0) << "Need a output path to save.";

  if (FLAGS_gpu.size()) {
    int gpu_id = boost::lexical_cast<int>(FLAGS_gpu);
    LOG(INFO) << "Use GPU with device ID " << gpu_id;
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpu_id);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU. If you use CuDNNBatchNormLayer, you must specify --gpu parameter.";
    Caffe::set_mode(Caffe::CPU);
  }
 
//  // Set device scope for gpu memory allocation
//#ifndef CPU_ONLY
//  vector<int> device_id;
//  if (Caffe::mode() == Caffe::GPU)
//    device_id.resize(1, Caffe::device_id());
//  caffe::GPUMemory::Scope gpu_memory_scope(device_id);
//#endif

  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, phase);
  if (FLAGS_weights.size()) {
    CopyLayers(&caffe_net, FLAGS_weights);
  }
  
  caffe::NetParameter out_param;
  caffe_net.ToProto(&out_param);
  WriteProtoToBinaryFile(out_param, FLAGS_output.c_str());
  LOG(INFO) << "Finish!";
}
