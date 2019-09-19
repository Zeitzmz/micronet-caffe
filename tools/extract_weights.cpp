#include <glog/logging.h>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>


#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::string;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;

int main(int argc, char** argv) {
  //ReadNetParamsFromTextFileOrDie(FLAGS_model, &param);
  if (argc != 3) {
    LOG(INFO) << "usage: " << argv[0] << " trained.caffemodel weights.txt";
    return -1;
  }
  
  FLAGS_alsologtostderr = 1;
  caffe::GlobalInit(&argc, &argv);

  const string model_file(argv[1]);
  const string weights_file(argv[2]);
 // Net<float> caffe_net(deploy_file, caffe::TRAIN);

  caffe::NetParameter net_param; 
  ReadNetParamsFromBinaryFileOrDie(model_file, &net_param);
  LOG(INFO) << "Finish read params.";
   
  const int layer_size = net_param.layer_size();
  LOG(INFO) << "Total layers: " << layer_size;
  std::ofstream fout;
  fout.open(weights_file.c_str());

  for (int i = 0; i < layer_size; ++i) {
    const caffe::LayerParameter  &current_param = net_param.layer(i);
    LOG(INFO) << "Processing layer " << current_param.name(); 
    const int blob_size = current_param.blobs_size();
    for (int j = 0; j < blob_size; ++j) {
      fout << current_param.name() << " blob" << j << ": \n";
      const caffe::BlobProto &blob = current_param.blobs(j);
      for(int c = 0; c < blob.data_size(); ++c) {
        fout << blob.data(c) << " ";
      }
      fout<< "\n\n";
    }
    fout<<"\n\n";
  }
  fout.close();
  LOG(INFO) << "Sucessfully generate "<< weights_file<<" from "<< model_file;
  return 0;
}
