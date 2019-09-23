#include <glog/logging.h>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <cmath>

#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

using std::string;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;

int main(int argc, char** argv){
  //ReadNetParamsFromTextFileOrDie(FLAGS_model, &param);
  if (argc != 4){
    LOG(INFO) << "usage: " << argv[0] << " model clip_file out_mask_file";
    return -1;
  }
  
  FLAGS_alsologtostderr = 1;
  
  const string model(argv[1]); 
  const string clip_file(argv[2]); 
  const string out_file(argv[3]); 
  std::ifstream fin(clip_file.c_str());
  string name;
  float ratio;
  std::vector<std::pair<string, float> > lines;
  int line_length = 0;
  while(fin >> name >> ratio) {
    lines.push_back(std::make_pair(name, ratio));
    ++line_length;
  }
  
  caffe::NetParameter net_param; 
  ReadNetParamsFromBinaryFileOrDie(model, &net_param);
  LOG(INFO) << "Finish read params.";
  
  const int layer_size = net_param.layer_size();
  LOG(INFO) << "Total layers: " << layer_size;

  std::ofstream fout(out_file.c_str());
  for (int i = 0; i < layer_size; ++i){
     caffe::LayerParameter  *current_param = net_param.mutable_layer(i);
     for (int line_id = 0; line_id < line_length; ++line_id) {
       if (current_param->name() == lines[line_id].first) {
         LOG(INFO) << lines[line_id].first;
         fout << lines[line_id].first << " ";
         const int blob_size = current_param->blobs_size();
         CHECK_GT(blob_size, 0);
         caffe::BlobProto *blob = current_param->mutable_blobs(0);
         int data_size = blob->data_size();
         fout << data_size << " ";
         float *data = blob->mutable_data()->mutable_data();
         std::vector<float> copy_data(data_size);
         caffe::caffe_abs(data_size, data, copy_data.data());
         std::sort(copy_data.begin(), copy_data.end());
         // float threshold = copy_data[data_size * (1 - lines[line_id].second)];
         float threshold = copy_data[data_size * lines[line_id].second];
         for (int j = 0; j < data_size; ++j) {
           if (fabs(data[j]) >= threshold) fout << "1 ";
           else fout << "0 ";
         }
         fout << "\n";
       }
     }
  }
  fout.close();
  return 0;
}
