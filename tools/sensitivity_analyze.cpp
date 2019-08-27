#include <vector>
#include <map>
#include <string>
#include <thread>
#include <chrono>
#include <cstring>
#include <cstdio>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <boost/algorithm/string.hpp>

#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

DEFINE_string(test_server, "", "model test server startup command line.");
DEFINE_string(deploy, "", "deploy config file for calculate operation nums.");
DEFINE_string(model, "", "initial model for sensitivity analyze.");
DEFINE_double(target, 0.0, "key sensitivity metric target.");
DEFINE_double(target_eps, 0.01, "key sensitivity metric target eps.");
DEFINE_double(ratio_eps, 0.001, "pruning ratio eps.");
DEFINE_string(mode, "output", "prune mode, one of output, input, element, map");
DEFINE_string(model_out, "", "output pruned model.");
DEFINE_string(workspace, "", "a folder for temp files.");
DEFINE_string(exclude, "", "exclude layers separated by ','");

using namespace caffe;
using namespace std;

class FileConditionVariable {
public:
  FileConditionVariable(const std::string& file): file_(file) {
  }

  void Wait(void) {
    for(;;) {
      FILE* fp=fopen(file_.c_str(), "rb");
      if(fp!=nullptr) {
        fclose(fp);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        fp=fopen(file_.c_str(), "rb");
        if(fp!=nullptr) {
          fclose(fp);
          system(("rm -f "+file_).c_str());
          break;
        }
      }
      else {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }

  void Signal(void) {
    for(;;) {
      FILE* fp=fopen(file_.c_str(), "wb");
      if(fp!=nullptr) {
        break;
        fclose(fp);
      }
      else {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }

private:
  std::string file_;
};

static inline float Evaluate(const caffe::NetParameter& model, const std::string& model_file,
    FileConditionVariable& model_cv, const std::string& result_file, FileConditionVariable& result_cv) {
  WriteProtoToBinaryFile(model, model_file);
  model_cv.Signal();
  result_cv.Wait();
  FILE* fp=fopen(result_file.c_str(), "r");
  float result;
  CHECK_EQ(fscanf(fp, "%f", &result), 1);
  return result;
}

static vector<BlobProto> Prune(const vector<BlobProto>& in, const string& mode, float ratio) {
  vector<BlobProto> out=in;
  vector<vector<float>> group;
  vector<vector<size_t>> group_blob;
  vector<vector<size_t>> group_index;
  if(mode=="output") {
    for(size_t i=0; i<in.size(); i++) {
      CHECK_EQ(in[i].shape().dim_size(), 4);
      CHECK_EQ(in[i].shape().dim(0), in[0].shape().dim(0));
    }
    size_t group_num=in[0].shape().dim(0);
    group.resize(group_num);
    group_blob.resize(group_num);
    group_index.resize(group_num);
    for(size_t i=0; i<in.size(); i++) {
      size_t stride=in[i].shape().dim(1)*in[i].shape().dim(2)*in[i].shape().dim(3);
      for(size_t j=0; j<in[i].data_size(); j++) {
        size_t group_id=j/stride;
        group[group_id].push_back(in[i].data(j));
        group_blob[group_id].push_back(i);
        group_index[group_id].push_back(j);
      }
    }
  }
  else if(mode=="input") {
    for(size_t i=0; i<in.size(); i++) {
      CHECK_EQ(in[i].shape().dim_size(), 4);
      CHECK_EQ(in[i].shape().dim(1), in[0].shape().dim(1));
    }
    size_t group_num=in[0].shape().dim(1);
    group.resize(group_num);
    group_blob.resize(group_num);
    group_index.resize(group_num);
    for(size_t i=0; i<in.size(); i++) {
      size_t stride=in[i].shape().dim(2)*in[i].shape().dim(3);
      for(size_t j=0; j<in[i].data_size(); j++) {
        size_t group_id=(j/stride)%group_num;
        group[group_id].push_back(in[i].data(j));
        group_blob[group_id].push_back(i);
        group_index[group_id].push_back(j);
      }
    }
  }
  else {
    LOG(FATAL) << "Unknown mode: " << mode;
  }

  vector<pair<float, size_t>> group_eval(group.size());
  for(size_t i=0; i<group_eval.size(); i++) {
    group_eval[i]=make_pair(0.0f, i);
    for(size_t j=0; j<group[i].size(); j++) {
      group_eval[i].first+=group[i][j]*group[i][j];
    }
  }
  std::sort(group_eval.begin(), group_eval.end(),
      [](const pair<float, size_t>& lhs, const pair<float, size_t>& rhs){return lhs.first<rhs.first;});
  for(size_t i=0; i<static_cast<size_t>(ratio*group_eval.size()) && i<group_eval.size(); i++) {
    const vector<size_t>& blob=group_blob[group_eval[i].second];
    const vector<size_t>& index=group_index[group_eval[i].second];
    for(size_t j=0; j<index.size(); j++) {
      out[blob[j]].set_data(index[j], 0.0f);
    }
  }
  return out;
}

static inline bool is_float_same(float lhs, float rhs) {
  return std::abs(lhs-rhs)<0.00001f;
}

template <typename T>
static inline string ToString(const char* format, const T& data) {
  char buf[1024];
  sprintf(buf, format, data);
  return string(buf);
}

static inline std::string Replace(std::string src,
    const std::string& mode, const std::string& re_mode) {
  for(;;) {
    size_t pos=src.find(mode);
    if(pos==std::string::npos) {
      break;
    }
    else {
      src.replace(pos, mode.size(), re_mode);
    }
  }
  return src;
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr=1;
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  CHECK_NE(FLAGS_deploy, "");
  CHECK_NE(FLAGS_model, "");
  CHECK_NE(FLAGS_model_out, "");
  CHECK_NE(FLAGS_mode, "");
  CHECK_GT(FLAGS_target_eps, 0);
  CHECK_GT(FLAGS_ratio_eps, 0);

  set<string> exclude_layers;
  if(FLAGS_exclude!="") {
    vector<string> ex;
    boost::split(ex, FLAGS_exclude, boost::is_any_of(","));
    for(const auto& item: ex) {
      exclude_layers.insert(item);
    }
  }

  const string test_model=FLAGS_workspace+"/model.caffemodel";
  const string test_model_cv=FLAGS_workspace+"/model_cv";
  const string test_result=FLAGS_workspace+"/result.txt";
  const string test_result_cv=FLAGS_workspace+"/result_cv";

  string test_server=FLAGS_test_server;
  if(test_server.find("{model}")==string::npos ||
      test_server.find("{model_cv}")==string::npos ||
      test_server.find("{result}")==string::npos ||
      test_server.find("{result_cv}")==string::npos) {
    LOG(ERROR) << "{model}, {model_cv}, {result}, {result_cv} are needed in test server startup command line.";
    LOG(ERROR) << "{model}: path to model to test";
    LOG(ERROR) << "{model_cv}: path to model file's ready signal (file condition variable)";
    LOG(ERROR) << "{result}: path to test result";
    LOG(ERROR) << "{result_cv}: path to test result's ready signal (file condition variable)";
    LOG(FATAL) << "Bad test server startup command: " << test_server;
  }
  test_server=Replace(test_server, "{model}", test_model);
  test_server=Replace(test_server, "{model_cv}", test_model_cv);
  test_server=Replace(test_server, "{result}", test_result);
  test_server=Replace(test_server, "{result_cv}", test_result_cv);
  LOG(INFO) << "Starting test server: " << test_server;
  vector<char> test_server_c_str(test_server.size()+1);
  strcpy(test_server_c_str.data(), test_server.c_str());
  std::thread server_thread(system, test_server_c_str.data());

  FileConditionVariable model_cv(test_model_cv);
  FileConditionVariable result_cv(test_result_cv);

#ifndef CPU_ONLY
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);
#else
  Caffe::set_mode(Caffe::CPU);
#endif

  NetParameter model;
  ReadNetParamsFromBinaryFileOrDie(FLAGS_model, &model);

  map<string, set<string>> father_layer_by_blob;
  for(size_t i=0; i<model.layer_size(); i++) {
    const LayerParameter& layer=model.layer(i);
    if(layer.type()=="Convolution" || layer.type()=="InnerProduct" || layer.type()=="Input") {
      for(size_t i=0; i<layer.top_size(); i++) {
        father_layer_by_blob[layer.top(i)]={layer.name()};
      }
    }
    else if(layer.type()=="CuDNNBatchNorm" || layer.type()=="ReLU" ||
        layer.type()=="Sigmoid" || layer.type()=="TanH" || layer.type()=="Pooling" ||
        layer.type()=="MBatchNorm" || layer.type()=="BatchNorm" || layer.type()=="Gather") {
      CHECK_EQ(layer.bottom_size(), 1);
      CHECK_EQ(layer.top_size(), 1);
      if(father_layer_by_blob.find(layer.bottom(0))!=father_layer_by_blob.end()) {
        father_layer_by_blob[layer.top(0)]=father_layer_by_blob[layer.bottom(0)];
      }
    }
    else if(layer.type()=="Split") {
      if(father_layer_by_blob.find(layer.bottom(0))!=father_layer_by_blob.end()) {
        const set<string>& bottom=father_layer_by_blob[layer.bottom(0)];
        for(size_t i=0; i<layer.top_size(); i++) {
          father_layer_by_blob[layer.top(i)]=bottom;
        }
      }
    }
    else if(layer.type()=="Eltwise") {
      set<string>& top=father_layer_by_blob[layer.top(0)];
      for(size_t i=0; i<layer.bottom_size(); i++) {
        CHECK(father_layer_by_blob.find(layer.bottom(i))!=father_layer_by_blob.end());
        const set<string>& bottom=father_layer_by_blob[layer.bottom(i)];
        top.insert(bottom.begin(), bottom.end());
      }
    }
    else {
      LOG(INFO) << "Warning: unknown layer type when analyze dependency: " << layer.type() << " of " << layer.name();
    }
  }

  set<set<size_t>> prune_set;
  if(FLAGS_mode=="input") {
    vector<set<size_t>> prune_set_v;
    vector<set<string>> affect_convs_to_set;
    for(size_t i=0; i<model.layer_size(); i++) {
      if(model.layer(i).type()!="Convolution") {
        continue;
      }
      set<string> affect_convs_to_layer;
      for(size_t j=0; j<model.layer(i).bottom_size(); j++) {
        CHECK(father_layer_by_blob.find(model.layer(i).bottom(j))!=father_layer_by_blob.end());
        const set<string>& convs=father_layer_by_blob[model.layer(i).bottom(j)];
        affect_convs_to_layer.insert(convs.begin(), convs.end());
      }
      bool inserted=false;
      for(size_t j=0; j<affect_convs_to_set.size(); j++) {
        bool exist=false;
        for(const auto& item: affect_convs_to_set[j]) {
          if(affect_convs_to_layer.find(item)!=affect_convs_to_layer.end()) {
            exist=true;
            break;
          }
        }
        if(exist) {
          prune_set_v[j].insert(i);
          affect_convs_to_set[j].insert(affect_convs_to_layer.begin(), affect_convs_to_layer.end());
          inserted=true;
          break;
        }
      }
      if(!inserted) {
        prune_set_v.push_back({i});
        affect_convs_to_set.push_back(affect_convs_to_layer);
      }
    }
    for(const auto& item: prune_set_v) {
      prune_set.insert(item);
    }
  }
  else {
    LOG(FATAL) << "Unknown mode: " << FLAGS_mode;
  }

  for(auto it=prune_set.begin(); it!=prune_set.end();) {
    bool exist=false;
    for(const auto& item: *it) {
      if(exclude_layers.find(model.layer(item).name())!=exclude_layers.end()) {
        exist=true;
        break;
      }
    }
    if(exist) {
      it=prune_set.erase(it);
    }
    else {
      it++;
    }
  }

  map<set<size_t>, string> prune_set_name;
  for(const auto& item: prune_set) {
    string name;
    for(const auto& id: item) {
      name+=model.layer(id).name();
      name+=" & ";
    }
    name=name.substr(0, name.size()-3);
    prune_set_name[item]=name;
    LOG(INFO) << "prune set: " << name;
  }

  float full_metric=Evaluate(model, test_model, model_cv, test_result, result_cv);
  LOG(INFO) << "baseline metric: " << full_metric;

  set<size_t> prune_layers;
  for(const auto& item: prune_set) {
    prune_layers.insert(item.begin(), item.end());
  }

  float target_metric_step=(FLAGS_target-full_metric)/prune_layers.size();
  float target_metric=full_metric;
  float current_metric=full_metric;
  for(auto it=prune_set.begin(); it!=prune_set.end(); it++) {
    vector<BlobProto> raw_blobs;
    vector<size_t> layer_id;
    for(const auto& id: *it) {
      CHECK_GE(model.layer(id).blobs_size(), 1);
      raw_blobs.push_back(model.layer(id).blobs(0));
      layer_id.push_back(id);
    }

    NetParameter model_prune=model;

    float lower_ratio=0.0f;
    float upper_ratio=1.0f;
    float lower_metric=current_metric;
    vector<BlobProto> pruned_blobs=Prune(raw_blobs, FLAGS_mode, upper_ratio);
    for(size_t i=0; i<layer_id.size(); i++) {
      *(model_prune.mutable_layer(layer_id[i])->mutable_blobs(0))=pruned_blobs[i];
    }
    float upper_metric=Evaluate(model_prune, test_model, model_cv, test_result, result_cv);
    target_metric+=target_metric_step*it->size();
    LOG(INFO) << "layer: " << prune_set_name[*it] << ToString("   prune: %4.3f", upper_ratio)
        << ToString("   metric: %.6f", upper_metric);
    if((upper_metric-target_metric)*(lower_metric-target_metric)<0) {
      while(std::abs(upper_metric-lower_metric)>FLAGS_target_eps && std::abs(upper_ratio-lower_ratio)>FLAGS_ratio_eps) {
        float ratio=(upper_ratio+lower_ratio)/2.0f;
        vector<BlobProto> pruned_blobs=Prune(raw_blobs, FLAGS_mode, ratio);
        for(size_t i=0; i<layer_id.size(); i++) {
          *(model_prune.mutable_layer(layer_id[i])->mutable_blobs(0))=pruned_blobs[i];
        }
        float metric=Evaluate(model_prune, test_model, model_cv, test_result, result_cv);
        LOG(INFO) << "layer: " << prune_set_name[*it] << ToString("   prune: %4.3f", ratio)
            << ToString("   metric: %.6f", metric);
        if((metric-target_metric)*(lower_metric-target_metric)<0) {
          upper_ratio=ratio;
          upper_metric=metric;
        }
        else if((metric-target_metric)*(upper_metric-target_metric)<0) {
          lower_ratio=ratio;
          lower_metric=metric;
        }
        else {
          break;
        }
      }
    }

    float ratio=(upper_metric-target_metric)*(full_metric-target_metric)>=0?upper_ratio:lower_ratio;
    pruned_blobs=Prune(raw_blobs, FLAGS_mode, ratio);
    for(size_t i=0; i<layer_id.size(); i++) {
      *(model.mutable_layer(layer_id[i])->mutable_blobs(0))=pruned_blobs[i];
    }
    current_metric=(upper_metric-target_metric)*(full_metric-target_metric)>=0?upper_metric:lower_metric;
    LOG(INFO) << "layer: " << prune_set_name[*it] << ToString("   select prune: %4.3f", ratio)
        << ToString("   metric: %.6f", current_metric);
  }

  LOG(INFO) << "pruned metric: " << current_metric;
  WriteProtoToBinaryFile(model, FLAGS_model_out);
  return 0;
}
