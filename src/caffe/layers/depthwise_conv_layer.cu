#include <vector>
#include <algorithm>
#include <cfloat>
#include "caffe/layers/depthwise_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype, int kSize>
__global__ void ConvForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width,const int conved_height,
    const int conved_width,const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {
  const int KW_LIMIT = (kSize !=0) ? kSize : kernel_w;
  const int KH_LIMIT = (kSize !=0) ? kSize : kernel_h;
  CUDA_KERNEL_LOOP(index, nthreads) {

    // avoid / and % operation, will improve ~10%
    int tmp1 = index / conved_width;
    const int pw = index - tmp1 * conved_width;
    int tmp2 = tmp1 / conved_height;
    const int ph = tmp1 - tmp2 * conved_height;
    tmp1 = tmp2;
    tmp2 = tmp1 / channels;
    const int c = tmp1 - tmp2 * channels;
    const int n = tmp2;

    Dtype aveval = 0;
    const Dtype* const bottom_slice =
    bottom_data + (n * channels + c) * height * width;
    const Dtype* const weight_slice =
    weight + c * kernel_h * kernel_w;

    // use KH_LIMIT and KW_LIMIT, make use of template code, will improve ~20%
    int w_idx = 0;
    #pragma unroll
    for (int kh = 0; kh < KH_LIMIT; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < KW_LIMIT; ++kw) {
        const int h_in = ph * stride_h - pad_h + kh;
        const int w_in = pw * stride_w - pad_w + kw;
        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
          aveval += bottom_slice[h_in * width + w_in] * weight_slice[w_idx];
        }
        ++w_idx;
      }
    }
    if(bias_term_) {
      aveval+=bias[c];
    }
    top_data[index] = aveval;
  }
}

template<typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.has_quantize_param()) {
    // use top data to store temporary buffer.
    this->QuantizeWeights_gpu(top[0]->mutable_gpu_data());
  }
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  int* stride_data = this->stride_.mutable_cpu_data();
  int* pad_data = this->pad_.mutable_cpu_data();
  if (this->fp16_setup_) {
    caffe_float2half(this->blobs_[0]->count(), weight, this->weight_buffer_fp16_);
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->gpu_data();
      caffe_float2half(this->blobs_[1]->count(), bias, this->bias_buffer_fp16_);
    }
  }

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int count = top[i]->count();
    vector<int> shape_ = bottom[i]->shape();
    const int channels_ = shape_[1];
    const int height_ = shape_[2];
    const int width_ = shape_[3];

    const int kernel_h_ = kernel_shape_data[0];
    const int kernel_w_ = kernel_shape_data[1];
    const int stride_h_ = stride_data[0];
    const int stride_w_ = stride_data[1];
    const int pad_h_ = pad_data[0];
    const int pad_w_ = pad_data[1];

    const int conved_height = this->output_shape_[0];
    const int conved_weight = this->output_shape_[1];

    const bool bias_term_ = this->bias_term_;
    const Dtype* const bias = bias_term_ ? this->blobs_[1]->gpu_data() : 0;

    if (kernel_h_ == 3 && kernel_w_ == 3) {
      ConvForward<Dtype, 3><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[i]->num(), channels_,
        height_, width_,conved_height,conved_weight,kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
    } else if (kernel_h_ == 5 && kernel_w_ == 5) {
      ConvForward<Dtype, 5><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[i]->num(), channels_,
        height_, width_,conved_height,conved_weight,kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
    } else if (kernel_h_ == 1 && kernel_w_ == 1) {
      ConvForward<Dtype, 1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[i]->num(), channels_,
        height_, width_,conved_height,conved_weight,kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
    } else {
      ConvForward<Dtype, 0><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[i]->num(), channels_,
        height_, width_,conved_height,conved_weight,kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
    }
  }
}


template <typename Dtype, int kSize, int kStride>
__global__ void ConvBackward(const int nthreads,
    const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int conved_height, const int conved_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff,
    const Dtype* const weight) {

  const int KW_LIMIT = (kSize !=0) ? kSize : kernel_w;
  const int KH_LIMIT = (kSize !=0) ? kSize : kernel_h;
  const int STRIDE_H = (kStride != 0) ? kStride : stride_h;
  const int STRIDE_W = (kStride != 0) ? kStride : stride_w;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int tmp1 = index / width;
    const int w = index - tmp1 * width;
    int tmp2 = tmp1 / height;
    const int h = tmp1 - tmp2 * height;
    tmp1 = tmp2;
    tmp2 = tmp1 / channels;
    const int c = tmp1 - tmp2 * channels;
    const int n = tmp2;
    
    Dtype grad = 0.0;

    int w_idx = c * KH_LIMIT * KW_LIMIT;
      #pragma unroll
      for (int kh = 0; kh < KH_LIMIT; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < KW_LIMIT; ++kw) {
          int h_out = h + pad_h - kh;
          int w_out = w + pad_w - kw;
          if ((h_out % STRIDE_H == 0) && (w_out % STRIDE_W == 0)) {
            h_out = h_out / STRIDE_H;
            w_out = w_out / STRIDE_W;

            if ((h_out >= 0) && (h_out < conved_height)
                  && (w_out >= 0) && (w_out < conved_width)) {

              const int offset = ((n * channels + c) * conved_height + h_out)
                    * conved_width + w_out;
              grad += weight[w_idx] * top_diff[offset];
            }
          }
          ++ w_idx;
        }
      }

    bottom_diff[index] = grad;
  }
}

__device__ float atomicAddme(float* address, float val)
{
    return atomicAdd(address,val);
}

__device__ double atomicAddme(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}



#define DIVIDE_CEIL(a,b) a/b+((a/b*b)<a)


template <typename Dtype, int kSize, int kStride>
__global__ void ConvBackwardWeight(const int nthreads,
    const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int conved_height, const int conved_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const weight_diff,
    const Dtype* const bottom_data) {

  const int KH_LIMIT = kSize > 0 ? kSize : kernel_h;
  const int KW_LIMIT = kSize > 0 ? kSize : kernel_w;
  const int STRIDE_H = kStride > 0 ? kStride : stride_h;
  const int STRIDE_W = kStride > 0 ? kStride : stride_w;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int tmp1 = index / KW_LIMIT;
    const int kw = index - tmp1 * KW_LIMIT;            // kw=index % kernel_w
    int tmp2 = tmp1 / KH_LIMIT;
    const int kh = tmp1 - tmp2 * KH_LIMIT;  // kh = (index / kernel_w) % kernel_h
    tmp1 = tmp2;
    tmp2 = tmp1 / channels;
    const int c = tmp1 - tmp2 * channels;   // c = (index / kernel_w / kernel_h) % channels
    const int n = tmp2;                     // n = index / kernel_w / kernel_h / channels

    Dtype gradient = 0;
      
    const Dtype* const top_diff_slice = top_diff + (n * channels + c) * conved_height * conved_width;
    const Dtype* const bottom_data_slice = bottom_data + (n * channels + c) * height * width;
    
    const int phstart=max(DIVIDE_CEIL((pad_h-kh),STRIDE_H),0);
    const int phend=min(DIVIDE_CEIL((height+pad_h-kh),STRIDE_H),conved_height);
    
    const int pwstart=max(DIVIDE_CEIL((pad_w-kw),STRIDE_W),0);
      
    const int pwend=min(DIVIDE_CEIL((width+pad_w-kw),STRIDE_W),conved_width);

    // do not use unroll here.
    // 869ms -> 894ms in benchmark env.
    for(int ph=phstart;ph<phend;ph++){
      for (int pw=pwstart;pw<pwend;pw++){
        const int h=ph*STRIDE_H+kh-pad_h;
        const int w=pw*STRIDE_W+kw-pad_w;
        gradient+=top_diff_slice[ph * conved_width + pw]*bottom_data_slice[h*width+w];

      }
    }
    atomicAddme(weight_diff + c * KH_LIMIT * KW_LIMIT+kh*KW_LIMIT+kw, gradient);
  }
}

template <typename Dtype>
__global__ void ConvBackwardBias(const int nthreads,
    const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int conved_height, const int conved_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bias_diff) {

  // index = N*C*H
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int tmp1 = index / conved_height;
    const int h = index - tmp1 * conved_height;  // h = index % H
    const int n = tmp1 / channels;        // n = index / H / C
    const int c = tmp1 - n*channels;  // c = (index / H) % C

    const Dtype* const top_diff_slice = top_diff + ((n*channels+c)*conved_height+h)*conved_width;
    Dtype grad = 0;
    #pragma unroll
    for (int w = 0; w < conved_width; ++w) {
      grad += top_diff_slice[w];
    }
    atomicAddme(bias_diff+c, grad);
  }
}


template<typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Backward_gpu(
const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
const vector<Blob<Dtype>*>& bottom) {


  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  int* stride_data = this->stride_.mutable_cpu_data();
  int* pad_data = this->pad_.mutable_cpu_data();

  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

  const bool bias_term_ = this->bias_term_;
  Dtype* bias_diff = bias_term_ ? this->blobs_[1]->mutable_gpu_diff() : 0;
  const bool bias_propagate_down_ = this->param_propagate_down_[1];
  const bool weight_propagate_down_ = this->param_propagate_down_[0];


  const int kernel_h_ = kernel_shape_data[0];
  const int kernel_w_ = kernel_shape_data[1];
  const int stride_h_ = stride_data[0];
  const int stride_w_ = stride_data[1];
  const int pad_h_ = pad_data[0];
  const int pad_w_ = pad_data[1];

  const int conved_height = this->output_shape_[0];
  const int conved_weight = this->output_shape_[1];

//  CHECK_EQ(stride_h_, 1)
//          << "The backward of the net whose stride is bigger than 1 is not implemented now. ";
//  CHECK_EQ(stride_w_, 1)
//          << "The backward of the net whose stride is bigger than 1 is not implemented now. ";


  for (int i = 0; i < top.size(); ++i) {

    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

    vector<int> shape_ = bottom[i]->shape();
    const int batch_ = shape_[0];
    const int channels_ = shape_[1];
    const int height_ = shape_[2];
    const int width_ = shape_[3];

    // Bias gradient, if necessary.
    if (bias_term_ && bias_propagate_down_) {
      const int nthreads = batch_ * channels_ * height_;
      ConvBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top_diff, bottom[i]->num(), channels_,
        height_, width_,conved_height,conved_weight,kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bias_diff);
    }
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    if (weight_propagate_down_) {
      const int nthreads = batch_ * channels_ * kernel_h_ * kernel_w_;

      // template expand with stride = 1
      if (stride_h_ == stride_w_ && stride_h_ == 1) {
        if (kernel_h_ == kernel_w_ && kernel_h_ == 3) {
          ConvBackwardWeight<Dtype, 3, 1><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, weight_diff,
            bottom_data);
        } else if (kernel_h_ == kernel_w_ && kernel_h_ == 5) {
          ConvBackwardWeight<Dtype, 5, 1><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, weight_diff,
            bottom_data);
        } else if (kernel_h_ == kernel_w_ && kernel_h_ == 1) {
          ConvBackwardWeight<Dtype, 1, 1><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, weight_diff,
            bottom_data);
        } else {
          // no match template
          ConvBackwardWeight<Dtype, 0, 1><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, weight_diff,
            bottom_data);
        }
      } else {
        // no match stride
        if (kernel_h_ == kernel_w_ && kernel_h_ == 3) {
          ConvBackwardWeight<Dtype, 3, 0><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, weight_diff,
            bottom_data);
        } else if (kernel_h_ == kernel_w_ && kernel_h_ == 5) {
          ConvBackwardWeight<Dtype, 5, 0><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, weight_diff,
            bottom_data);
        } else if (kernel_h_ == kernel_w_ && kernel_h_ == 1) {
          ConvBackwardWeight<Dtype, 1, 0><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, weight_diff,
            bottom_data);
        } else {
          // both kernel & stride not match
          ConvBackwardWeight<Dtype, 0, 0><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, weight_diff,
            bottom_data);
        }
      }
    } // if (weight_propagate_down_)

    // gradient w.r.t. bottom data, if necessary.
    if (propagate_down[i]) {
      const int count_bottom = bottom[i]->count();
      if (stride_h_ == stride_w_ && stride_h_ == 1) {
        if (kernel_h_ == kernel_w_ && kernel_h_ == 3) {
          ConvBackward<Dtype, 3, 1><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
            count_bottom, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
            bottom_diff, weight);
        } else if (kernel_h_ == kernel_w_ && kernel_h_ == 5) {
          ConvBackward<Dtype, 5, 1><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
            count_bottom, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
            bottom_diff, weight);
        } else if (kernel_h_ == kernel_w_ && kernel_h_ == 1) {
          ConvBackward<Dtype, 1, 1><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
            count_bottom, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
            bottom_diff, weight);
        } else {
          ConvBackward<Dtype, 0, 1><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
            count_bottom, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
            bottom_diff, weight);
        }
      } else {
        if (kernel_h_ == kernel_w_ && kernel_h_ == 3) {
          ConvBackward<Dtype, 3, 0><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
            count_bottom, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
            bottom_diff, weight);
        } else if (kernel_h_ == kernel_w_ && kernel_h_ == 5) {
          ConvBackward<Dtype, 5, 0><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
            count_bottom, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
            bottom_diff, weight);
        } else if (kernel_h_ == kernel_w_ && kernel_h_ == 1) {
          ConvBackward<Dtype, 1, 0><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
            count_bottom, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
            bottom_diff, weight);
        } else {
          ConvBackward<Dtype, 0, 0><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
            count_bottom, top_diff, bottom[i]->num(), channels_,
            height_, width_, conved_height, conved_weight, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
            bottom_diff, weight);
        }
      }
    } // if (propagate_down[i])
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(DepthwiseConvolutionLayer);

}  // namespace caffe
