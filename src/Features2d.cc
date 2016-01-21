#include "Features2d.h"
#include "Matrix.h"
#include <nan.h>
#include <stdio.h>

#if ((CV_MAJOR_VERSION >= 2) && (CV_MINOR_VERSION >=4))

void Features::Init(Local<Object> target) {
  Nan::HandleScope scope;

  Nan::SetMethod(target, "ImageSimilarity", Similarity);
  Nan::SetMethod(target, "DetectAndCompute", DetectAndCompute);
  Nan::SetMethod(target, "FilteredMatch", FilteredMatch);
}

class AsyncDetectSimilarity: public Nan::AsyncWorker {
public:
  AsyncDetectSimilarity(Nan::Callback *callback, cv::Mat image1, cv::Mat image2) :
      Nan::AsyncWorker(callback),
      image1(image1),
      image2(image2),
      dissimilarity(0) {
  }

  ~AsyncDetectSimilarity() {
  }

  void Execute() {

    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("ORB");
    cv::Ptr<cv::DescriptorExtractor> extractor =
        cv::DescriptorExtractor::create("ORB");
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(
        "BruteForce-Hamming");

    std::vector<cv::DMatch> matches;

    cv::Mat descriptors1 = cv::Mat();
    cv::Mat descriptors2 = cv::Mat();

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;

    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    extractor->compute(image1, keypoints1, descriptors1);
    extractor->compute(image2, keypoints2, descriptors2);

    matcher->match(descriptors1, descriptors2, matches);

    double max_dist = 0;
    double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors1.rows; i++) {
      double dist = matches[i].distance;
      if (dist < min_dist) {
        min_dist = dist;
      }
      if (dist > max_dist) {
        max_dist = dist;
      }
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector<cv::DMatch> good_matches;
    double good_matches_sum = 0.0;

    for (int i = 0; i < descriptors1.rows; i++) {
      double distance = matches[i].distance;
      if (distance <= std::max(2 * min_dist, 0.02)) {
        good_matches.push_back(matches[i]);
        good_matches_sum += distance;
      }
    }

    dissimilarity = (double) good_matches_sum / (double) good_matches.size();
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;

    Local<Value> argv[2];

    argv[0] = Nan::Null();
    argv[1] = Nan::New<Number>(dissimilarity);

    callback->Call(2, argv);
  }

private:
  cv::Mat image1;
  cv::Mat image2;
  double dissimilarity;
};

NAN_METHOD(Features::Similarity) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(2, cb);

  cv::Mat image1 = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat;
  cv::Mat image2 = Nan::ObjectWrap::Unwrap<Matrix>(info[1]->ToObject())->mat;

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncDetectSimilarity(callback, image1, image2) );
  return;
}



class AsyncDetectAndCompute: public Nan::AsyncWorker {
public:
  AsyncDetectAndCompute(Nan::Callback *callback, cv::Mat image) :
      Nan::AsyncWorker(callback),
      image(image) {
  }

  ~AsyncDetectAndCompute() {
  }

  void Execute() {

    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("ORB");
    cv::Ptr<cv::DescriptorExtractor> extractor =
        cv::DescriptorExtractor::create("ORB");

    descriptors = cv::Mat();

    detector->detect(image, keypoints);

    extractor->compute(image, keypoints, descriptors);
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;

    Local<Value> argv[2];
    v8:Local<Object> result = Nan::New<Object>();

    argv[0] = Nan::Null();
    argv[1] = result;

    int size = keypoints.size();

    Local<Object> key;

    Local<Array> keys = Nan::New<Array>(size); 
    for (int i = 0; i < size; i++) { 
      key = Nan::New<Object>();
      key->Set(Nan::New<String>("pointx").ToLocalChecked(), Nan::New<Number>(keypoints[i].pt.x));
      key->Set(Nan::New<String>("pointy").ToLocalChecked(), Nan::New<Number>(keypoints[i].pt.y));
      key->Set(Nan::New<String>("size").ToLocalChecked(), Nan::New<Number>(keypoints[i].size));
      key->Set(Nan::New<String>("angle").ToLocalChecked(), Nan::New<Number>(keypoints[i].angle));
      key->Set(Nan::New<String>("response").ToLocalChecked(), Nan::New<Number>(keypoints[i].response));
      key->Set(Nan::New<String>("octave").ToLocalChecked(), Nan::New<Number>(keypoints[i].octave));
      key->Set(Nan::New<String>("class_id").ToLocalChecked(), Nan::New<Number>(keypoints[i].class_id));

      keys->Set(Nan::New<Number>(i), key); 
    }

    result->Set(Nan::New<String>("keypoints").ToLocalChecked(), keys);
    Local<Object> im_h = Nan::New(Matrix::constructor)->GetFunction()->NewInstance();
    Matrix *img = Nan::ObjectWrap::Unwrap<Matrix>(im_h);
    img->mat = descriptors;

    result->Set(Nan::New<String>("descriptors").ToLocalChecked(), im_h);

    callback->Call(2, argv);
  }

private:
  cv::Mat image;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

NAN_METHOD(Features::DetectAndCompute) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(1, cb);

  cv::Mat image = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat;

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncDetectAndCompute(callback, image) );
  return;
}


class AsyncFilteredMatch: public Nan::AsyncWorker {
public:
  AsyncFilteredMatch(Nan::Callback *callback,  std::vector<cv::KeyPoint> keypoints1, cv::Mat descriptors1,  std::vector<cv::KeyPoint> keypoints2, cv::Mat descriptors2) :
      Nan::AsyncWorker(callback),
      keypoints1(keypoints1),
      descriptors1(descriptors1),
      keypoints2(keypoints2),
      descriptors2(descriptors2),
      dissimilarity(0) {
  }

  ~AsyncFilteredMatch() {
  }

  void Execute() {

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(
        "BruteForce-Hamming");

    std::vector<cv::DMatch> matches;

    matcher->match(descriptors1, descriptors2, matches);

    double max_dist = 0;
    double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors1.rows; i++) {
      double dist = matches[i].distance;
      if (dist < min_dist) {
        min_dist = dist;
      }
      if (dist > max_dist) {
        max_dist = dist;
      }
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector<cv::DMatch> good_matches;
    double good_matches_sum = 0.0;

    for (int i = 0; i < descriptors1.rows; i++) {
      double distance = matches[i].distance;
      if (distance <= std::max(2 * min_dist, 0.02)) {
        good_matches.push_back(matches[i]);
        good_matches_sum += distance;
      }
    }

    dissimilarity = (double) good_matches_sum / (double) good_matches.size();
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;

    Local<Value> argv[2];

    argv[0] = Nan::Null();
    argv[1] = Nan::New<Number>(dissimilarity);

    callback->Call(2, argv);
  }

private:
  std::vector<cv::KeyPoint> keypoints1;
  cv::Mat descriptors1;
  std::vector<cv::KeyPoint> keypoints2;
  cv::Mat descriptors2;
  double dissimilarity;
};

NAN_METHOD(Features::FilteredMatch) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(2, cb);

  std::vector<cv::KeyPoint> keypoints1;
  std::vector<cv::KeyPoint> keypoints2;
  Local<Object> features1 = info[0]->ToObject();
  Local<Object> features2 = info[1]->ToObject();

  int size;
  Local<Object> key;
  Local<Array> keys1 = Nan::Get(features1, Nan::New<String>("keypoints").ToLocalChecked()).ToLocalChecked()->ToObject().As<Array>();
  Local<Array> keys2 = Nan::Get(features2, Nan::New<String>("keypoints").ToLocalChecked()).ToLocalChecked()->ToObject().As<Array>();;

  size = Nan::Get(keys1, Nan::New<String>("length").ToLocalChecked()).ToLocalChecked()->Uint32Value();
  for(int i = 0; i < size; i++) {
    key =  Nan::Get(keys1, Nan::New<Number>(i)).ToLocalChecked().As<Object>();
    keypoints1.push_back(cv::KeyPoint(
      Nan::Get(key, Nan::New<String>("pointx").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("pointy").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("size").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("angle").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("response").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("octave").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("class_id").ToLocalChecked()).ToLocalChecked()->Uint32Value()
      ));  
  }
  size = Nan::Get(keys1, Nan::New<String>("length").ToLocalChecked()).ToLocalChecked()->Uint32Value();
  for(int i = 0; i < size; i++) {
    key =  Nan::Get(keys1, Nan::New<Number>(i)).ToLocalChecked().As<Object>();
    keypoints1.push_back(cv::KeyPoint(
      Nan::Get(key, Nan::New<String>("pointx").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("pointy").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("size").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("angle").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("response").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("octave").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<String>("class_id").ToLocalChecked()).ToLocalChecked()->Uint32Value()
      ));  
  }
  Local<String> name =  Nan::New<String>("descriptors").ToLocalChecked();
  Local<Object> des = Nan::Get(features1,name).ToLocalChecked().As<Object>();
  cv::Mat descriptors1 = Nan::ObjectWrap::Unwrap<Matrix>(des)->mat;
  des = Nan::Get(features2,name).ToLocalChecked().As<Object>();
  cv::Mat descriptors2 = Nan::ObjectWrap::Unwrap<Matrix>(des)->mat;


  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncFilteredMatch(callback, keypoints1, descriptors1, keypoints2, descriptors2));
  return;
}



#endif
