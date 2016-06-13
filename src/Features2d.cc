#include "Features2d.h"
#include "Matrix.h"
#include <nan.h>
#include <stdio.h>
#include <cmath>

#if ((CV_MAJOR_VERSION >= 2) && (CV_MINOR_VERSION >= 4))

using namespace cv;


void Features::Init(Local<Object> target) {
  Nan::HandleScope scope;

  Nan::SetMethod(target, "ImageSimilarity", Similarity);
  Nan::SetMethod(target, "DetectFeatures", DetectFeatures);
  Nan::SetMethod(target, "Match", Match);
  Nan::SetMethod(target, "DrawFeatures", DrawFeatures);
  Nan::SetMethod(target, "DrawMatches", DrawMatches);
}

class AsyncDetectSimilarity: public Nan::AsyncWorker {
public:
  AsyncDetectSimilarity(Nan::Callback *callback, Mat image1, Mat image2) :
      Nan::AsyncWorker(callback),
      image1(image1),
      image2(image2) {
  }

  ~AsyncDetectSimilarity() {
  }

  void Execute() {

    Mat blur1;
    Mat gray1;
    
    Mat blur;
    Mat gray;
 
    Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");
    Ptr<DescriptorExtractor> extractor =
        DescriptorExtractor::create("ORB");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(
        "BruteForce-Hamming");

    Mat descriptors1 = Mat();
    Mat descriptors2 = Mat();

    std::vector<KeyPoint> keypoints1;
    std::vector<KeyPoint> keypoints2;
    
    std::vector<DMatch> matches12;
    std::vector<DMatch> matches21;
    std::vector<DMatch> matches;
    
    bilateralFilter(image1, blur1, 9, 75, 75);
    cvtColor( blur1, gray1, CV_BGR2GRAY );
    bilateralFilter(image2, blur, 9, 75, 75);
    cvtColor( blur, gray, CV_BGR2GRAY );

    detector->detect(gray1, keypoints1);
    detector->detect(gray, keypoints2);

    extractor->compute(gray1, keypoints1, descriptors1);
    extractor->compute(gray, keypoints2, descriptors2);

    matcher->match(descriptors1, descriptors2, matches12);
    matcher->match(descriptors2, descriptors1, matches21);
    
    // cross check
    for( size_t i = 0; i < matches12.size(); i++ )
    { 
        DMatch forward = matches12[i]; 
        DMatch backward = matches21[forward.trainIdx];
        if( backward.trainIdx == forward.queryIdx ) 
            matches.push_back( forward ); 
    }   
    
    for( size_t i = 0; i < matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      objIndex.push_back(matches[i].queryIdx);
      sceneIndex.push_back(matches[i].trainIdx);
      obj.push_back( keypoints1[ matches[i].queryIdx ].pt );
      scene.push_back( keypoints2[ matches[i].trainIdx ].pt );
      distance.push_back(matches[i].distance);   
    }
    
    drawMatches(image1, keypoints1, image2, keypoints2, matches, img_matches);
    
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;

    Local<Value> argv[3];

    argv[0] = Nan::Null();

    Local<Object> im_h = Nan::New(Matrix::constructor)->GetFunction()->NewInstance();
    Matrix *img = Nan::ObjectWrap::Unwrap<Matrix>(im_h);
    img->mat = img_matches;

    Local<Object> match;
    Local<Object> point1;
    Local<Object> point2;

    Local<Array> matches = Nan::New<Array>(obj.size()); 
    for (size_t i = 0; i < obj.size(); i++) { 
      match = Nan::New<Object>();
      point1 = Nan::New<Object>();
      point2 = Nan::New<Object>();

      match->Set(Nan::New<v8::String>("point1").ToLocalChecked(), point1);
      match->Set(Nan::New<v8::String>("point2").ToLocalChecked(), point2);

      point1->Set(Nan::New<v8::String>("x").ToLocalChecked(), Nan::New<Number>(obj[i].x));
      point1->Set(Nan::New<v8::String>("y").ToLocalChecked(), Nan::New<Number>(obj[i].y));

      point2->Set(Nan::New<v8::String>("x").ToLocalChecked(), Nan::New<Number>(scene[i].x));
      point2->Set(Nan::New<v8::String>("y").ToLocalChecked(), Nan::New<Number>(scene[i].y));

      match->Set(Nan::New<v8::String>("d").ToLocalChecked(), Nan::New<Number>(distance[i]));
      match->Set(Nan::New<v8::String>("q").ToLocalChecked(), Nan::New<Number>(objIndex[i]));
      match->Set(Nan::New<v8::String>("t").ToLocalChecked(), Nan::New<Number>(sceneIndex[i]));

      matches->Set(Nan::New<Number>(i), match); 
    }
     
    argv[1] = im_h;
    argv[2] = matches;

    callback->Call(3, argv);
  }

private:
  Mat image1;
  Mat image2;
  Mat img_matches;

  std::vector<Point2f> obj;
  std::vector<Point2f> scene;
  std::vector<int> objIndex;
  std::vector<int> sceneIndex;
  std::vector<double> distance;
};

NAN_METHOD(Features::Similarity) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(2, cb);

  Mat image1 = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat.clone();
  Mat image2 = Nan::ObjectWrap::Unwrap<Matrix>(info[1]->ToObject())->mat.clone();

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncDetectSimilarity(callback, image1, image2) );
  return;
}

class AsyncDetectFeatures: public Nan::AsyncWorker {
public:
  AsyncDetectFeatures(Nan::Callback *callback, Mat image) :
      Nan::AsyncWorker(callback),
      image(image) {
  }

  ~AsyncDetectFeatures() {
  }

  void Execute() {

    Mat blur;
    Mat gray;
 
    Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");
    Ptr<DescriptorExtractor> extractor =
        DescriptorExtractor::create("ORB");
    
    bilateralFilter(image, blur, 9, 75, 75);
    cvtColor( blur, gray, CV_BGR2GRAY );

    detector->detect(gray, keypoints);
    extractor->compute(gray, keypoints, descriptors);
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;

    Local<Value> argv[2];
    Local<Object> result = Nan::New<Object>();

    argv[0] = Nan::Null();
    argv[1] = result;

    int size = keypoints.size();

    Local<Object> key;

    Local<Array> keys = Nan::New<Array>(size); 
    for (int i = 0; i < size; i++) { 
      key = Nan::New<Object>();
      key->Set(Nan::New<v8::String>("pointx").ToLocalChecked(), Nan::New<Number>(keypoints[i].pt.x));
      key->Set(Nan::New<v8::String>("pointy").ToLocalChecked(), Nan::New<Number>(keypoints[i].pt.y));
      key->Set(Nan::New<v8::String>("size").ToLocalChecked(), Nan::New<Number>(keypoints[i].size));
      key->Set(Nan::New<v8::String>("angle").ToLocalChecked(), Nan::New<Number>(keypoints[i].angle));
      key->Set(Nan::New<v8::String>("response").ToLocalChecked(), Nan::New<Number>(keypoints[i].response));
      key->Set(Nan::New<v8::String>("octave").ToLocalChecked(), Nan::New<Number>(keypoints[i].octave));
      key->Set(Nan::New<v8::String>("class_id").ToLocalChecked(), Nan::New<Number>(keypoints[i].class_id));

      keys->Set(Nan::New<Number>(i), key); 
    }

    result->Set(Nan::New<v8::String>("keypoints").ToLocalChecked(), keys);
    Local<Object> im_h = Nan::New(Matrix::constructor)->GetFunction()->NewInstance();
    Matrix *img = Nan::ObjectWrap::Unwrap<Matrix>(im_h);
    img->mat = descriptors;

    result->Set(Nan::New<v8::String>("descriptors").ToLocalChecked(), im_h);

    callback->Call(2, argv);
  }

private:
  Mat image;
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
};

NAN_METHOD(Features::DetectFeatures) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(1, cb);

  Mat image = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat.clone();

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncDetectFeatures(callback, image) );
  return;
}


class AsyncMatch: public Nan::AsyncWorker {
public:
  AsyncMatch(Nan::Callback *callback,  std::vector<KeyPoint> keypoints1, Mat descriptors1,  std::vector<KeyPoint> keypoints2, Mat descriptors2) :
      Nan::AsyncWorker(callback),
      keypoints1(keypoints1),
      descriptors1(descriptors1),
      keypoints2(keypoints2),
      descriptors2(descriptors2) {
  }

  ~AsyncMatch() {
  }

  void Execute() {

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(
        "BruteForce-Hamming");
        
    std::vector<DMatch> matches12;
    std::vector<DMatch> matches21;
    std::vector<DMatch> matches;
        
    matcher->match(descriptors1, descriptors2, matches12);
    matcher->match(descriptors2, descriptors1, matches21);
    
    // cross check
    for( size_t i = 0; i < matches12.size(); i++ )
    { 
        DMatch forward = matches12[i]; 
        DMatch backward = matches21[forward.trainIdx];
        if( backward.trainIdx == forward.queryIdx ) 
            matches.push_back( forward ); 
    }  
    
    for( size_t i = 0; i < matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      objIndex.push_back(matches[i].queryIdx);
      sceneIndex.push_back(matches[i].trainIdx);
      obj.push_back( keypoints1[ matches[i].queryIdx ].pt );
      scene.push_back( keypoints2[ matches[i].trainIdx ].pt );
      distance.push_back(matches[i].distance);   
    }
   
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;

    Local<Value> argv[2];

    argv[0] = Nan::Null();

    Local<Object> match;
    Local<Object> point1;
    Local<Object> point2;

    Local<Array> matches = Nan::New<Array>(obj.size()); 
    for (size_t i = 0; i < obj.size(); i++) { 
      match = Nan::New<Object>();
      point1 = Nan::New<Object>();
      point2 = Nan::New<Object>();

      match->Set(Nan::New<v8::String>("point1").ToLocalChecked(), point1);
      match->Set(Nan::New<v8::String>("point2").ToLocalChecked(), point2);

      point1->Set(Nan::New<v8::String>("x").ToLocalChecked(), Nan::New<Number>(obj[i].x));
      point1->Set(Nan::New<v8::String>("y").ToLocalChecked(), Nan::New<Number>(obj[i].y));
      

      point2->Set(Nan::New<v8::String>("x").ToLocalChecked(), Nan::New<Number>(scene[i].x));
      point2->Set(Nan::New<v8::String>("y").ToLocalChecked(), Nan::New<Number>(scene[i].y));

      match->Set(Nan::New<v8::String>("d").ToLocalChecked(), Nan::New<Number>(distance[i]));
      match->Set(Nan::New<v8::String>("q").ToLocalChecked(), Nan::New<Number>(objIndex[i]));
      match->Set(Nan::New<v8::String>("t").ToLocalChecked(), Nan::New<Number>(sceneIndex[i]));
      
      matches->Set(Nan::New<Number>(i), match); 
    }
     
    argv[1] = matches;

    callback->Call(2, argv);
  }

private:
  std::vector<KeyPoint> keypoints1;
  Mat descriptors1;
  std::vector<KeyPoint> keypoints2;
  Mat descriptors2;
  
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;
  std::vector<int> objIndex;
  std::vector<int> sceneIndex;
  std::vector<double> distance;
};

NAN_METHOD(Features::Match) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(2, cb);

  std::vector<KeyPoint> keypoints1;
  std::vector<KeyPoint> keypoints2;
  Local<Object> features1 = info[0]->ToObject();
  Local<Object> features2 = info[1]->ToObject();

  int size;
  Local<Object> key;
  Local<Array> keys1 = Nan::Get(features1, Nan::New<v8::String>("keypoints").ToLocalChecked()).ToLocalChecked()->ToObject().As<Array>();
  Local<Array> keys2 = Nan::Get(features2, Nan::New<v8::String>("keypoints").ToLocalChecked()).ToLocalChecked()->ToObject().As<Array>();;

  size = Nan::Get(keys1, Nan::New<v8::String>("length").ToLocalChecked()).ToLocalChecked()->Uint32Value();
  for(int i = 0; i < size; i++) {
    key =  Nan::Get(keys1, Nan::New<Number>(i)).ToLocalChecked().As<Object>();
    keypoints1.push_back(KeyPoint(
      Nan::Get(key, Nan::New<v8::String>("pointx").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("pointy").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("size").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("angle").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("response").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("octave").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("class_id").ToLocalChecked()).ToLocalChecked()->Uint32Value()
      ));  
  }

  size = Nan::Get(keys2, Nan::New<v8::String>("length").ToLocalChecked()).ToLocalChecked()->Uint32Value();
  for(int i = 0; i < size; i++) {
    key =  Nan::Get(keys2, Nan::New<Number>(i)).ToLocalChecked().As<Object>();
    keypoints2.push_back(KeyPoint(
      Nan::Get(key, Nan::New<v8::String>("pointx").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("pointy").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("size").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("angle").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("response").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("octave").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("class_id").ToLocalChecked()).ToLocalChecked()->Uint32Value()
      ));  
  }
  Local<v8::String> name =  Nan::New<v8::String>("descriptors").ToLocalChecked();
  Local<Object> des = Nan::Get(features1,name).ToLocalChecked().As<Object>();
  Mat descriptors1 = Nan::ObjectWrap::Unwrap<Matrix>(des)->mat.clone();
  des = Nan::Get(features2,name).ToLocalChecked().As<Object>();
  Mat descriptors2 = Nan::ObjectWrap::Unwrap<Matrix>(des)->mat.clone();

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncMatch(callback, keypoints1, descriptors1, keypoints2, descriptors2));
  return;
}

class AsyncDrawFeatures: public Nan::AsyncWorker {
public:
  AsyncDrawFeatures(Nan::Callback *callback, Mat image) :
      Nan::AsyncWorker(callback),
      image(image) {
  }

  ~AsyncDrawFeatures() {
  }

  void Execute() {

    Mat blur;
    Mat eqhist;
    Mat gray;
    Mat eqgray;
    Mat mask;

    bilateralFilter(image, blur, 9, 75, 75);
    cvtColor( blur, gray, CV_BGR2GRAY );

    std::vector<KeyPoint> keypoints;
    Mat descriptors;

    Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("ORB");

    descriptors = Mat();

    detector->detect(blur, keypoints);

    extractor->compute(blur, keypoints, descriptors);

    drawKeypoints(blur, keypoints, final);
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;

    Local<Value> argv[2];
    Local<Object> im_h = Nan::New(Matrix::constructor)->GetFunction()->NewInstance();
    Matrix *img = Nan::ObjectWrap::Unwrap<Matrix>(im_h);
    img->mat = final;

    argv[0] = Nan::Null();
    argv[1] = im_h;

    callback->Call(2, argv);
  }

private:
  Mat image;
  Mat final;
};

NAN_METHOD(Features::DrawFeatures) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(1, cb);

  Mat image = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat.clone();

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncDrawFeatures(callback, image));
  return;
}

class AsyncDrawMatches: public Nan::AsyncWorker {
public:
  AsyncDrawMatches(Nan::Callback *callback, Mat image1, std::vector<KeyPoint> keypoints1, Mat image2, std::vector<KeyPoint> keypoints2, std::vector<DMatch> matches ) :
      Nan::AsyncWorker(callback),
      image1(image1),
      keypoints1(keypoints1),
      image2(image2),
      keypoints2(keypoints2),
      matches(matches){
  }

  ~AsyncDrawMatches() {
  }

  void Execute() {
     drawMatches(image1, keypoints1, image2, keypoints2, matches, final);
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;

    Local<Value> argv[2];
    Local<Object> im_h = Nan::New(Matrix::constructor)->GetFunction()->NewInstance();
    Matrix *img = Nan::ObjectWrap::Unwrap<Matrix>(im_h);
    img->mat = final;

    argv[0] = Nan::Null();
    argv[1] = im_h;

    callback->Call(2, argv);
  }

private:
  Mat image1;
  std::vector<KeyPoint> keypoints1;
  Mat image2;
  std::vector<KeyPoint> keypoints2;
  std::vector<DMatch> matches;
  
  Mat final;
  
};

NAN_METHOD(Features::DrawMatches) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(5, cb);

  Mat image1 = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat.clone();
  Mat image2 = Nan::ObjectWrap::Unwrap<Matrix>(info[2]->ToObject())->mat.clone();
  
  std::vector<KeyPoint> keypoints1;
  std::vector<KeyPoint> keypoints2;
  Local<Object> features1 = info[1]->ToObject();
  Local<Object> features2 = info[3]->ToObject();

  int size;
  Local<Object> key;
  Local<Array> keys1 = Nan::Get(features1, Nan::New<v8::String>("keypoints").ToLocalChecked()).ToLocalChecked()->ToObject().As<Array>();
  Local<Array> keys2 = Nan::Get(features2, Nan::New<v8::String>("keypoints").ToLocalChecked()).ToLocalChecked()->ToObject().As<Array>();

  size = Nan::Get(keys1, Nan::New<v8::String>("length").ToLocalChecked()).ToLocalChecked()->Uint32Value();
  for(int i = 0; i < size; i++) {
    key =  Nan::Get(keys1, Nan::New<Number>(i)).ToLocalChecked().As<Object>();
    keypoints1.push_back(KeyPoint(
      Nan::Get(key, Nan::New<v8::String>("pointx").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("pointy").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("size").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("angle").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("response").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("octave").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("class_id").ToLocalChecked()).ToLocalChecked()->Uint32Value()
      ));  
  }

  size = Nan::Get(keys2, Nan::New<v8::String>("length").ToLocalChecked()).ToLocalChecked()->Uint32Value();
  for(int i = 0; i < size; i++) {
    key =  Nan::Get(keys2, Nan::New<Number>(i)).ToLocalChecked().As<Object>();
    keypoints2.push_back(KeyPoint(
      Nan::Get(key, Nan::New<v8::String>("pointx").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("pointy").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("size").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("angle").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("response").ToLocalChecked()).ToLocalChecked()->NumberValue(),
      Nan::Get(key, Nan::New<v8::String>("octave").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("class_id").ToLocalChecked()).ToLocalChecked()->Uint32Value()
      ));  
  }

  std::vector<DMatch> matches;
  Local<Object> match;
  Local<Array> matchList = info[4]->ToObject().As<Array>();
  size = Nan::Get(matchList, Nan::New<v8::String>("length").ToLocalChecked()).ToLocalChecked()->Uint32Value();
  for(int i = 0; i < size; i++) {
    match =  Nan::Get(matchList, Nan::New<Number>(i)).ToLocalChecked().As<Object>();
    matches.push_back(DMatch(
       Nan::Get(match, Nan::New<v8::String>("q").ToLocalChecked()).ToLocalChecked()->Int32Value(),
       Nan::Get(match, Nan::New<v8::String>("t").ToLocalChecked()).ToLocalChecked()->Int32Value(),
       Nan::Get(match, Nan::New<v8::String>("d").ToLocalChecked()).ToLocalChecked()->NumberValue()
      ));
  }

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncDrawMatches(callback, image1, keypoints1, image2, keypoints2, matches));
  return;
}



#endif
