#include "Features2d.h"
#include "Matrix.h"
#include <nan.h>
#include <stdio.h>

#if ((CV_MAJOR_VERSION >= 2) && (CV_MINOR_VERSION >=4))

bool niceHomography(cv::Mat H)
{

    const double det = H.at<double>(0, 0) * H.at<double>(1, 1) - H.at<double>(1, 0) * H.at<double>(0, 1);
    if (det < 0)
    {  
        return false;
    }

    const double N1 = sqrt(H.at<double>(0, 0) * H.at<double>(0, 0) + H.at<double>(1, 0) * H.at<double>(1, 0));
    if (N1 > 4 || N1 < 0.1)
    {
        return false;
    }

    const double N2 = sqrt(H.at<double>(0, 1) * H.at<double>(0, 1) + H.at<double>(1, 1) * H.at<double>(1, 1));
    if (N2 > 4 || N2 < 0.1)
    {
        return false;
    }

    const double N3 = sqrt(H.at<double>(2, 0) * H.at<double>(2, 0) + H.at<double>(2, 1) * H.at<double>(2, 1));
    if (N3 > 0.002)
    {
        return false;
    }

    return true;
}

void Features::Init(Local<Object> target) {
  Nan::HandleScope scope;

  Nan::SetMethod(target, "ImageSimilarity", Similarity);
  Nan::SetMethod(target, "DetectAndCompute", DetectAndCompute);
  Nan::SetMethod(target, "FilteredMatch", FilteredMatch);
  Nan::SetMethod(target, "MaskText", MaskText);
}

class AsyncDetectSimilarity: public Nan::AsyncWorker {
public:
  AsyncDetectSimilarity(Nan::Callback *callback, cv::Mat image1, cv::Mat image2) :
      Nan::AsyncWorker(callback),
      image1(image1),
      image2(image2),
      d_good(0),
      n_good(0),
      d_h(0),
      n_h(0),
      condition(0) {
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
    //-- or a smalli arbitary value ( 0.02 ) in the event that min_dist is very
    //-- smalli)
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

    d_good = (double) good_matches_sum / (double) good_matches.size();
    n_good = good_matches.size();

    // we can't make a homography matrix with less than 4 points
    if(good_matches.size() < 4) {
      drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches);
      return;
    };

     //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    cv::Mat mask;
    std::vector<cv::DMatch> h_matches;
    double h_matches_sum = 0.0;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
      scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    cv::Mat H = findHomography( obj, scene, cv::RANSAC, 3, mask);

    if(niceHomography(H)){
      for (int i = 0; i < good_matches.size(); i++)
      {
          // Select only the inliers (mask entry set to 1)
          if ((int)mask.at<uchar>(i, 0) == 1)
          {
              h_matches.push_back(good_matches[i]);
              h_matches_sum += good_matches[i].distance;
          }
      }
    }

    d_h = (double) h_matches_sum / (double) h_matches.size();
    n_h = h_matches.size();
    drawMatches(image1, keypoints1, image2, keypoints2, h_matches, img_matches);

    cv::Mat w;
    cv::SVD::compute(H,w);
    condition = ((double*)w.data)[H.cols-1]/((double*)w.data)[0];

  }

  void HandleOKCallback() {
    Nan::HandleScope scope;

    Local<Value> argv[7];

    argv[0] = Nan::Null();

    Local<Object> im_h = Nan::New(Matrix::constructor)->GetFunction()->NewInstance();
    Matrix *img = Nan::ObjectWrap::Unwrap<Matrix>(im_h);
    img->mat = img_matches;
    
    argv[1] = im_h;
    argv[2] = Nan::New<Number>(d_good);
    argv[3] = Nan::New<Number>(n_good);
    argv[4] = Nan::New<Number>(d_h);
    argv[5] = Nan::New<Number>(n_h);
    argv[6] = Nan::New<Number>(condition);

    callback->Call(7, argv);
  }

private:
  cv::Mat image1;
  cv::Mat image2;
  double d_good;
  int n_good;
  double d_h;
  int n_h;
  cv::Mat img_matches;
  double condition;
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
      d_good(0),
      n_good(0),
      d_h(0),
      n_h(0),
      condition(0) {
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
    //-- or a smalli arbitary value ( 0.02 ) in the event that min_dist is very
    //-- smalli)
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

    d_good = (double) good_matches_sum / (double) good_matches.size();
    n_good = good_matches.size();

    // we can't make a homography matrix with less than 4 points
    if(n_good < 4) {return;};

     //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    cv::Mat mask;
    std::vector<cv::DMatch> h_matches;
    double h_matches_sum = 0.0;

    for( size_t i = 0; i < n_good; i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
      scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    cv::Mat H = findHomography( obj, scene, cv::RANSAC, 3, mask);

    if(niceHomography(H)){
      for (int i = 0; i < good_matches.size(); i++)
      {
          // Select only the inliers (mask entry set to 1)
          if ((int)mask.at<uchar>(i, 0) == 1)
          {
              h_matches.push_back(good_matches[i]);
              h_matches_sum += good_matches[i].distance;
          }
      }
    }
    
    d_h = (double) h_matches_sum / (double) h_matches.size();
    n_h = h_matches.size();

    cv::Mat w;
    cv::SVD::compute(H,w);
    condition = ((double*)w.data)[H.cols-1]/((double*)w.data)[0];

  }

  void HandleOKCallback() {
    Nan::HandleScope scope;

    Local<Value> argv[6];

    argv[0] = Nan::Null();
    argv[1] = Nan::New<Number>(d_good);
    argv[2] = Nan::New<Number>(n_good);
    argv[3] = Nan::New<Number>(d_h);
    argv[4] = Nan::New<Number>(n_h);
    argv[5] = Nan::New<Number>(condition);

    callback->Call(6, argv);
  }

private:
  std::vector<cv::KeyPoint> keypoints1;
  cv::Mat descriptors1;
  std::vector<cv::KeyPoint> keypoints2;
  cv::Mat descriptors2;
  double d_good;
  int n_good;
  double d_h;
  int n_h;
  double condition;
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

  size = Nan::Get(keys2, Nan::New<String>("length").ToLocalChecked()).ToLocalChecked()->Uint32Value();
  for(int i = 0; i < size; i++) {
    key =  Nan::Get(keys2, Nan::New<Number>(i)).ToLocalChecked().As<Object>();
    keypoints2.push_back(cv::KeyPoint(
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


class AsyncMaskText: public Nan::AsyncWorker {
public:
  AsyncMaskText(Nan::Callback *callback, cv::Mat image) :
      Nan::AsyncWorker(callback),
      image(image) {
  }

  ~AsyncMaskText() {
  }

  void Execute() {

    using namespace cv;

    Mat smalli;
    // downsample and use it for processing
    //pyrDown(image, rgb);
    cvtColor(image, smalli, CV_BGR2GRAY);
    // morphological gradient
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(smalli, grad, MORPH_GRADIENT, morphKernel);
    // binarize
    Mat bw;
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    // connect horizontally oriented regions
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(7, 2));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
    // find contours
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    // filter contours
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
    {
        Rect rect = boundingRect(contours[idx]);
        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
        // fill the contour
        drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
        // ratio of non-zero pixels in the filled region
        double r = (double)countNonZero(maskROI)/(rect.width*rect.height);

        if (r > .40 /* assume at least 40% of the area is filled if it contains text */
            && 
            (rect.height > 8 && rect.width > 8) /* constraints on region size */
            /* these two conditions alone are not very robust. better to use something 
            like the number of significant peaks in a horizontal projection as a third condition */
            &&
            ((double)rect.height / (double) image.rows < 0.20 )
            )
        {
            // find out what color to paint

          /// Separate the image in 3 places ( B, G and R )
          Mat src = image(rect);
          vector<cv::Mat> bgr_planes;
          split( src, bgr_planes );

          /// Establish the number of bins
          int histSize = 256;

          /// Set the ranges ( for B,G,R) )
          float range[] = { 0, 256 } ;
          const float* histRange = { range };

          bool uniform = true; bool accumulate = false;

          Mat b_hist, g_hist, r_hist;

          /// Compute the histograms:
          calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
          calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
          calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

          int maxr = 0;
          int maxg = 0;
          int maxb = 0;

          for(int i = 0; i < 256; i++){
            if(r_hist.at<float>(i) > r_hist.at<float>(maxr)) {
              maxr = i;
            }
            if(g_hist.at<float>(i) > g_hist.at<float>(maxg)) {
              maxg = i;
            }
            if(b_hist.at<float>(i) > b_hist.at<float>(maxb)) {
              maxb = i;
            }
          }

          rectangle(image, rect, Scalar(maxb, maxg, maxr), CV_FILLED);
        }
    }

    final = image;

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
  cv::Mat image;
  cv::Mat final;
};

NAN_METHOD(Features::MaskText) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(1, cb);

  cv::Mat image = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat;

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncMaskText(callback, image) );
  return;
}




#endif
