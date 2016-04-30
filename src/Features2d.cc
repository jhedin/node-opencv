#include "Features2d.h"
#include "Matrix.h"
#include <nan.h>
#include <stdio.h>
#include <cmath>

#if ((CV_MAJOR_VERSION >= 2) && (CV_MINOR_VERSION >=4))

using namespace cv;

Mat equalizeIntensity(const Mat& inputImage)
{
    if(inputImage.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        Mat result;
        merge(channels,ycrcb);

        cvtColor(ycrcb,result,CV_YCrCb2BGR);

        return result;
    }
    return Mat();
}

bool niceHomography(Mat H)
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

   /**
     * @brief makeCanvas Makes composite image from the given images
     * @param vecMat Vector of Images.
     * @param windowHeight The height of the new composite image to be formed.
     * @param nRows Number of rows of images. (Number of columns will be calculated
     *              depending on the value of total number of images).
     * @return new composite image.
     */
    cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows) {
            int N = vecMat.size();
            nRows  = nRows > N ? N : nRows; 
            int edgeThickness = 10;
            int imagesPerRow = ceil(double(N) / nRows);
            int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
            int maxRowLength = 0;

            std::vector<int> resizeWidth;
            for (int i = 0; i < N;) {
                    int thisRowLen = 0;
                    for (int k = 0; k < imagesPerRow; k++) {
                            double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
                            int temp = int( ceil(resizeHeight * aspectRatio));
                            resizeWidth.push_back(temp);
                            thisRowLen += temp;
                            if (++i == N) break;
                    }
                    if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
                            maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
                    }
            }
            int windowWidth = maxRowLength;
            cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));

            for (int k = 0, i = 0; i < nRows; i++) {
                    int y = i * resizeHeight + (i + 1) * edgeThickness;
                    int x_end = edgeThickness;
                    for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
                            int x = x_end;
                            cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
                            cv::Size s = canvasImage(roi).size();
                            // change the number of channels to three
                            cv::Mat target_ROI(s, CV_8UC3);
                            if (vecMat[k].channels() != canvasImage.channels()) {
                                if (vecMat[k].channels() == 1) {
                                    cv::cvtColor(vecMat[k], target_ROI, CV_GRAY2BGR);
                                }
                            }
                            cv::resize(target_ROI, target_ROI, s);
                            if (target_ROI.type() != canvasImage.type()) {
                                target_ROI.convertTo(target_ROI, canvasImage.type());
                            }
                            target_ROI.copyTo(canvasImage(roi));
                            x_end += resizeWidth[k] + edgeThickness;
                    }
            }
            return canvasImage;
    }

double otsu_8u_with_mask(const Mat src, const Mat& mask)
{
    const int N = 256;
    int M = 0;
    int i, j, h[N] = { 0 };
    for (i = 0; i < src.rows; i++)
    {
        const uchar* psrc = src.ptr(i);
        const uchar* pmask = mask.ptr(i);
        for (j = 0; j < src.cols; j++)
        {
            if (pmask[j])
            {
                h[psrc[j]]++;
                ++M;
            }
        }
    }

    double mu = 0, scale = 1. / (M);
    for (i = 0; i < N; i++)
        mu += i*(double)h[i];

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for (i = 0; i < N; i++)
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i] * scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
            continue;

        mu1 = (mu1 + i*p_i) / q1;
        mu2 = (mu - q1*mu1) / q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if (sigma > max_sigma)
        {
            max_sigma = sigma;
            max_val = i;
        }
    }

    return max_val;
}

double threshold_with_mask(Mat& src, Mat& dst, double thresh, double maxval, int type, const Mat& mask = Mat())
{
    if (mask.empty() || (mask.rows == src.rows && mask.cols == src.cols && countNonZero(mask) == src.rows * src.cols))
    {
        // If empty mask, or all-white mask, use threshold
        thresh = threshold(src, dst, thresh, maxval, type);
    }
    else
    {
        // Use mask
        bool use_otsu = (type & THRESH_OTSU) != 0;
        if (use_otsu)
        {
            // If OTSU, get thresh value on mask only
            thresh = otsu_8u_with_mask(src, mask);
            // Remove THRESH_OTSU from type
            type &= THRESH_MASK;
        }

        // Apply threshold on all image
        thresh = threshold(src, dst, thresh, maxval, type);

        // Copy original image on inverted mask
        src.copyTo(dst, ~mask);
    }
    return thresh;
}


void Features::Init(Local<Object> target) {
  Nan::HandleScope scope;

  Nan::SetMethod(target, "ImageSimilarity", Similarity);
  Nan::SetMethod(target, "DetectAndCompute", DetectAndCompute);
  Nan::SetMethod(target, "FilteredMatch", FilteredMatch);
  Nan::SetMethod(target, "MaskText", MaskText);
  Nan::SetMethod(target, "DrawFeatures", DrawFeatures);
}

class AsyncDetectSimilarity: public Nan::AsyncWorker {
public:
  AsyncDetectSimilarity(Nan::Callback *callback, Mat image1, Mat image2) :
      Nan::AsyncWorker(callback),
      image1(image1),
      image2(image2),
      d_good(NAN),
      n_good(0),
      d_h(NAN),
      n_h(0),
      condition(NAN) {
  }

  ~AsyncDetectSimilarity() {
  }

  void Execute() {

    Mat blur1;
    Mat eqhist1;
    Mat gray1;
    Mat eqgray1;
    Mat mask1;

    bilateralFilter(image1, blur1, 9, 75, 75);
    eqhist1 = equalizeIntensity(blur1);
    cvtColor( blur1, gray1, CV_BGR2GRAY );
    /*threshold(gray1, mask1, 230.0, 255.0, THRESH_BINARY);
    gray1.setTo(Scalar(255), mask1);
    equalizeHist( gray1, eqgray1);
    bilateralFilter(eqgray1, blur1, 9, 75, 75);*/
   

    Mat blur2;
    Mat eqhist2;
    Mat gray2;
    Mat eqgray2;
    Mat mask2;

    bilateralFilter(image2, blur2, 9, 75, 75);
    eqhist2 = equalizeIntensity(blur2);
    cvtColor( blur2, gray2, CV_BGR2GRAY );
    /*threshold(gray2, mask2, 230.0, 255.0, THRESH_BINARY);
    gray2.setTo(Scalar(255), mask2);
    equalizeHist( gray2, eqgray2);
    bilateralFilter(eqgray2, blur2, 9, 75, 75);*/
    
    blur1 = gray1;
    blur2 = gray2;
    
    Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");
    Ptr<DescriptorExtractor> extractor =
        DescriptorExtractor::create("ORB");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(
        "BruteForce-Hamming");

    std::vector<DMatch> matches;

    Mat descriptors1 = Mat();
    Mat descriptors2 = Mat();

    std::vector<KeyPoint> keypoints1;
    std::vector<KeyPoint> keypoints2;

    detector->detect(blur1, keypoints1);
    detector->detect(blur2, keypoints2);

    extractor->compute(blur1, keypoints1, descriptors1);
    extractor->compute(blur2, keypoints2, descriptors2);

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
    std::vector<DMatch> good_matches;
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
      drawMatches(blur1, keypoints1, blur2, keypoints2, good_matches, img_matches);
      return;
    };

     //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    Mat mask;
    std::vector<DMatch> h_matches;
    double h_matches_sum = 0.0;

    for( size_t i = 0; i < matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( keypoints1[ matches[i].queryIdx ].pt );
      scene.push_back( keypoints2[ matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC, 3, mask);

    if(niceHomography(H)){
      for (int i = 0; i < matches.size(); i++)
      {
          // Select only the inliers (mask entry set to 1)
          if ((int)mask.at<uchar>(i, 0) == 1)
          {
              h_matches.push_back(matches[i]);
              h_matches_sum += matches[i].distance;
          }
      }
    }

    d_h = (double) h_matches_sum / (double) h_matches.size();
    n_h = h_matches.size();
    if(n_h > 0){
      drawMatches(blur1, keypoints1, blur2, keypoints2, h_matches, img_matches);
    } else {
      drawMatches(blur1, keypoints1, blur2, keypoints2, good_matches, img_matches);
    }

    Mat w;
    SVD::compute(H,w);
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
  Mat image1;
  Mat image2;
  double d_good;
  int n_good;
  double d_h;
  int n_h;
  Mat img_matches;
  double condition;
};

NAN_METHOD(Features::Similarity) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(2, cb);

  Mat image1 = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat;
  Mat image2 = Nan::ObjectWrap::Unwrap<Matrix>(info[1]->ToObject())->mat;

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncDetectSimilarity(callback, image1, image2) );
  return;
}

class AsyncDetectAndCompute: public Nan::AsyncWorker {
public:
  AsyncDetectAndCompute(Nan::Callback *callback, Mat image) :
      Nan::AsyncWorker(callback),
      image(image) {
  }

  ~AsyncDetectAndCompute() {
  }

  void Execute() {

    Mat blur;
    Mat eqhist;
    Mat gray;
    Mat eqgray;
    Mat mask;

    bilateralFilter(image, blur, 9, 75, 75);
    eqhist = equalizeIntensity(blur);
    cvtColor( blur, gray, CV_BGR2GRAY );
    threshold(gray, mask, 230.0, 255.0, THRESH_BINARY);
    gray.setTo(Scalar(255), mask);
    equalizeHist( gray, eqgray);
    bilateralFilter(eqgray, blur, 9, 75, 75);
    

    Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("ORB");

    descriptors = Mat();

    detector->detect(blur, keypoints);

    extractor->compute(blur, keypoints, descriptors);
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

NAN_METHOD(Features::DetectAndCompute) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(1, cb);

  Mat image = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat;

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncDetectAndCompute(callback, image) );
  return;
}


class AsyncFilteredMatch: public Nan::AsyncWorker {
public:
  AsyncFilteredMatch(Nan::Callback *callback,  std::vector<KeyPoint> keypoints1, Mat descriptors1,  std::vector<KeyPoint> keypoints2, Mat descriptors2) :
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

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(
        "BruteForce-Hamming");

    std::vector<DMatch> matches;

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
    std::vector<DMatch> good_matches;
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
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    Mat mask;
    std::vector<DMatch> h_matches;
    double h_matches_sum = 0.0;

    for( size_t i = 0; i < n_good; i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
      scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC, 3, mask);

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

    Mat w;
    SVD::compute(H,w);
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
  std::vector<KeyPoint> keypoints1;
  Mat descriptors1;
  std::vector<KeyPoint> keypoints2;
  Mat descriptors2;
  double d_good;
  int n_good;
  double d_h;
  int n_h;
  double condition;
};

NAN_METHOD(Features::FilteredMatch) {
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
      Nan::Get(key, Nan::New<v8::String>("pointx").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("pointy").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("size").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("angle").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("response").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("octave").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("class_id").ToLocalChecked()).ToLocalChecked()->Uint32Value()
      ));  
  }

  size = Nan::Get(keys2, Nan::New<v8::String>("length").ToLocalChecked()).ToLocalChecked()->Uint32Value();
  for(int i = 0; i < size; i++) {
    key =  Nan::Get(keys2, Nan::New<Number>(i)).ToLocalChecked().As<Object>();
    keypoints2.push_back(KeyPoint(
      Nan::Get(key, Nan::New<v8::String>("pointx").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("pointy").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("size").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("angle").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("response").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("octave").ToLocalChecked()).ToLocalChecked()->Uint32Value(),
      Nan::Get(key, Nan::New<v8::String>("class_id").ToLocalChecked()).ToLocalChecked()->Uint32Value()
      ));  
  }
  Local<v8::String> name =  Nan::New<v8::String>("descriptors").ToLocalChecked();
  Local<Object> des = Nan::Get(features1,name).ToLocalChecked().As<Object>();
  Mat descriptors1 = Nan::ObjectWrap::Unwrap<Matrix>(des)->mat;
  des = Nan::Get(features2,name).ToLocalChecked().As<Object>();
  Mat descriptors2 = Nan::ObjectWrap::Unwrap<Matrix>(des)->mat;


  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncFilteredMatch(callback, keypoints1, descriptors1, keypoints2, descriptors2));
  return;
}


class AsyncMaskText: public Nan::AsyncWorker {
public:
  AsyncMaskText(Nan::Callback *callback, Mat image, int gradx, int grady, int connx, int conny, double filled, int boundx, int boundy, double heights, double thresh, int diax, int diay) :
      Nan::AsyncWorker(callback),
      gradx(gradx),
      grady(grady),
      connx(connx),
      conny(conny),
      filled(filled),
      boundx(boundx),
      boundy(boundy),
      heights(heights),
      thresh(thresh),
      diax(diax),
      diay(diay),
      image(image) {
  }

  ~AsyncMaskText() {
  }

  void Execute() {

    Mat morphKernel;
    Mat gray;

    Mat blur2;
    bilateralFilter(image, blur2, 9, 75, 75);
    Mat eqhist2;
    eqhist2 = equalizeIntensity(blur2);
    Mat gray2;
    cvtColor( blur2, gray2, CV_BGR2GRAY );

    Mat nMask2;
    threshold(gray2, nMask2, 230.0, 255.0, THRESH_BINARY);

    gray2.setTo(Scalar(255), nMask2);


    /// Apply Histogram Equalization
    cvtColor( blur2, gray, CV_BGR2GRAY );

    Mat nMask;
    threshold(gray, nMask, 230.0, 255.0, THRESH_BINARY);

    gray.setTo(Scalar(255), nMask);


    Mat blur;
    bilateralFilter(gray, blur, 9, 75, 75);
    //GaussianBlur(grad, blur, Size(5,5),0);

    Mat eqhist;
    equalizeHist( gray2, eqhist);

    // downsample and use it for processing
    //pyrDown(image, rgb);
    //cvtColor(image, smalli, CV_BGR2GRAY);
    // morphological gradient
    Mat grad;
    morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(gradx, grady));
    morphologyEx(gray2, grad, MORPH_GRADIENT, morphKernel);
  
    // need to ignore the brightest sections
    threshold(grad, nMask2, thresh, 255.0, THRESH_BINARY_INV); 

    Mat bw;
    threshold_with_mask(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU, nMask2);
    //threshold(grad, bw, thresh, 255.0, THRESH_BINARY);
    threshold(bw, bw, thresh, 255.0, THRESH_BINARY);

    // connect horizontally oriented regions
    Mat connected;
    morphKernel = getStructuringElement(MORPH_CROSS, Size(connx, conny));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);

    //Mat dilated;
    //morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(diax, diay));
    //morphologyEx(connected, dilated, MORPH_ERODE, morphKernel);

    Mat open;
    morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(diax, diay));
    morphologyEx(connected, open, MORPH_OPEN, morphKernel);
  
    //second set
    Mat gradeq;
    morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(gradx, grady));
    morphologyEx(eqhist, gradeq, MORPH_GRADIENT, morphKernel);

    // need to ignore the brightest sections
    threshold(gradeq, nMask, thresh, 255.0, THRESH_BINARY_INV); 

    Mat bweq;
    threshold_with_mask(gradeq, bweq, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU, nMask);
    //threshold(grad, bw, thresh, 255.0, THRESH_BINARY);
    threshold(bweq, bweq, thresh, 255.0, THRESH_BINARY); 

    // connect horizontally oriented regions
    Mat connectedeq;
    morphKernel = getStructuringElement(MORPH_CROSS, Size(connx, conny));
    morphologyEx(bweq, connectedeq, MORPH_CLOSE, morphKernel);

    //Mat dilated;
    //morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(diax, diay));
    //morphologyEx(connected, dilated, MORPH_ERODE, morphKernel);

    Mat openeq;
    morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(diax, diay));
    morphologyEx(connectedeq, openeq, MORPH_OPEN, morphKernel);

    Mat h2;
    Mat h3;
    // Create structure element for extracting horizontal lines through morphology operations
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size((int) open.cols/5,3));
    // Apply morphology operations
    erode(bweq, h2, horizontalStructure, Point(-1, -1));
    dilate(h2, h3, horizontalStructure, Point(-1, -1));

    Mat v2;
    Mat v3;
    // Create structure element for extracting horizontal lines through morphology operations
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(3,(int) open.rows/5));
    // Apply morphology operations
    erode(bweq, v2, verticalStructure, Point(-1, -1));
    dilate(v2, v3, verticalStructure, Point(-1, -1));

    Mat gradsum;
    addWeighted(grad,0.8,gradeq,0.3,0,gradsum,-1);

    gradsum.setTo(Scalar(0), h3);
    gradsum.setTo(Scalar(0), v3);

    // need to ignore the brightest sections
    threshold(gradsum, nMask, thresh, 255.0, THRESH_BINARY_INV); 

    Mat bwsum;
    threshold_with_mask(gradsum, bwsum, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU, nMask);
    //threshold(grad, bw, thresh, 255.0, THRESH_BINARY);
    threshold(bwsum, bwsum, thresh, 255.0, THRESH_BINARY); 

    // connect horizontally oriented regions
    Mat connectedsum;
    morphKernel = getStructuringElement(MORPH_CROSS, Size(connx, conny));
    morphologyEx(bwsum, connectedsum, MORPH_CLOSE, morphKernel);

    //Mat dilated;
    //morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(diax, diay));
    //morphologyEx(connected, dilated, MORPH_ERODE, morphKernel);

    Mat opensum;
    morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(diax, diay));
    morphologyEx(connectedsum, opensum, MORPH_OPEN, morphKernel);


    //std::vector<Mat> v { gray2, h3, v3, eqhist, grad, gradeq, gradsum, bwsum, closesum, erodesum, connectedsum, opensum};
    //final = makeCanvas(v,2500,3);
    //return;

    //final = open;
    //return;

    connected = opensum;

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
        maskROI = Scalar(255, 255, 255);
        // fill the contour
        drawContours(mask, contours, idx, Scalar(0, 0, 0), CV_FILLED);
        // ratio of non-zero pixels in the filled region
        double r = (double)countNonZero(maskROI)/(rect.width*rect.height);

        if (r < filled /* assume at least 40% of the area is filled if it contains text */
            && 
            (rect.height > boundy && rect.width > boundx) /* constraints on region size */
            /* these two conditions alone are not very robust. better to use something 
            like the number of significant peaks in a horizontal projection as a third condition */
            &&
            ((double)rect.height / (double) image.rows < heights )
            )
        {
          // find out what color to paint
          Mat src(image, rect);
          Mat hsv;
          cvtColor(src, hsv, CV_BGR2HSV);

          /// Separate the image in 3 places ( H, S and V )
          
          vector<Mat> hsv_planes;
          split( hsv, hsv_planes );

          /// Establish the number of bins
          int histSize = 256;

          /// Set the ranges ( for B,G,R) )
          float range[] = { 0, 256 } ;
          const float* histRange = { range };

          bool uniform = true; bool accumulate = false;

          Mat h_hist, s_hist, v_hist;

          /// Compute the histograms:
          calcHist( &hsv_planes[0], 1, 0, maskROI, h_hist, 1, &histSize, &histRange, uniform, accumulate );
          calcHist( &hsv_planes[1], 1, 0, maskROI, s_hist, 1, &histSize, &histRange, uniform, accumulate );
          calcHist( &hsv_planes[2], 1, 0, maskROI, v_hist, 1, &histSize, &histRange, uniform, accumulate );

          int maxh = 0;
          int maxs = 0;
          int maxv = 0;

          for(int i = 0; i < 256; i++){
            if(h_hist.at<float>(i) > h_hist.at<float>(maxh)) {
              maxh = i;
            }
            if(s_hist.at<float>(i) > s_hist.at<float>(maxs)) {
              maxs = i;
            }
            if(v_hist.at<float>(i) > v_hist.at<float>(maxv)) {
              maxv = i;
            }
          }
          
          hsv = Scalar(maxh, maxs, maxv);

          cvtColor(hsv, src, CV_HSV2BGR);

          Scalar color = src.at<Vec3b>(0,0);

          rectangle(image, rect, color, CV_FILLED);
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
  Mat image;
  Mat final;

  int gradx;
  int grady;
  int connx;
  int conny;
  double filled;
  int boundx;
  int boundy;
  double heights;
  double thresh;
  int diax;
  int diay;

};

NAN_METHOD(Features::MaskText) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(12, cb);

  Mat image = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat;

  int gradx = info[1]->NumberValue();
  int grady = info[2]->NumberValue();
  int connx = info[3]->NumberValue();
  int conny = info[4]->NumberValue();
  double filled = info[5]->NumberValue();
  int boundx = info[6]->NumberValue();
  int boundy = info[7]->NumberValue();
  double heights = info[8]->NumberValue();
  double thresh = info[9]->NumberValue();
  int diax = info[10]->NumberValue();
  int diay = info[11]->NumberValue();

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncMaskText(callback, image, gradx, grady, connx, conny, filled, boundx, boundy, heights, thresh, diax, diay));
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
    eqhist = equalizeIntensity(blur);
    cvtColor( blur, gray, CV_BGR2GRAY );
    threshold(gray, mask, 230.0, 255.0, THRESH_BINARY);
    gray.setTo(Scalar(255), mask);
    equalizeHist( gray, eqgray);
    bilateralFilter(eqgray, blur, 9, 75, 75);

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

  Mat image = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat;

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncDrawFeatures(callback, image));
  return;
}






#endif
