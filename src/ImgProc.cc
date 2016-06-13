#include "ImgProc.h"
#include "Matrix.h"

void ImgProc::Init(Local<Object> target) {
  Nan::Persistent<Object> inner;
  Local<Object> obj = Nan::New<Object>();
  inner.Reset(obj);

  Nan::SetMethod(obj, "undistort", Undistort);
  Nan::SetMethod(obj, "initUndistortRectifyMap", InitUndistortRectifyMap);
  Nan::SetMethod(obj, "remap", Remap);
  Nan::SetMethod(obj, "getStructuringElement", GetStructuringElement);
  Nan::SetMethod(obj, "MaskText", MaskText);

  target->Set(Nan::New("imgproc").ToLocalChecked(), obj);
}

double otsu_8u_with_mask(const cv::Mat src, const cv::Mat& mask)
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

double threshold_with_mask(cv::Mat& src, cv::Mat& dst, double thresh, double maxval, int type, const cv::Mat& mask = cv::Mat())
{
    if (mask.empty() || (mask.rows == src.rows && mask.cols == src.cols && cv::countNonZero(mask) == src.rows * src.cols))
    {
        // If empty mask, or all-white mask, use threshold
        thresh = cv::threshold(src, dst, thresh, maxval, type);
    }
    else
    {
        // Use mask
        bool use_otsu = (type & cv::THRESH_OTSU) != 0;
        if (use_otsu)
        {
            // If OTSU, get thresh value on mask only
            thresh = otsu_8u_with_mask(src, mask);
            // Remove cv::THRESH_OTSU from type
            type &= cv::THRESH_MASK;
        }

        // Apply threshold on all image
        thresh = cv::threshold(src, dst, thresh, maxval, type);

        // Copy original image on inverted mask
        src.copyTo(dst, ~mask);
    }
    return thresh;
}

// cv::undistort
NAN_METHOD(ImgProc::Undistort) {
  Nan::EscapableHandleScope scope;

  try {
    // Get the arguments

    // Arg 0 is the image
    Matrix* m0 = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject());
    cv::Mat inputImage = m0->mat;

    // Arg 1 is the camera matrix
    Matrix* m1 = Nan::ObjectWrap::Unwrap<Matrix>(info[1]->ToObject());
    cv::Mat K = m1->mat;

    // Arg 2 is the distortion coefficents
    Matrix* m2 = Nan::ObjectWrap::Unwrap<Matrix>(info[2]->ToObject());
    cv::Mat dist = m2->mat;

    // Make an mat to hold the result image
    cv::Mat outputImage;

    // Undistort
    cv::undistort(inputImage, outputImage, K, dist);

    // Wrap the output image
    Local<Object> outMatrixWrap = Nan::New(Matrix::constructor)->GetFunction()->NewInstance();
    Matrix *outMatrix = Nan::ObjectWrap::Unwrap<Matrix>(outMatrixWrap);
    outMatrix->mat = outputImage;

    // Return the output image
    info.GetReturnValue().Set(outMatrixWrap);
  } catch (cv::Exception &e) {
    const char *err_msg = e.what();
    Nan::ThrowError(err_msg);
    return;
  }
}

// cv::initUndistortRectifyMap
NAN_METHOD(ImgProc::InitUndistortRectifyMap) {
  Nan::EscapableHandleScope scope;

  try {
    // Arg 0 is the camera matrix
    Matrix* m0 = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject());
    cv::Mat K = m0->mat;

    // Arg 1 is the distortion coefficents
    Matrix* m1 = Nan::ObjectWrap::Unwrap<Matrix>(info[1]->ToObject());
    cv::Mat dist = m1->mat;

    // Arg 2 is the recification transformation
    Matrix* m2 = Nan::ObjectWrap::Unwrap<Matrix>(info[2]->ToObject());
    cv::Mat R = m2->mat;

    // Arg 3 is the new camera matrix
    Matrix* m3 = Nan::ObjectWrap::Unwrap<Matrix>(info[3]->ToObject());
    cv::Mat newK = m3->mat;

    // Arg 4 is the image size
    cv::Size imageSize;
    if (info[4]->IsArray()) {
      Local<Object> v8sz = info[4]->ToObject();
      imageSize = cv::Size(v8sz->Get(1)->IntegerValue(), v8sz->Get(0)->IntegerValue());
    } else {
      JSTHROW_TYPE("Must pass image size");
    }

    // Arg 5 is the first map type, skip for now
    int m1type = info[5]->IntegerValue();

    // Make matrices to hold the output maps
    cv::Mat map1, map2;

    // Compute the rectification map
    cv::initUndistortRectifyMap(K, dist, R, newK, imageSize, m1type, map1, map2);

    // Wrap the output maps
    Local<Object> map1Wrap = Nan::New(Matrix::constructor)->GetFunction()->NewInstance();
    Matrix *map1Matrix = Nan::ObjectWrap::Unwrap<Matrix>(map1Wrap);
    map1Matrix->mat = map1;

    Local<Object> map2Wrap = Nan::New(Matrix::constructor)->GetFunction()->NewInstance();
    Matrix *map2Matrix = Nan::ObjectWrap::Unwrap<Matrix>(map2Wrap);
    map2Matrix->mat = map2;

    // Make a return object with the two maps
    Local<Object> ret = Nan::New<Object>();
    ret->Set(Nan::New<String>("map1").ToLocalChecked(), map1Wrap);
    ret->Set(Nan::New<String>("map2").ToLocalChecked(), map2Wrap);

    // Return the maps
    info.GetReturnValue().Set(ret);
  } catch (cv::Exception &e) {
    const char *err_msg = e.what();
    Nan::ThrowError(err_msg);
    return;
  }
}

// cv::remap
NAN_METHOD(ImgProc::Remap) {
  Nan::EscapableHandleScope scope;

  try {
    // Get the arguments

    // Arg 0 is the image
    Matrix* m0 = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject());
    cv::Mat inputImage = m0->mat;

    // Arg 1 is the first map
    Matrix* m1 = Nan::ObjectWrap::Unwrap<Matrix>(info[1]->ToObject());
    cv::Mat map1 = m1->mat;

    // Arg 2 is the second map
    Matrix* m2 = Nan::ObjectWrap::Unwrap<Matrix>(info[2]->ToObject());
    cv::Mat map2 = m2->mat;

    // Arg 3 is the interpolation mode
    int interpolation = info[3]->IntegerValue();

    // Args 4, 5 border settings, skipping for now

    // Output image
    cv::Mat outputImage;

    // Remap
    cv::remap(inputImage, outputImage, map1, map2, interpolation);

    // Wrap the output image
    Local<Object> outMatrixWrap = Nan::New(Matrix::constructor)->GetFunction()->NewInstance();
    Matrix *outMatrix = Nan::ObjectWrap::Unwrap<Matrix>(outMatrixWrap);
    outMatrix->mat = outputImage;

    // Return the image
    info.GetReturnValue().Set(outMatrixWrap);
  } catch (cv::Exception &e) {
    const char *err_msg = e.what();
    Nan::ThrowError(err_msg);
    return;
  }
}

// cv::getStructuringElement
NAN_METHOD(ImgProc::GetStructuringElement) {
  Nan::EscapableHandleScope scope;

  try {
    // Get the arguments

    if (info.Length() != 2) {
      Nan::ThrowTypeError("Invalid number of arguments");
    }

    // Arg 0 is the element shape
    if (!info[0]->IsNumber()) {
      JSTHROW_TYPE("'shape' argument must be a number");
    }
    int shape = info[0]->NumberValue();

    // Arg 1 is the size of the structuring element
    cv::Size ksize;
    if (!info[1]->IsArray()) {
      JSTHROW_TYPE("'ksize' argument must be a 2 double array");
    }
    Local<Object> v8sz = info[1]->ToObject();
    ksize = cv::Size(v8sz->Get(0)->IntegerValue(), v8sz->Get(1)->IntegerValue());

    // GetStructuringElement
    cv::Mat mat = cv::getStructuringElement(shape, ksize);

    // Wrap the output image
    Local<Object> outMatrixWrap = Nan::New(Matrix::constructor)->GetFunction()->NewInstance();
    Matrix *outMatrix = ObjectWrap::Unwrap<Matrix>(outMatrixWrap);
    outMatrix->mat = mat;

    // Return the image
    info.GetReturnValue().Set(outMatrixWrap);
  } catch (cv::Exception &e) {
    const char *err_msg = e.what();
    JSTHROW(err_msg);
    return;
  }
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

    cv::Mat morphKernel;
    cv::Mat gray;

    cv::Mat blur;
    cv::bilateralFilter(image, blur, 9, 75, 75);
    cv::cvtColor( blur, gray, CV_BGR2GRAY );

    cv::Mat nMask;
    cv::threshold(gray, nMask, 230.0, 255.0, cv::THRESH_BINARY);
    gray.setTo(cv::Scalar(255), nMask);

    cv::Mat eqhist;
    cv::equalizeHist( gray, eqhist);

    // morphological gradient
    cv::Mat grad;
    morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(gray, grad, cv::MORPH_GRADIENT, morphKernel);
  
    //second set
    cv::Mat gradeq;
    morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(eqhist, gradeq, cv::MORPH_GRADIENT, morphKernel);

    // need to ignore the brightest sections
    cv::Mat bweq;
    cv::threshold(gradeq, nMask, 190, 255.0, cv::THRESH_BINARY_INV); 
    threshold_with_mask(gradeq, bweq, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU, nMask);
    cv::threshold(bweq, bweq, 190, 255.0, cv::THRESH_BINARY); 

    cv::Mat gradsum;
    cv::addWeighted(grad,0.8,gradeq,0.3,0,gradsum,-1);

    // remove lines
    cv::Mat h2;
    cv::Mat h3;
    int minh = image.cols*0.6 < 200 ? (int) image.cols*0.6 : 200;
    // Create structure element for extracting horizontal lines through morphology operations
    cv::Mat horizontalStructure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(minh,3));
    // Apply morphology operations
    cv::erode(bweq, h2, horizontalStructure, cv::Point(-1, -1));
    cv::dilate(h2, h3, horizontalStructure, cv::Point(-1, -1));

    cv::Mat v2;
    cv::Mat v3;
    int minv = image.rows*0.6 < 200 ? (int) image.rows*0.6 : 200;
    // Create structure element for extracting horizontal lines through morphology operations
    cv::Mat verticalStructure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,minv));
    // Apply morphology operations
    cv::erode(bweq, v2, verticalStructure, cv::Point(-1, -1));
    cv::dilate(v2, v3, verticalStructure, cv::Point(-1, -1));

    // need to check that the lines aren't actually text.
    // find contours
    cv::Mat h4 = h3.clone();
    cv::Mat v4 = v3.clone();
    cv::Mat linesGray = gray.clone();
    std::vector<std::vector<cv::Point>> lineContours;
    std::vector<cv::Vec4i> lineHierarchy;
    cv::findContours(h4, lineContours, lineHierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    // filter contours
    if(lineContours.size()) {
      for(int idx = 0; idx >= 0; idx = lineHierarchy[idx][0])
      {
          if(lineContours[idx].size() < 2) continue;
          cv::Rect rect = cv::boundingRect(lineContours[idx]);
          cv::Mat roi(gray, rect);
          cv::Mat test;
          cv::threshold(roi, test, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
          double r = (double)cv::countNonZero(test)/(test.rows * test.cols);
          //printf("r: %2.1lf %d %d\n ", r, countNonZero(test), (test.rows * test.cols));
          if((r >= 0.5 || r <= 1.0 - 0.5) && (r < 5.5 || r > 4.5)) {
            //printf("found h line\n");
            cv::rectangle(gradsum, rect, cv::Scalar(0,0,0), CV_FILLED);
            cv::rectangle(h3, rect, cv::Scalar(50,50,50), CV_FILLED);
          }   
      }
    }
    //printf("found contours v, %d\n", lineContours.size());
    cv::findContours(v4, lineContours, lineHierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    // filter contours
    if(lineContours.size()) {
      for(int idx = 0; idx >= 0; idx = lineHierarchy[idx][0])
      {
          if(lineContours[idx].size() < 2) continue;
          cv::Rect rect = cv::boundingRect(lineContours[idx]);
          cv::Mat roi(gray, rect);
          cv::Mat test;
          cv::threshold(roi, test, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
          double r = (double)cv::countNonZero(test)/(test.rows * test.cols);
          if((r >= 0.5 || r <= 1.0 - 0.5) && (r < 5.5 || r > 4.5)) {
            cv::rectangle(gradsum, rect, cv::Scalar(0,0,0), CV_FILLED);
            cv::rectangle(v3, rect, cv::Scalar(50,50,50), CV_FILLED);
          }
      }
    }

    // need to ignore the brightest sections
    cv::threshold(gradsum, nMask, 190, 255.0, cv::THRESH_BINARY_INV); 

    cv::Mat bwsum;
    threshold_with_mask(gradsum, bwsum, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU, nMask);
    cv::threshold(bwsum, bwsum, 190, 255.0, cv::THRESH_BINARY); 

    // connect horizontally oriented regions
    cv::Mat connectedsum;
    morphKernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(2, 2));
    cv::morphologyEx(bwsum, connectedsum, cv::MORPH_CLOSE, morphKernel);

    cv::Mat opensum;
    morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::morphologyEx(connectedsum, opensum, cv::MORPH_OPEN, morphKernel);

    // find contours
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(opensum, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    // filter contours
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
    {
        cv::Rect rect = cv::boundingRect(contours[idx]);
        cv::Mat maskROI(mask, rect);
        maskROI = cv::Scalar(255, 255, 255);
        // fill the contour
        cv::drawContours(mask, contours, idx, cv::Scalar(0, 0, 0), CV_FILLED);
        // ratio of non-zero pixels in the filled region
        double r = (double)cv::countNonZero(maskROI)/(rect.width*rect.height);
        if (r < 5 /* assume at least 40% of the area is filled if it contains text */
            && (rect.height > 5 && rect.width > 5) /* constraints on region size */
            /* these two conditions alone are not very robust. better to use something 
            like the number of significant peaks in a horizontal projection as a third condition */
            && (rect.height > 5 + 10 || rect.width > 5 + 10)
            &&((double)rect.height / (double) image.rows < 0.6 )
            &&(rect.height < 110)
            )
        {
          // find out what color to paint
          cv::Mat src(image, rect);
          cv::Mat hsv;
          cv::cvtColor(src, hsv, CV_BGR2HSV);

          /// Separate the image in 3 places ( H, S and V )
          
          std::vector<cv::Mat> hsv_planes;
          cv::split( hsv, hsv_planes );

          /// Establish the number of bins
          int histSize = 256;

          /// Set the ranges ( for B,G,R) )
          float range[] = { 0, 256 } ;
          const float* histRange = { range };

          bool uniform = true; bool accumulate = false;

          cv::Mat h_hist, s_hist, v_hist;

          /// Compute the histograms:
          cv::calcHist( &hsv_planes[0], 1, 0, maskROI, h_hist, 1, &histSize, &histRange, uniform, accumulate );
          cv::calcHist( &hsv_planes[1], 1, 0, maskROI, s_hist, 1, &histSize, &histRange, uniform, accumulate );
          cv::calcHist( &hsv_planes[2], 1, 0, maskROI, v_hist, 1, &histSize, &histRange, uniform, accumulate );

          int maxh = 0;
          int maxs = 0;
          int maxv = 0;

          for(int i = 0; i < 256; i++){
            if(i < 180 && h_hist.at<float>(i) > h_hist.at<float>(maxh)) {
              maxh = i;
            }
            if(s_hist.at<float>(i) > s_hist.at<float>(maxs)) {
              maxs = i;
            }
            if(v_hist.at<float>(i) > v_hist.at<float>(maxv)) {
              maxv = i;
            }
          }
          cv::Mat hsvpx(1,1, CV_8UC3, cv::Scalar(maxh, maxs, maxv));
          cv::Mat bgrpx;
          cv::cvtColor(hsvpx, bgrpx, CV_HSV2BGR);

          cv::Vec3b color = bgrpx.at<cv::Vec3b>(0,0);
          cv::rectangle(image, rect, cv::Scalar(color[0],color[1],color[2]), CV_FILLED);
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

NAN_METHOD(ImgProc::MaskText) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(1, cb);

  cv::Mat image = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat.clone();
  
  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncMaskText(callback, image));
  return;
}