#include "OpenCV.h"

#if ((CV_MAJOR_VERSION >= 2) && (CV_MINOR_VERSION >=4))

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

class Features: public Nan::ObjectWrap {
public:
  static Nan::Persistent<FunctionTemplate> constructor;
  static void Init(Local<Object> target);

  static NAN_METHOD(Similarity);
  static NAN_METHOD(DetectFeatures);
  static NAN_METHOD(Match);
  static NAN_METHOD(DrawFeatures);
  static NAN_METHOD(DrawMatches);
};

#endif
