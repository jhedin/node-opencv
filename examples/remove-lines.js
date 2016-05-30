var cv = require('../lib/opencv');

// Load the image
cv.readImage('./files/t3_46bpsy.jpg', function(err, im) {
  if (err) {
    throw err;
  }
  if (im.width() < 1 || im.height() < 1) {
    throw new Error('Image has no size');
  }
  var orig = im.clone();

  im.cvtColor('CV_BGR2GRAY');
  var bw = im.adaptiveThreshold(255, 0, 0, 15, 2);
  bw.bitwiseNot(bw);

  var vertical = bw.clone();

  var verticalsize = Math.max(vertical.size()[0] / 30,30);
  var verticalStructure = cv.imgproc.getStructuringElement(1, [1, verticalsize]);

  // Apply morphology operations
  vertical.erode(1, verticalStructure);
  vertical.dilate(1, verticalStructure);

  vertical.bitwiseNot(vertical);
  vertical.bitwiseNot(vertical);
  //vertical.gaussianBlur([3, 3]);

  // Save output image
  vertical.save('./tmp/vertical.png');
  
  
  var horizontal = bw.clone();

  var horizontalsize = Math.max(vertical.size()[1] / 30, 30);
  var horizontalStructure = cv.imgproc.getStructuringElement(1, [horizontalsize, 1]);

  // Apply morphology operations
  horizontal.erode(1, horizontalStructure);
  horizontal.dilate(1, horizontalStructure);

  horizontal.bitwiseNot(horizontal);
  horizontal.bitwiseNot(horizontal);
  //horizontal.gaussianBlur([3, 3]);

  // Save output image
  horizontal.save('./tmp/horizntal.png');
  
  var both = bw.clone();
  both.addWeighted(vertical, 1, horizontal, 1);
  both.save('./tmp/both.png');
  both.cvtColor('CV_GRAY2BGR');
  var rem = orig.clone();
  rem.addWeighted(orig, 1, both, 0.5);
  rem.save('./tmp/removed.png');
  
});
