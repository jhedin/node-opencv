var cv = require('../lib/opencv');

cv.readImage("../examples/files/car1.jpg", function(err, car1) {
  if (err) throw err;

  cv.readImage("../examples/files/car2.jpg", function(err, car2) {
    if (err) throw err;

    cv.ImageSimilarity(car1, car2, function (err, img, d_g, n_g, d_h, n_h) {
      if (err) throw err;
      /*
      console.log(d_g, n_g);
      console.log(d_h, n_h);*/
      img.save("./comparison.png");
    });
  });

});
