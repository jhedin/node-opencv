var cv = require('../lib/opencv');

cv.readImage("../examples/files/car1.jpg", function(err, car1) {
  if (err) throw err;

  cv.readImage("../examples/files/car2.jpg", function(err, car2) {
    if (err) throw err;

    cv.ImageSimilarity(car1, car2, function (err, img, d_g, n_g, d_h, n_h) {
      if (err) throw err;

      console.log(d_g, n_g, d_h, n_h);
      img.save("./comparison.png");
    });

    cv.DetectAndCompute(car1, function (err, results1){
    	cv.DetectAndCompute(car2, function (err, results2){
    		cv.FilteredMatch(results1, results2, function (err, d_g, n_g, d_h, n_h){
    			console.log(d_g, n_g, d_h, n_h);
          
    		});
    	});
    });

  });

});
