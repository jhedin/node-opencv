var cv = require('../lib/opencv');


cv.readImage("../examples/files/jQQAdlv.jpg", function(err, car1) {
  if (err) throw err;

  cv.readImage("../examples/files/PhrfypL.jpg", function(err, car2) {
    if (err) throw err;

    cv.ImageSimilarity(car1, car2, function (err, img, d_g, n_g, d_h, n_h) {
      if (err) throw err;

      console.log(d_g, n_g, d_h, n_h);
      img.save("./comparison.png");
    });

    cv.MaskText(car1,5,5,3,3,.7,3,3, .2, 160.0, 1, 1, function(err, masked1){
      cv.MaskText(car2,5,5,3,3,.7,3,3, .2, 160.0, 1, 1, function(err, masked2){

        cv.ImageSimilarity(masked1, masked2, function (err, img, d_g, n_g, d_h, n_h) {
          if (err) throw err;

          console.log(d_g, n_g, d_h, n_h);
          img.save("./comparison_masked.png");
        });

        cv.DetectAndCompute(masked1, function (err, results1){
        	cv.DetectAndCompute(masked2, function (err, results2){
        		cv.FilteredMatch(results1, results2, function (err, d_g, n_g, d_h, n_h){
        			console.log(d_g, n_g, d_h, n_h);
              
        		});
        	});
        });

      });
    });


  });

});
