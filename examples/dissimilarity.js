var cv = require('../lib/opencv');

console.log(cv.DetectAndCompute);

cv.readImage("../examples/files/car1.jpg", function(err, car1) {
  if (err) throw err;

  cv.readImage("../examples/files/car2.jpg", function(err, car2) {
    if (err) throw err;

    cv.ImageSimilarity(car1, car2, function (err, dissimilarity) {
      if (err) throw err;

      console.log('Dissimilarity: ', dissimilarity);
    });

    cv.DetectAndCompute(car1, function (err, results1){
    	cv.DetectAndCompute(car2, function (err, results2){
    		cv.FilteredMatch(results1,results2,function(err, d){
    			console.log('Dissimilarity: ', d);
    		});
    	});
    });

  });

});
