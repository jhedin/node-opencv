'use strict'
var cv = require('../lib/opencv');

var matches = [
['t3_46bsxf.jpg', 't3_46bsia.jpg'],
['t3_46brp9.jpg', 't3_46bro2.jpg'],
['t3_46bqon.jpg', 't3_46bqnf.jpg'],
['t3_46bsxf.jpg', 't3_46bsju.jpg'],
['t3_46bpw7.jpg', 't3_46bpsy.jpg'],
['t3_46bptz.jpg', 't3_46bpsy.jpg'],
['t3_46bplz.jpg', 't3_46bpkw.jpg'],
['t3_46bpcd.jpg', 't3_46bpbe.jpg'],
['t3_46bpcd.jpg', 't3_46bpad.jpg'],
['t3_46bpbe.jpg', 't3_46bpad.jpg']
];

matches = [
["t3_46bpuz.jpg","t3_46bptz.jpg"],
['t3_46bpw7.jpg', 't3_46bpsy.jpg'],
['t3_46bsxf.jpg', 't3_46bsju.jpg'],
['t3_46bsxf.jpg', 't3_46bsia.jpg'],
['t3_46brun.jpg', 't3_46bpbe.jpg'],
['t3_46brun.jpg', 't3_46bpad.jpg'],
['t3_46brp9.jpg', 't3_46bro2.jpg'],
['t3_46bqon.jpg', 't3_46bqnf.jpg'],
['t3_46bplz.jpg', 't3_46bpkw.jpg'],
['t3_46bpcd.jpg', 't3_46bpad.jpg']
];

matches = [['t3_46bsxf.jpg', 't3_46bsia.jpg']];

//var comparison = matches[9];
for(let comparison of matches) {
cv.readImage("../examples/files/"+comparison[0], function(err, car1) {
  if (err) throw err;

  cv.readImage("../examples/files/"+comparison[1], function(err, car2) {
    if (err) throw err;


    cv.ImageSimilarity(car1, car2, 5, function (err, img, d_g, n_g, d_h, n_h, cond, d_p, n_p) {
      if (err) throw err;

      console.log("text:", comparison[0], comparison[1],d_g, n_g, d_h, n_h, cond, d_p, n_p);
      img.save("./comparison.png");
    });
    cv.MaskText(car1,   5, 5, 2, 2, .55, 5, 5, .6, 190.0, 2, 2, 10, .5, function(err, masked1, mid1){
      cv.MaskText(car2, 5, 5, 2, 2, .55, 5, 5, .6, 190.0, 2, 2, 10, .5, function(err, masked2, mid2){
        cv.ImageSimilarity(masked1, masked2, 3, function (err, img, d_g, n_g, d_h, n_h, cond, d_p, n_p) {
          if (err) throw err;

          console.log("new: ", comparison[0], comparison[1],d_g, n_g, d_h, n_h, cond, d_p, n_p);
          img.save("./comparison_masked.png");

          /*cv.DetectAndCompute(masked1, function (err, results1){
            cv.DetectAndCompute(masked2, function (err, results2){
              cv.FilteredMatch(results1, results2, function (err, d_g, n_g, d_h, n_h, cond){
                console.log("old:",d_g, n_g, d_h, n_h, cond);
              
                });
              });
            });*/
        });

      });
    });


  });

});
}