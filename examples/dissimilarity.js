'use strict'
var cv = require('../lib/opencv');
var ransac = require('node-ransac');
var fit = require('./linear-fit-2d');
var Promise = require('bluebird');
//good

var matches = [
["t3_46bqfs.jpg", "t3_46bqf0.jpg"],
["t3_46bq57.jpg", "t3_46bq3u.jpg"],
["t3_46br5y.jpg", "t3_46br4p.jpg"],
["t3_46bu44.jpg", "t3_46bu32.jpg"],
["t3_46bp97.jpg", "t3_46bp8b.jpg"],
["t3_46bp6w.jpg", "t3_46bp5u.jpg"],
["t3_46bs3a.jpg", "t3_46bs1w.jpg"],
["t3_46bsju.jpg", "t3_46bsia.jpg"],
["t3_46bpcd.jpg", "t3_46bpbe.jpg"],
["t3_46bsop.jpg", "t3_46bsmq.jpg"],
["t3_46bqua.jpg", "t3_46bqta.jpg"],
["t3_46bpuz.jpg", "t3_46bpsy.jpg"],
["t3_46bt50.jpg", "t3_46bt45.jpg"],
["t3_46brk8.jpg", "t3_46brit.jpg"],
["t3_46bqon.jpg", "t3_46bqnf.jpg"],
["t3_46brun.jpg", "t3_46bpbe.jpg"],
["t3_46bpw7.jpg", "t3_46bpsy.jpg"],
["t3_46bps1.jpg", "t3_46bpqn.jpg"],
["t3_46bps1.jpg", "t3_46bpp3.jpg"],
["t3_46bpuz.jpg", "t3_46bptz.jpg"],
["t3_46bpw7.jpg", "t3_46bpuz.jpg"],
["t3_46brun.jpg", "t3_46bpcd.jpg"],
["t3_46brp9.jpg", "t3_46bro2.jpg"],
["t3_46bpw7.jpg", "t3_46bptz.jpg"],
["t3_46bpqn.jpg", "t3_46bpp3.jpg"],
["t3_46brun.jpg", "t3_46bpad.jpg"],
["t3_46bpbe.jpg", "t3_46bpad.jpg"],
["t3_46bpcd.jpg", "t3_46bpad.jpg"],
["t3_46bplz.jpg", "t3_46bpkw.jpg"],
["t3_46bsxf.jpg", "t3_46bsia.jpg"],
["t3_46bsxf.jpg", "t3_46bsju.jpg"],
["t3_46bptz.jpg", "t3_46bpsy.jpg"]

]

//bad
/*
var matches = [ 
["t3_46br5y.jpg",	 "t3_46bqzk.jpg"],
["t3_46bugz.jpg",	 "t3_46bqzk.jpg"],
["t3_46bugz.jpg",	 "t3_46br4p.jpg"],
["t3_46bugz.jpg",	 "t3_46btex.jpg"],
["t3_46btex.jpg",	 "t3_46br5y.jpg"],
["t3_46bugz.jpg",	 "t3_46brp9.jpg"],
["t3_46br4p.jpg",	 "t3_46bpbe.jpg"],
["t3_46brp9.jpg",	 "t3_46br5y.jpg"],
["t3_46bugz.jpg",	 "t3_46bthc.jpg"],
["t3_46br4p.jpg",	 "t3_46bqnf.jpg"],
["t3_46br4p.jpg",	 "t3_46bpcd.jpg"],
["t3_46br4p.jpg",	 "t3_46bqzk.jpg"],
["t3_46brp9.jpg",	 "t3_46br4p.jpg"],
["t3_46btex.jpg",	 "t3_46br4p.jpg"],
["t3_46bthc.jpg",	 "t3_46br4p.jpg"],
["t3_46br4p.jpg",	 "t3_46bpsy.jpg"],
["t3_46btb8.jpg",	 "t3_46br4p.jpg"],
["t3_46bro2.jpg",	 "t3_46br4p.jpg"],
["t3_46bu32.jpg",	 "t3_46br4p.jpg"],
["t3_46btg2.jpg",	 "t3_46br4p.jpg"],
["t3_46brk8.jpg",	 "t3_46br4p.jpg"],
["t3_46bt50.jpg",	 "t3_46br4p.jpg"],
["t3_46bsxf.jpg",	 "t3_46br4p.jpg"],
["t3_46brxf.jpg",	 "t3_46bpw7.jpg"],
["t3_46br4p.jpg",	 "t3_46bpqn.jpg"],
["t3_46br4p.jpg",	 "t3_46bqon.jpg"],
["t3_46bugz.jpg",	 "t3_46bu32.jpg"],
["t3_46br4p.jpg",	 "t3_46bp8b.jpg"],
["t3_46bpuz.jpg",	 "t3_46bp97.jpg"],
["t3_46bpuz.jpg",	 "t3_46bpbe.jpg"],
["t3_46brp9.jpg",	 "t3_46bpuz.jpg"],
["t3_46bpw7.jpg",	 "t3_46bp97.jpg"],
["t3_46br4p.jpg",	 "t3_46bp5u.jpg"],
["t3_46bsmq.jpg",	 "t3_46bsia.jpg"],
["t3_46brp9.jpg",	 "t3_46bpw7.jpg"],
["t3_46bsmq.jpg",	 "t3_46bpw7.jpg"],
["t3_46br4p.jpg",	 "t3_46bpw7.jpg"],
["t3_46bthc.jpg",	 "t3_46bsmq.jpg"],
["t3_46bpw7.jpg",	 "t3_46bpbe.jpg"],
["t3_46brxf.jpg",	 "t3_46br4p.jpg"],
["t3_46bsmq.jpg",	 "t3_46bqzk.jpg"],
["t3_46bpw7.jpg",	 "t3_46bpqn.jpg"],
["t3_46bsju.jpg",	 "t3_46br4p.jpg"],
["t3_46bsmq.jpg",	 "t3_46bpsy.jpg"],
["t3_46bsmq.jpg",	 "t3_46bpuz.jpg"],
["t3_46brp9.jpg",	 "t3_46bpbe.jpg"],
["t3_46brun.jpg",	 "t3_46br4p.jpg"],
["t3_46bsmq.jpg",	 "t3_46bpcd.jpg"],
["t3_46bqyf.jpg",	 "t3_46bqua.jpg"],
["t3_46bsmq.jpg",	 "t3_46bpbe.jpg"]

]*/



//var comparison = matches[9];
for(let comparison of matches) {
cv.readImage("../examples/files/"+comparison[0], function(err, car1) {
  if (err) throw err;

  cv.readImage("../examples/files/"+comparison[1], function(err, car2) {
    if (err) throw err;
    //console.log("images read");

    cv.ImageSimilarity(car1, car2, 5, function (err, img, d_g, n_g, d_h, n_h, cond, d_p, n_p, matches, good_matches) {
      
      cv.ImageSimilarity(car2, car1, 5, function (err2, img2, d_g2, n_g2, d_h2, n_h2, cond2, d_p2, n_p2, matches2, good_matches2) {
        img.save("./tmp/"+comparison[0]+"_"+comparison[1]+"comparison.png");
        img2.save("./tmp/"+comparison[1]+"_"+comparison[0]+"comparison.png");
        //console.log(comparison[0], comparison[1], good_matches.length, good_matches2.length);
        
        Promise.props({
          good: ransac(fit, good_matches, 3, 0.999, 0.3, 5),
          all:  ransac(fit, matches,      3, 0.999, 0.7, 10),
          good2: ransac(fit, good_matches2, 3, 0.999, 0.3, 5),
          all2:  ransac(fit, matches2,      3, 0.999, 0.7, 10)
        }).then(function(res){
          res.good.dis = res.good.inliers.reduce((a,b)=>{return a+b.d},0) / res.good.inliers.length;
          res.all.dis =  res.all.inliers.reduce((a,b)=>{return a+b.d},0) / res.all.inliers.length;
          res.good2.dis = res.good2.inliers.reduce((a,b)=>{return a+b.d},0) / res.good2.inliers.length;
          res.all2.dis =  res.all2.inliers.reduce((a,b)=>{return a+b.d},0) / res.all2.inliers.length;
          
          console.log("text:", comparison[0], comparison[1], res.good.quality, res.good.model.x.a/res.good.model.y.a  - 1, (res.good.model.x.a+res.good.model.y.a)/2, res.good.dis, res.good.inliers.length, res.all.quality, res.all.model.x.a/res.all.model.y.a  - 1, (res.all.model.x.a+res.all.model.y.a)/2, res.all.dis, res.all.inliers.length);
          console.log("text:", comparison[1], comparison[0], res.good2.quality, res.good2.model.x.a/res.good2.model.y.a  - 1, (res.good2.model.x.a+res.good.model.y.a)/2, res.good2.dis, res.good2.inliers.length, res.all.quality, res.all2.model.x.a/res.all2.model.y.a  - 1, (res.all2.model.x.a+res.all2.model.y.a)/2, res.all2.dis, res.all2.inliers.length);
        })
        
        cv.MaskText(car1,   5, 5, 2, 2, .55, 5, 5, .6, 190.0, 2, 2, 10, .5, function(err, masked1, mid1){
          cv.MaskText(car2, 5, 5, 2, 2, .55, 5, 5, .6, 190.0, 2, 2, 10, .5, function(err, masked2, mid2){
            //console.log("images masked")
            
            cv.ImageSimilarity(masked1, masked2, 3, function (err, img, d_g, n_g, d_h, n_h, cond, d_p, n_p, matches, good_matches) {
              cv.ImageSimilarity(masked2, masked1, 3, function (err2, img2, d_g2, n_g2, d_h2, n_h2, cond2, d_p2, n_p2, matches2, good_matches2) {
                //console.log(comparison[0], comparison[1], good_matches.length, good_matches2.length);
                img.save("./tmp/"+comparison[0]+"_"+comparison[1]+"comparison_masked.png");
                img2.save("./tmp/"+comparison[1]+"_"+comparison[0]+"comparison_masked.png");
                
              });
              
              return Promise.props({
                good: ransac(fit, good_matches, 3, 0.999, 0.3, 5),
                all:  ransac(fit, matches,      3, 0.999, 0.7, 10),
                good2: ransac(fit, good_matches2, 3, 0.999, 0.3, 5),
                all2:  ransac(fit, matches2,      3, 0.999, 0.7, 10)
              }).then(function(res){
                res.good.dis = res.good.inliers.reduce((a,b)=>{return a+b.d},0) / res.good.inliers.length;
                res.all.dis =  res.all.inliers.reduce((a,b)=>{return a+b.d},0) / res.all.inliers.length;
                res.good2.dis = res.good2.inliers.reduce((a,b)=>{return a+b.d},0) / res.good2.inliers.length;
                res.all2.dis =  res.all2.inliers.reduce((a,b)=>{return a+b.d},0) / res.all2.inliers.length;
                
                console.log("mask:", comparison[0], comparison[1], res.good.quality, res.good.model.x.a/res.good.model.y.a  - 1, (res.good.model.x.a+res.good.model.y.a)/2, res.good.dis, res.good.inliers.length, res.all.quality, res.all.model.x.a/res.all.model.y.a  - 1, (res.all.model.x.a+res.all.model.y.a)/2, res.all.dis, res.all.inliers.length);
                console.log("mask:", comparison[1], comparison[0], res.good2.quality, res.good2.model.x.a/res.good2.model.y.a  - 1, (res.good2.model.x.a+res.good.model.y.a)/2, res.good2.dis, res.good2.inliers.length, res.all.quality, res.all2.model.x.a/res.all2.model.y.a  - 1, (res.all2.model.x.a+res.all2.model.y.a)/2, res.all2.dis, res.all2.inliers.length);
              })
            });
            
          });
        });
      });
    });
      
   
          

          /*cv.DetectAndCompute(masked1, function (err, results1){
            cv.DetectAndCompute(masked2, function (err, results2){
              cv.FilteredMatch(results1, results2, function (err, d_g, n_g, d_h, n_h, cond){
                console.log("old:",d_g, n_g, d_h, n_h, cond);
              
                });
              });
            });*/


  });

});
}