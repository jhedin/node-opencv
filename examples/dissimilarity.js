'use strict'
var cv = require('../lib/opencv');
var ransac = require('node-ransac');
var fit = require('ransac-linear-fit-2d');
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

matches = [
["t3_46bt50.jpg", "t3_46bt45.jpg"],
["t3_46brun.jpg",	 "t3_46bpad.jpg"],
["t3_46bpcd.jpg",	 "t3_46bpad.jpg"],
["t3_46bpbe.jpg",	 "t3_46bpad.jpg"],
["t3_46bptz.jpg",	 "t3_46bpsy.jpg"],
["t3_46brun.jpg",	 "t3_46bpbe.jpg"],
["t3_46brp9.jpg",	 "t3_46bro2.jpg"],
["t3_46bpuz.jpg",	 "t3_46bpsy.jpg"],
["t3_46bsxf.jpg", "t3_46bsia.jpg"],
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

  
Promise.each(matches, function(comparison){
  
cv.readImage("../examples/files/"+comparison[0], function(err, car1) {
  if (err) throw err;

  cv.readImage("../examples/files/"+comparison[1], function(err, car2) {
    if (err) throw err;
    //console.log("images read");

    cv.ImageSimilarity(car1, car2, function (err, img, matches) {
      
        img.save("./tmp/"+comparison[0]+"_"+comparison[1]+"comparison.png");
        
        Promise.props({     
          all:  ransac(fit, matches,        5, 0.99999, .95  , 10),
        }).then(function(res){
         
          res.all.dis =  res.all.inliers.reduce((a,b)=>{return a+b.d},0) / res.all.inliers.length;
          var aspect = Math.max(res.all.model.x.a/res.all.model.y.a, res.all.model.y.a/res.all.model.x.a);
          var textMatch = false;
          if(res.all.quality < 0.5, aspect < 1.6, aspect > .7, (res.all.model.x.a+res.all.model.y.a)/2 < 10, (res.all.model.x.a+res.all.model.y.a)/2 > 0.1, res.all.dis < 64){
            textMatch = true;
          }
          
          //console.log("text:", comparison[0], comparison[1], matches.length, res.all.quality, res.all.model.x.a/res.all.model.y.a , (res.all.model.x.a+res.all.model.y.a)/2, res.all.dis, res.all.inliers.length);
                    
          cv.DetectFeatures(car1, function (err, results1){
            cv.DetectFeatures(car2, function (err, results2){
              cv.Match(results1, results2, function (err, matches){
                
                  return Promise.props({
                    all:  ransac(fit, matches,        5, 0.99999, .93, 10),
                  }).then(function(res){
                  
                    res.all.dis =  res.all.inliers.reduce((a,b)=>{return a+b.d},0) / res.all.inliers.length;
                    
                    //console.log("sepr:", comparison[0], comparison[1], matches.length, res.all.quality, res.all.model.x.a/res.all.model.y.a , (res.all.model.x.a+res.all.model.y.a)/2, res.all.dis, res.all.inliers.length);
                                       
                    cv.DrawMatches(car1, results1, car2, results2, res.all.inliers, function(err, match_img){
                      match_img.save("./tmp/"+comparison[0]+"_"+comparison[1]+"comparison_fn.png");
                    });
                    
                   
                  })
              
              });
            });
          });        
        
        cv.imgproc.MaskText(car1, function(err, masked1){
          cv.imgproc.MaskText(car2, function(err, masked2){
            //console.log("images masked")
            
            cv.ImageSimilarity(masked1, masked2, function (err, img, m_matches) {
                //console.log(comparison[0], comparison[1], good_matches.length, good_matches2.length);
                img.save("./tmp/"+comparison[0]+"_"+comparison[1]+"comparison_masked.png");

                return Promise.props({
                  all:  ransac(fit, m_matches,        5, 0.99999, .95, 10),
                }).then(function(res){
                
                  res.all.dis =  res.all.inliers.reduce((a,b)=>{return a+b.d},0) / res.all.inliers.length;
                  var aspect = Math.max(res.all.model.x.a/res.all.model.y.a, res.all.model.y.a/res.all.model.x.a);
                  var maskMatch = false;
                  if(res.all.quality < 0.5, aspect < 1.4, aspect > .6, (res.all.model.x.a+res.all.model.y.a)/2 < 10, (res.all.model.x.a+res.all.model.y.a)/2 > 0.1, res.all.dis < 64){
                    maskMatch = true;
                  }
                  console.log(comparison[0], comparison[1], textMatch, maskMatch, textMatch || maskMatch, textMatch && maskMatch)         
                  //console.log("mask:", comparison[0], comparison[1], m_matches.length, res.all.quality, res.all.model.x.a/res.all.model.y.a , (res.all.model.x.a+res.all.model.y.a)/2, res.all.dis, res.all.inliers.length);
                                    
                  cv.DetectFeatures(masked1, function (err, results1){
                    cv.DetectFeatures(masked2, function (err, results2){
                      cv.Match(results1, results2, function (err, matches){
                        
                          return Promise.props({
                            all:  ransac(fit, matches,        5, 0.99999, .93 , 10),
                           
                          }).then(function(res){
                          
                            res.all.dis =  res.all.inliers.reduce((a,b)=>{return a+b.d},0) / res.all.inliers.length;            
                            
                            //console.log("spmk:", comparison[0], comparison[1], matches.length, res.all.quality, res.all.model.x.a/res.all.model.y.a , (res.all.model.x.a+res.all.model.y.a)/2, res.all.dis, res.all.inliers.length);
                                                       
                            cv.DrawMatches(masked1, results1, masked2, results2, res.all.inliers, function(err, match_img){
                              match_img.save("./tmp/"+comparison[0]+"_"+comparison[1]+"comparison_masked_fn.png");
                            });
                            
                          })
                      
                        });
                      });
                  });
          
              })
            });
            
          });
        });
      });
    });
      
  });

});
return Promise.delay(10);
},{concurrency:1})