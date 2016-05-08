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
matches = [['t3_46bplz.jpg', 't3_46bpkw.jpg']];
matches = [['t3_46bsxf.jpg', 't3_46bpkw.jpg']];


//var comparison = matches[9];
for(let comparison of matches) {
cv.readImage("../examples/files/"+comparison[0], function(err, car1) {
  if (err) throw err;

  cv.readImage("../examples/files/"+comparison[1], function(err, car2) {
    if (err) throw err;
    //console.log("images read");

    cv.ImageSimilarity(car1, car2, 5, function (err, img, d_g, n_g, d_h, n_h, cond, d_p, n_p, matches, good_matches) {
      if (err) throw err;
      /*for(var match of matches){
        console.log(match);
      }
      for(var match of good_matches){
        console.log(match);
      }*/
      RobustLineFitting(matches, 5, function(){}, function(i, n, bestInliers, lastModel, model){        
        var dis = 0;
        for(var inlier of bestInliers) {
          dis += matches[inlier].d;
        }
        dis = dis / bestInliers.length;
        console.log("inliers:", bestInliers);
        console.log(model, bestInliers.length, dis, i, n);
      }).run();
      RobustLineFitting(good_matches, 5, function(){}, function(i, n, bestInliers, lastModel, model){
        var dis = 0;
        for(var inlier of bestInliers) {
          dis += good_matches[inlier].d;
        }
        dis = dis / bestInliers.length;
        console.log("inliers:", bestInliers);
        console.log(model, bestInliers.length, dis, i, n);
      }).run();

      console.log("text:", comparison[0], comparison[1],d_g, n_g, d_h, n_h, cond, d_p, n_p);
      img.save("./"+comparison[0]+"_"+comparison[1]+"comparison.png");
    });
    cv.MaskText(car1,   5, 5, 2, 2, .55, 5, 5, .6, 190.0, 2, 2, 10, .5, function(err, masked1, mid1){
      cv.MaskText(car2, 5, 5, 2, 2, .55, 5, 5, .6, 190.0, 2, 2, 10, .5, function(err, masked2, mid2){
        //console.log("images masked")
        cv.ImageSimilarity(masked1, masked2, 3, function (err, img, d_g, n_g, d_h, n_h, cond, d_p, n_p, matches, good_matches) {
          if (err) throw err;
          
          /*for(var match of matches){
            console.log(match);
          }
          for(var match of good_matches){
            console.log(match);
          }*/

          RobustLineFitting(matches, 5, function(){}, function(i, n, bestInliers, lastModel, model){
            var dis = 0;
            for(var inlier of bestInliers) {
              dis += matches[inlier].d;
            }
            dis = dis / bestInliers.length;
            console.log("inliers:", bestInliers);
            console.log(model, bestInliers.length, dis, i, n);
          }).run();
          RobustLineFitting(good_matches, 5, function(){}, function(i, n, bestInliers, lastModel, model){
            var dis = 0;
            for(var inlier of bestInliers) {
              dis += good_matches[inlier].d;
            }
            dis = dis / bestInliers.length;
            console.log("inliers:", bestInliers);
            console.log(model, bestInliers.length, dis, i, n);
          }).run();


          console.log("new: ", comparison[0], comparison[1],d_g, n_g, d_h, n_h, cond, d_p, n_p);
          img.save("./"+comparison[0]+"_"+comparison[1]+"comparison_masked.png");

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

// http://www.visual-experiments.com/demo/ransac.js/

/*
  Copyright (c) 2010 ASTRE Henri (http://www.visual-experiments.com)

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

function RobustLineFitting(points, threshold, onUpdate, onComplete) {
  return new Ransac(new LineFitting(), points, threshold, onUpdate, onComplete);
}

function LineFitting() {

  this.Model = function(xa,xb,ya,yb) {
    this.x = {a: xa || 1, b: xb || 0};
    this.y = {a: ya || 1, b: yb || 0};
  }
  this.Model.prototype.copy = function(model) {
    this.x = {a: model.x.a, b: model.x.b};
    this.y = {a: model.y.a, b: model.y.b};
  }

  this.nbSampleNeeded = 2;  
  
  this.estimateModel = function(points, sample, model) {
    var counter = 0;
    for (var i in sample) {
      _samplePoints[counter] = points[i]; 
      counter++;
    }
    
    var s1 = _samplePoints[0];
    var s2 = _samplePoints[1];

    model.x = {};
    model.y = {};
    
    model.x.a = (s2.point2.x - s1.point2.x) / (s2.point1.x - s1.point1.x);
    model.x.b = s1.point2.x - model.x.a * s1.point1.x;

    model.y.a = (s2.point2.y - s1.point2.y) / (s2.point1.y - s1.point1.y);
    model.y.b = s1.point2.y - model.y.a * s1.point1.y;
  };
  
  this.estimateError = function(points, index, model) {

    var dx = Math.abs(points[index].point2.x - model.x.a * points[index].point1.x - model.x.b) / Math.sqrt(1 + model.x.a * model.x.a);
    var dy = Math.abs(points[index].point2.y - model.y.a * points[index].point1.y - model.y.b) / Math.sqrt(1 + model.y.a * model.y.a);

    return Math.sqrt(dx * dx + dy * dy);
  };

  // refine the solution by perpendicular linear regression on the inliers
  this.refine = function(points, inliers, model) {


    var refined = {x:{},y:{}};
    var Bx, By;
    var preprocess = [];
    var bar = {
      point1: {
        x:0,
        y:0
      },
      point2: {
        x:0,
        y:0
      }
    }
    var sum = {
      point1:{x:0,y:0},
      point2:{x:0,y:0},
      point1point1:{x:0,y:0},
      point2point2:{x:0,y:0},
      point1point2:{x:0,y:0}
    }
    var n = inliers.length;

    for(var inlier of inliers){
      preprocess.push({
        point1point1:{
          x: points[inlier].point1.x * points[inlier].point1.x, 
          y: points[inlier].point1.y * points[inlier].point1.y
        },
        point2point2:{
          x: points[inlier].point2.x * points[inlier].point2.x, 
          y: points[inlier].point2.y * points[inlier].point2.y
        },
        point1point2: {
          x: points[inlier].point1.x * points[inlier].point2.x, 
          y: points[inlier].point1.y * points[inlier].point2.y
        }
      });
    }

    for(var i = 0; i < n; i++) {
      sum.point1.x += points[inlier].point1.x;
      sum.point1.y += points[inlier].point1.y;

      sum.point2.x += points[inlier].point2.x;
      sum.point2.y += points[inlier].point2.y;

      sum.point1point1.x += preprocess[i].point1point1.x;
      sum.point1point1.y += preprocess[i].point1point1.y;

      sum.point2point2.x += preprocess[i].point2point2.x;
      sum.point2point2.y += preprocess[i].point2point2.y;
      sum.point1point2.x += preprocess[i].point1point2.x;
      sum.point1point2.y += preprocess[i].point1point2.y;
    }

    bar.point1.x = sum.point1.x / n;
    bar.point1.y = sum.point1.y / n;

    bar.point2.x = sum.point2.x / n;
    bar.point2.y = sum.point2.y / n;
    
    // use the model slope t inform whether it's positive or negative
    Bx = (1/2) * ((sum.point2.x - n * sum.point2point2.x)-(sum.point1.x - n * sum.point1point1.x))/(n*sum.point1.x*sum.point2.x - sum.point1point2.x);
    if(model.x.a > 0) {
      refined.x.a = -Bx + Math.sqrt(Bx*Bx + 1);
    }
    else {
      refined.x.a = -Bx - Math.sqrt(Bx*Bx + 1);
    }
    refined.x.b = (sum.point2.x - refined.x.a * sum.point1.x) / n;

    By = (1/2) * ((sum.point2.y - n * sum.point2point2.y)-(sum.point1.y - n * sum.point1point1.y))/(n*sum.point1.y*sum.point2.y - sum.point1point2.y);
    if(model.y.a > 0) {
      refined.y.a = -By + Math.sqrt(By*By + 1);
    }
    else {
      refined.y.a = -By - Math.sqrt(By*By + 1);
    }
    refined.y.b = (sum.point2.y - refined.y.a * sum.point1.y) / n;

    return refined;
  }
  
  var _samplePoints = new Array(this.nbSampleNeeded);
}

function Ransac(fittingProblem, points, threshold, onUpdate, onComplete) {

  var _points     = points;
  var _threshold  = threshold;
  var _onUpdate   = onUpdate;
  var _onComplete = onComplete;
  
  //var _random      = new Random();
  var _problem     = fittingProblem;  
  var _bestModel   = new fittingProblem.Model();
  var _bestInliers = {};
  var _bestScore   = 4294967295;
  
  var _currentInliers = [];
  var _currentModel   = new fittingProblem.Model();
  var _nbIters        = nbIterations(0.9999, 0.8, fittingProblem.nbSampleNeeded);
  
  var _iterationCounter = 0;
  var _iterationTimer;
  var _that = this; 
  var _combinations = new Set();
  var _maxCombinations = _nCr(fittingProblem.nbSampleNeeded, _points.length);

  function _nCr(k,n) {
    var max = Math.max(k, n - k);
    var result = 1;
    for (var i = 1; i <= n - max; i++) {
      result = result * (max + i) / i;
    }

    return result;
  }
  
  function nbIterations(ransacProba, outlierRatio, sampleSize) {
    return Math.ceil(Math.log(1-ransacProba) / Math.log(1-Math.pow(1-outlierRatio, sampleSize))) + _points.length;
  }
  
  function randomInt(min, max) {
    return Math.floor(Math.random()*(max - min + 1)) + min;
    //return Math.floor(_random.uniform(min, max + 1));
  }
  
  function randomSample(k, n, sample) {
    var nbInserted = 0;
    var sel = [];
    while (nbInserted < k) {
      var t = randomInt(0, n-1);
      if (sample[t] === undefined) {
        sample[t] = true;
        sel.push(t);
        nbInserted++;
      }
    }
    return sel;
  }
  
  this.run = function() {
    _that.stop();
    _iterationTimer = setInterval(_that.next, 10);
  };
  
  this.stop = function() {
    if (_iterationTimer) {
      clearInterval(_iterationTimer);
      _iterationTimer   = undefined;
    }
    _iterationCounter = 0;
    _bestModel   = new fittingProblem.Model();
    _bestInliers = {};
    _bestScore   = 4294967295;
  };
  
  this.next = function() {
    _currentInliers.length = 0;
    
    var sample = {};
    var sel;
    while(1) {
      sample = {};
      sel = randomSample(_problem.nbSampleNeeded, _points.length, sample);
      sel.sort(function(a,b){return a-b});
      if(!_combinations.has(sel))
        break;
      if(_combinations.size  >= _maxCombinations){
        _iterationCounter = 4294967295;
        break;
      }
    }
    _combinations.add(sel);
  
    _problem.estimateModel(_points, sample, _currentModel);
    
    var score = 0;
    for (var j=0; j<_points.length; ++j) {
      var err = _problem.estimateError(_points, j, _currentModel);
      if (err > _threshold) {
        score += _threshold;
      }
      else {
        score += err;
        _currentInliers.push(j);
      }
    }
    if (score < _bestScore) {
      _bestModel.copy(_currentModel);
      _bestInliers = _currentInliers;
      _bestScore   = score;
    }
    
    _onUpdate(_iterationCounter+1, _nbIters, _currentInliers, _currentModel, _bestModel);   
    
    _iterationCounter++;
    if (_iterationCounter >= _nbIters) {
      _onComplete(_iterationCounter, _nbIters, _bestInliers, _currentModel, fittingProblem.refine(_points, _bestInliers, _bestModel));
      _that.stop();
    }
  };
}