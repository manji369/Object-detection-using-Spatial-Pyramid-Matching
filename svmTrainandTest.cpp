#include <opencv2/ml/ml.hpp>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace cv;
using namespace std;
using namespace cv::ml;

template <typename T>
  std::string NumberToString ( T Number )
  {
     std::ostringstream ss;
     ss << Number;
     return ss.str();
  }


#define SIFT_or_ORB 0 // Set 0 for SIFT and 1 for ORB


int main(){

  //Declaring and Initializing required variables
  int k, l = 0;
  cv::Size s;
  Mat kernel(1, 4200, CV_32F);
  int labels[120];
  Mat trainingDataMat(120, 4200, CV_32F);
  string descriptorType = "SIFT";
  if (SIFT_or_ORB == 1){
    descriptorType = "ORB";
  }
	FileStorage fs("./" + descriptorType + "/kernelTrainAndTest.yml", FileStorage::READ);

//Preparing the trainingDataMatrix with first 30 images from each class
  for(int i = 1; i <= 4; i++){
    for(int j = 1; j <= 30; j++){
      k = j + 50*(i-1);
      fs["Kernel"+NumberToString(k)] >> kernel;
      if(i == 3){
    }
      kernel.row(0).copyTo(trainingDataMat.row(l));
      labels[l] = i;
      l++;
    }
  }
	fs.release();
  Mat labelsMat(120, 1, CV_32S, labels);

  //Initializing svm
  Ptr<SVM> svm = SVM::create();
  svm->setType(SVM::C_SVC);
  svm->setKernel(SVM::RBF);
  //Train svm
  svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
  FileStorage fs1("./" + descriptorType + "/kernelTrainAndTest.yml", FileStorage::READ);
  int responseLabels[80];
  l = 0;

  //Predict using trained svm
  for(int i = 1; i <= 4; i++){
    for(int j = 1; j <= 20; j++){
      //31 to 50, 81 to 100, 131 to 150 and 181 to 200
      k = j + 50*(i-1) + 30;
      fs1["Kernel"+NumberToString(k)] >> kernel;
      responseLabels[l] = svm->predict(kernel);
      l++;
    }
  }
  fs1.release();

  // Getting accuracy
  int cnt = 0;
  for(int i = 0; i < 80; i++){
    //i/20+1 = 1 for i = 0 to 19, = 2 for i = 20 to 39..
    if(responseLabels[i] == ((i/20)+1)){
      cnt++;
    }
  }
  cout << "Accuracy of " + descriptorType + " features is " << cnt*100/80 << "%" << endl;
}
