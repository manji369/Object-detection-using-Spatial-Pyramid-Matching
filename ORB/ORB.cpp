
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <sstream>
#include <cmath>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

#include <typeinfo>

#define DICTIONARY_BUILD 0 // set DICTIONARY_BUILD 1 to do Step 1, otherwise it goes to step 2

template <typename T>
  std::string NumberToString ( T Number )
  {
     std::ostringstream ss;
     ss << Number;
     return ss.str();
  }

int main()
{
int minHessian = 400;
#if DICTIONARY_BUILD == 1
// Build the dictionary using all the images.

	char * filename = new char[100];
	Mat input;

  // Get the keypoints and compute descriptors for all the images in the dataset.
	vector<KeyPoint> keypoints;
	Mat descriptor;
	Mat featuresUnclustered(0,0,CV_32F);
  Ptr<Feature2D> f2d = cv::ORB::create(minHessian);
	for(int cls=1;cls<=4;cls++){
		for(int imgs=1;imgs<=50;imgs++){
		sprintf(filename,"./../Data/%i (%i).jpg",cls, imgs);
    // cout << filename << endl;
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale
		f2d->detect(input, keypoints);
		f2d->compute(input, keypoints, descriptor);
		featuresUnclustered.push_back(descriptor);
		printf("class %i: %i percent done\n", cls, ((imgs)*2));
	}
	}

  cout << "Building dictionary" << endl;

  //Build the vocabulary by clustering the obtained descriptors and obtain 200 words.
	int dictionarySize=200;
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.01);
	int retries=1;
	int flags=KMEANS_PP_CENTERS;
	BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
  Mat featuresUnclusteredF(featuresUnclustered.rows,featuresUnclustered.cols,CV_32F);
	featuresUnclustered.convertTo(featuresUnclusteredF,CV_32F);
	Mat dictionary=bowTrainer.cluster(featuresUnclusteredF);
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

#else

// Load the vocabulary from file.
  Mat dictionaryF;
	FileStorage fs("dictionary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionaryF;
	fs.release();

  Mat dictionary(dictionaryF.rows,dictionaryF.cols,CV_8U);
	dictionaryF.convertTo(dictionary,CV_8U);

// Initialize the detector, matcher and DescriptorExtractor
  Ptr<Feature2D> f2d = cv::ORB::create(minHessian);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	Ptr<DescriptorExtractor> extractor = cv::ORB::create(minHessian);
	BOWImgDescriptorExtractor bowDE(extractor,matcher);
	bowDE.setVocabulary(dictionary);

	char * filename = new char[100];
	char * descTag = new char[10];

//Initialize and declare other variables
	vector<KeyPoint> keypoints;
	Mat bowDescriptor;
  Mat levelBowDescriptors[3];
  cv::Size s;
  cv::Size s1;
  int rows, cols, X, Y, width, height, k = 0, bias = 0;
  Mat croppedImage;
	Mat croppedBowDescriptor;
  Mat levelBowDescriptor;
  Mat kernels[100];
  FileStorage fs1("kernelTrainAndTest.yml", FileStorage::WRITE);

  //For all the classes and all the images
  for(int label = 1; label <= 4; label++){
    for(int imgNum = 1; imgNum <= 50; imgNum++){
      sprintf(filename,"./../Data/%i (%i).jpg", label, imgNum);
      cout << "getting kernel for:" << NumberToString(label) << " (" << NumberToString(imgNum) + ").jpg" << endl;
    	Mat img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
    	// imshow("original", img);
    	// cvWaitKey(27);

    	s = img.size();
    	rows = s.height;
    	cols = s.width;

      //Initialize coordinates and size of subimage
    	X = 0;
      Y = 0;
      width = cols/2;
      height = rows/2;

      //For level = 0 to 2
    	for(int j = 0; j <= 2; j++){
        //Set the width and height of subimage
    		width = cols/pow(2, j);
    		height = rows/pow(2, j);
        Mat levelBowDescriptor;
      	for(int i = 0; i < pow(2, 2*j); i++){
          //Set the left corner of subimage
      		X = (i%(int) pow(2, j))*width;
      		Y = (i/(int) pow(2, j))*height;
      		croppedImage = img(Rect(X, Y, width, height));
      		cv::Size s1 = croppedImage.size();
      		// cout << "Calculating L"+NumberToString(j)+" "+NumberToString(i) << endl;
      		// imshow("cropped", croppedImage);
      		// cvWaitKey(27);

      		keypoints.clear();
          //Detect and match the descriptor with vocabulary
      		f2d->detect(croppedImage, keypoints);
      		bowDE.compute(croppedImage,keypoints,croppedBowDescriptor);
          s1 = croppedBowDescriptor.size();
          // If there are no matches set the variable to a zero vector
      		if (s1.width == 0){
            croppedBowDescriptor = Mat(1,200, CV_32F, cvScalar(0.));
            s1 = croppedBowDescriptor.size();
          }
          //Obtaining all the features of different subimages into a single vector for a given level
          if(i == 0){
            levelBowDescriptor = croppedBowDescriptor;
          }else{
          hconcat(levelBowDescriptor, croppedBowDescriptor, levelBowDescriptor);
        }
        }
        //Feature vectors of each levels
      levelBowDescriptors[j] = levelBowDescriptor;
    }
    //Weight the features of level according to the equation given in paper.
      for(int i = 0; i <= 2; i++){
        levelBowDescriptor = levelBowDescriptors[i];
        float weight = ((float) pow(2, 2-i));
        levelBowDescriptor = levelBowDescriptor/weight;
        s = kernels[k-bias].size();
        if(s.width == 4200){
          kernels[k-bias].release();
        }
        s = kernels[k-bias].size();
        if(s.width == 0 && s.height == 0){
          kernels[k-bias] = levelBowDescriptor;
        }else{
        hconcat(kernels[k-bias], levelBowDescriptor, kernels[k-bias]);
      }
      }
      k++;
      // To avoid out of memory error, saving kernel for first 100 images
      if(k == 100){
        bias = 100;
        for(int i = 0; i < 100; i++){
          fs1 << "Kernel"+NumberToString(i+1) << kernels[i];
        }
      }
    }
}
// Finally saving kernels for other 100 images
  for(int i = 0; i < 100; i++){
    fs1 << "Kernel"+NumberToString(i+1+bias) << kernels[i];
  }
  fs1.release();

#endif
	printf("\nDone\n");
    return 0;
}
