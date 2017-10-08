# Object detection using SIFT and ORB

Data taken form caltech 101 is inside the data folder.
Location of SIFT.cpp: ./SIFT/SIFT.cpp
Location of ORB.cpp: ./ORB/ORB.cpp
Kernels extracted and dictionaries built from each of the SIFT and ORB are stored in their
respective folders as kernelTrainAndTest.yml and dictionary.yml respectively.

Note: Building the dictionary takes much time for SIFT.

Note: Ignore the instructions for SIFT and ORB and jump to SVM if don't want to generate features or dictionary (because
they are already present in the files kernelTrainAndTest.yml and dictionary.yml)

Instructions for both SIFT and ORB are similar.
For SIFT:
1. Dictionary file (dictionary.yml) is available. If needed, to build the dictionary, set DICTIONARY_BUILD
inside SIFT.cpp to 1 (default 0).
2. In linux terminal, cd into the folder SIFT and execute:
g++ `pkg-config --cflags --libs opencv` SIFT.cpp -o SIFT
3. In linux terminal, execute:
./SIFT
4. If decided to build the dictionary in step 1, change DICTIONARY_BUILD to 0 and do steps 2 and 3 again.
This will generate the file kernelTrainAndTest.yml (which is present by default).

For ORB:
1. Dictionary file (dictionary.yml) is available. If needed, to build the dictionary, set DICTIONARY_BUILD
inside ORB.cpp to 1 (default 0).
2. In linux terminal, cd into the folder ORB and execute:
g++ `pkg-config --cflags --libs opencv` ORB.cpp -o ORB
3. In linux terminal, execute:
./ORB
4. If decided to build the dictionary in step 1, change DICTIONARY_BUILD to 0 and do steps 2 and 3 again.
This will generate the file kernelTrainAndTest.yml (which is present by default).

For SVM:
1. To get the accuracy for SIFT set SIFT_or_ORB inside svmTrainandTest.cpp to 0 and 1 for ORB.
2. In linux terminal, cd into the root folder of this project and execute:
g++ `pkg-config --cflags --libs opencv` svmTrainandTest.cpp -o svmTrainandTest
3. In linux terminal, execute:
./svmTrainandTest
This will display the accuracy.

References:
https://www.codeproject.com/articles/619039/bag-of-features-descriptor-on-sift-features-with-o
