//=============================================================================
//
// akaze_features.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Toshiba Research Europe Ltd (1)
//               TrueVision Solutions (2)
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file akaze_features.cpp
 * @brief Main program for detecting and computing binary descriptors in an
 * accelerated nonlinear scale space
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "./lib/AKAZE.h"

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <cuda_profiler_api.h>

using namespace std;
using namespace libAKAZECU;

/* ************************************************************************* */
/**
 * @brief This function parses the command line arguments for setting A-KAZE
 * parameters
 * @param options Structure that contains A-KAZE settings
 * @param img_path Path for the input image
 * @param kpts_path Path for the file where the keypoints where be stored
 */
int parse_input_options(AKAZEOptions& options, std::string& img_path,
                        std::string& kpts_path, int argc, char* argv[]);

/* ************************************************************************* */
int main(int argc, char* argv[]) {
  // Variables
  AKAZEOptions options;
  string img_path, kpts_path;

  // Variable for computation times.
  double t1 = 0.0, t2 = 0.0, tdet = 0.0, tdesc = 0.0;

  // Parse the input command line options
  if (parse_input_options(options, img_path, kpts_path, argc, argv)) return -1;

  if (options.verbosity) {
    cout << "Check AKAZE options:" << endl;
    cout << options << endl;
  }

  // Try to read the image and if necessary convert to grayscale.
  cv::Mat img = cv::imread(img_path.c_str(), 0);
  if (img.data == NULL) {
    cerr << "Error: cannot load image from file:" << endl
         << img_path << endl;
    return -1;
  }

  // Convert the image to float to extract features
  cv::Mat img_32;
  img.convertTo(img_32, CV_32F, 1.0 / 255.0, 0);

  // Don't forget to specify image dimensions in AKAZE's options
  options.img_width = img.cols;
  options.img_height = img.rows;

  // Extract features
  AKAZE evolution(options);
  vector<cv::KeyPoint> kpts;

  cudaProfilerStart();

  t1 = cv::getTickCount();
  evolution.Create_Nonlinear_Scale_Space(img_32);
  evolution.Feature_Detection(kpts);
  t2 = cv::getTickCount();
  tdet = 1000.0 * (t2 - t1) / cv::getTickFrequency();

  // Compute descriptors.
  cv::Mat desc;
  t1 = cv::getTickCount();
  evolution.Compute_Descriptors(kpts, desc);
  t2 = cv::getTickCount();
  tdesc = 1000.0 * (t2 - t1) / cv::getTickFrequency();

/*  for (int i=0; i<kpts.size(); ++i) {
      cv::KeyPoint &pt = kpts[i];
      if (pt.size < 0)
          std::cout << pt.pt.y << " " << pt.pt.x << std::endl;
	  }*/

  cudaProfilerStop();

  std::vector<std::vector<cv::DMatch> > dmatches;
  MatchDescriptors(desc, desc, dmatches);

  // opencv bruteforce macher
  std::vector<std::vector<cv::DMatch> > cv_matches;

  cv::Ptr<cv::DescriptorMatcher> matcher_l1 =
      cv::DescriptorMatcher::create("BruteForce-Hamming");
  // cv::BFMatcher matcher(cv::NORM_HAMMING);

  matcher_l1->knnMatch(desc, desc, cv_matches, 2);


/*  vector<int> fst_errs;
  vector<int> scd_errs;

  for (int i = 0; i < dmatches.size(); ++i) {
    if (dmatches[i][0].trainIdx != cv_matches[i][0].trainIdx ||
        dmatches[i][0].queryIdx != cv_matches[i][0].queryIdx ||
        dmatches[i][0].distance != cv_matches[i][0].distance) {
      fst_errs.push_back(i);

      std::cout << "cuda " << dmatches[i][0].trainIdx << " "
                << dmatches[i][0].queryIdx << " " << dmatches[i][0].distance
                << " " << dmatches[i][1].trainIdx << " "
                << dmatches[i][1].queryIdx << " " << dmatches[i][1].distance
                << "\n"
                << "cv " << cv_matches[i][0].trainIdx << " "
                << cv_matches[i][0].queryIdx << " " << cv_matches[i][0].distance
                << " " << cv_matches[i][1].trainIdx << " "
                << cv_matches[i][1].queryIdx << " " << cv_matches[i][1].distance
                << " first \n";
    }
    else if( dmatches[i][1].trainIdx != cv_matches[i][1].trainIdx ||
            dmatches[i][1].queryIdx != cv_matches[i][1].queryIdx ||
            dmatches[i][1].distance != cv_matches[i][1].distance)
    {
        scd_errs.push_back(i);
        std::cout << "cuda " << dmatches[i][0].trainIdx << " "
                  << dmatches[i][0].queryIdx << " " << dmatches[i][0].distance
                  << " " << dmatches[i][1].trainIdx << " "
                  << dmatches[i][1].queryIdx << " " << dmatches[i][1].distance
                  << "\n"
                  << "cv " << cv_matches[i][0].trainIdx << " "
                  << cv_matches[i][0].queryIdx << " " << cv_matches[i][0].distance
                  << " " << cv_matches[i][1].trainIdx << " "
                  << cv_matches[i][1].queryIdx << " " << cv_matches[i][1].distance
                  << " second \n";
    }
  }

  int fst_count = fst_errs.size();
  int scd_count = scd_errs.size();

  std::cout << "Total num of errors best " << fst_count << " errors second " << scd_count << "\n" << std::endl;

  for (int j = 0; j < desc.cols; ++j) {
    std::cout << static_cast<int>(
                     desc.at<uchar>(dmatches[fst_errs[fst_count-1]][0].trainIdx, j))
              << "," << static_cast<int>(desc.at<uchar>(
                            dmatches[fst_errs[fst_count-1]][0].queryIdx, j)) << " ";
  }
  std::cout << "\n";*/

  if (true) {
    // Summarize the computation times.
    evolution.Show_Computation_Times();

    cout << "Number of points: " << kpts.size() << endl;
    cout << "Time Detector: " << tdet << " ms" << endl;
    cout << "Time Descriptor: " << tdesc << " ms" << endl;

    cv::Mat img_rgb = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3);
    cvtColor(img,img_rgb, cv::COLOR_GRAY2BGR);
    draw_keypoints(img_rgb, kpts);

    cv::namedWindow("A-KAZE", cv::WINDOW_AUTOSIZE);
    cv::imshow("A-KAZE", img_rgb);
    cv::waitKey(0);
  }

  // Save keypoints in ASCII format
  if (!kpts_path.empty()) save_keypoints(kpts_path, kpts, desc, true);

  //  }
}

/* ************************************************************************* */
int parse_input_options(AKAZEOptions& options, std::string& img_path,
                        std::string& kpts_path, int argc, char* argv[]) {
  // If there is only one argument return
  if (argc == 1) {
    show_input_options_help(0);
    return -1;
  }
  // Set the options from the command line
  else if (argc >= 2) {
    options = AKAZEOptions();
    kpts_path = "./keypoints.txt";

    if (!strcmp(argv[1], "--help")) {
      show_input_options_help(0);
      return -1;
    }

    img_path = argv[1];

    for (int i = 2; i < argc; i++) {
      if (!strcmp(argv[i], "--soffset")) {
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else {
          options.soffset = atof(argv[i]);
        }
      } else if (!strcmp(argv[i], "--omax")) {
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else {
          options.omax = atof(argv[i]);
        }
      } else if (!strcmp(argv[i], "--dthreshold")) {
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else {
          options.dthreshold = atof(argv[i]);
        }
      } else if (!strcmp(argv[i], "--sderivatives")) {
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else {
          options.sderivatives = atof(argv[i]);
        }
      } else if (!strcmp(argv[i], "--nsublevels")) {
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else
          options.nsublevels = atoi(argv[i]);
      } else if (!strcmp(argv[i], "--diffusivity")) {
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else
          options.diffusivity = DIFFUSIVITY_TYPE(atoi(argv[i]));
      } else if (!strcmp(argv[i], "--descriptor")) {
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else {
          options.descriptor = DESCRIPTOR_TYPE(atoi(argv[i]));

          if (options.descriptor < 0 || options.descriptor > MLDB) {
            options.descriptor = MLDB;
          }
        }
      } else if (!strcmp(argv[i], "--descriptor_channels")) {
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else {
          options.descriptor_channels = atoi(argv[i]);

          if (options.descriptor_channels <= 0 ||
              options.descriptor_channels > 3) {
            options.descriptor_channels = 3;
          }
        }
      } else if (!strcmp(argv[i], "--descriptor_size")) {
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else {
          options.descriptor_size = atoi(argv[i]);

          if (options.descriptor_size < 0) {
            options.descriptor_size = 0;
          }
        }
      } else if (!strcmp(argv[i], "--save_scale_space")) {
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else {
          options.save_scale_space = (bool)atoi(argv[i]);
        }
      } else if (!strcmp(argv[i], "--show_results")) {
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else {
          // options.show_results = (bool)atoi(argv[i]);
        }
      } else if (!strcmp(argv[i], "--verbose")) {
        options.verbosity = true;
      } else if (!strcmp(argv[i], "--output")) {
        options.save_keypoints = true;
        i = i + 1;
        if (i >= argc) {
          cerr << "Error introducing input options!!" << endl;
          return -1;
        } else
          kpts_path = argv[i];
      }
    }
  }

  return 0;
}
