#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/*
const int histSize = 256;

void drawHistogram(cv::Mat& hist) {

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0));

    cv::normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++) {
      cv::line(
          histImage,
          cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
          cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
          cv::Scalar(255, 0, 0), 2, 8, 0);
    }

    cv::namedWindow("calcHist Demo", cv::WINDOW_AUTOSIZE);
    cv::imshow("calcHist Demo", histImage);

}
*/

int main(int argc, char** argv) {

	string source_path = "./Dataset/game1_clip2/frames/frame_last.png";
	string mask_path = "./Dataset/game1_clip2/masks/frame_last.png";
	
	// Read image
	/*Mat im = imread(source_path, IMREAD_GRAYSCALE );
	imshow("im", im);*/
	
	Mat source = imread(source_path);
	imshow("source", source);
	
	Mat gs_src;
	cvtColor(source, gs_src, COLOR_BGR2GRAY);
	
	imshow("gs", gs_src);
	
	Mat kp_src = imread(source_path);
	
	Mat mask = imread(mask_path);
	Mat img_balls = imread(source_path);
	/*Mat mask_out;
	inRange(mask_pf, Scalar(0,0,1), Scalar(255,255,255), mask_out);
	imshow("mask_out", mask_out);*/
	
	for(int i = 0; i < source.rows; i++)
	{
		for(int j = 0; j < source.cols; j++)
		{
		    Vec3b s_p = source.at<Vec3b>(i, j);
		    Vec3b m_p = mask.at<Vec3b>(i, j);
		    
		    if((m_p.val[0] == 0 && m_p.val[1] == 0 && m_p.val[2] == 0) || (m_p.val[0] == 5 && m_p.val[1] == 5 && m_p.val[2] == 5)){
		    	img_balls.at<Vec3b>(i, j) = 0;
		    	gs_src.at<uchar>(i,j) = 0;
		    	kp_src.at<Vec3b>(i,j) = 0;
		    } else {
		    	/*if(m_p.val[0] >= 200 && m_p.val[1] >= 200 && m_p.val[2] >= 200){
		    		source.at<Vec3b>(i, j) = {150, 150, 150};
		    	}*/
		    	img_balls.at<Vec3b>(i, j) = {200, 200, 200};
		    }
		    
		    /*if(m_p.val[0] == 0 && m_p.val[1] == 0 && m_p.val[2] == 0){
		    	im.at<Vec3b>(i, j) = 0;
		    }*/
		}
	}
	
	imshow("gs_src", gs_src);
	imshow("kp_src", kp_src);
	
	Mat sharp;
	Mat sharpening_kernel = (Mat_<double>(3, 3) << -1, -1, -1,
		    -1, 9, -1,
		    -1, -1, -1);
	filter2D(img_balls, sharp, -1, sharpening_kernel);
    
	Mat grayscale;
	cvtColor(sharp, grayscale, COLOR_BGR2GRAY);
	imshow("Image Sharpening", grayscale);
    
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;
	 
	// Change thresholds
	params.minThreshold = 1;
	params.maxThreshold = 255;
	 
	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 10;
	 
	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.5;
	 
	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.75;
	 
	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.1;
	
	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	 
	// SimpleBlobDetector::create creates a smart pointer. 
	// So you need to use arrow ( ->) instead of dot ( . )
	// detector->detect( im, keypoints);
	
	// Set up the detector with default parameters.
	//SimpleBlobDetector detector;
	 
	// Detect blobs.
	std::vector<KeyPoint> keypoints;
	detector->detect( grayscale, keypoints);
	/*printf("Keypoints size: %d\n", (int) keypoints.size());
	printf("Keypoint 0 -> pt_x: %d, pt_y: %d, diameter: %d\n", (int) keypoints[0].pt.x, (int) keypoints[0].pt.y, (int) keypoints[0].size);*/
	
	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	Mat source_keypoints;
	Mat masked_keypoints;
	drawKeypoints( source, keypoints, source_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	drawKeypoints( grayscale, keypoints, masked_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	 
	// Show blobs
	imshow("source with keypoints", source_keypoints );
	imshow("masked source with keypoints", masked_keypoints );
	
	int maxX, maxY, minX, minY;
	Mat bbox = imread(source_path);
	
	for(KeyPoint k: keypoints){
		Point2f p = k.pt;
		
		maxX = (int) (p.x + (k.size / 2));
		maxY = (int) (p.y + (k.size / 2));
		minX = (int) (p.x - (k.size / 2));
		minY = (int) (p.y - (k.size / 2));
		
		//cout<<"center: "<<p.x<<", "<<p.y<<", minX:"<<minX<<", minY:"<<minY<<", maxX:"<<maxX<<", maxY:"<<maxY<<"\n"<<endl;
		
		rectangle( bbox, Point(minX,minY), Point(maxX, maxY), Scalar(0, 0, 255) );
		
	}
	
	imshow("Bounding boxes", bbox);
	
	Mat kp_img;
	int key;
	
	//for(KeyPoint k: keypoints){
	for(int i = 0; i < keypoints.size(); i++){
		KeyPoint k = keypoints[i];
		Point2f p = k.pt;
		
		maxX = (int) (p.x + (k.size / 2));
		maxY = (int) (p.y + (k.size / 2));
		minX = (int) (p.x - (k.size / 2));
		minY = (int) (p.y - (k.size / 2));
		//Grayscale Histogram
		kp_img = gs_src(Rect(minX, minY, k.size, k.size));
		imshow("kp", kp_img);
		
		// Number of bins
		int histSize = 256;

		// Set the ranges (for B,G,R) )
		float range[] = { 0, 256 };
		const float* histRange = { range };

		bool uniform = true;
		bool accumulate = false;

		Mat hist;

		// Compute the histogram
		calcHist(&kp_img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

		// Normalize the result to [0, histImage.rows]
		Mat histImage(400, 512, CV_8UC1, Scalar(0));

		normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		// Draw the histogram
		int binWidth = cvRound((double) histImage.cols / histSize);

		for (int i = 0; i < histSize; i++) {
		    rectangle(histImage,
		                  Point(binWidth * i, histImage.rows),
		                  Point(binWidth * (i + 1), histImage.rows - cvRound(hist.at<float>(i))),
		                  Scalar(255),
		                  FILLED);
		}

		// Display the histogram
		namedWindow("Grayscale Histogram", WINDOW_AUTOSIZE);
		imshow("Grayscale Histogram", histImage);

		//waitKey(0);
		
		/*
		//RGB Histogram
		kp_img = kp_src(Rect(minX, minY, k.size, k.size));
		imshow("kp", kp_img);
		vector<Mat> bgr_planes;
		split( kp_img, bgr_planes );
		int histSize = 256;
		float range[] = { 0, 256 }; //the upper boundary is exclusive
		const float* histRange[] = { range };
		bool uniform = true, accumulate = false;
		Mat b_hist, g_hist, r_hist;
		calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate );
		calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate );
		calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate );
		int hist_w = 768, hist_h = 400;
		int bin_w = cvRound( (double) hist_w/histSize );
		Mat histImage_r( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
		Mat histImage_g( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
		Mat histImage_b( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
		normalize(b_hist, b_hist, 0, histImage_b.rows, NORM_MINMAX, -1, Mat() );
		normalize(g_hist, g_hist, 0, histImage_g.rows, NORM_MINMAX, -1, Mat() );
		normalize(r_hist, r_hist, 0, histImage_r.rows, NORM_MINMAX, -1, Mat() );
		for( int i = 1; i < histSize; i++ )
		{
			rectangle(histImage_b,
				              Point(bin_w * i, histImage_b.rows),
				              Point(bin_w * (i + 1), histImage_b.rows - cvRound(b_hist.at<float>(i))),
				              Scalar(255, 0, 0),
				              FILLED);
			rectangle(histImage_g,
				              Point(bin_w * i, histImage_g.rows),
				              Point(bin_w * (i + 1), histImage_g.rows - cvRound(g_hist.at<float>(i))),
				              Scalar(0, 255, 0),
				              FILLED);
			rectangle(histImage_r,
				              Point(bin_w * i, histImage_r.rows),
				              Point(bin_w * (i + 1), histImage_r.rows - cvRound(r_hist.at<float>(i))),
				              Scalar(0, 0, 255),
				              FILLED);
			*//*line( histImage_b, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
			Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
			Scalar( 255, 0, 0), 2, 8, 0 );
			line( histImage_g, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
			Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
			Scalar( 0, 255, 0), 2, 8, 0 );
			line( histImage_r, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
			Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
			Scalar( 0, 0, 255), 2, 8, 0 );*//*
		}
		imshow("calcHist Red", histImage_r );
		imshow("calcHist Green", histImage_g );
		imshow("calcHist Blue", histImage_b );*/
		
		/*//OTSU THRESHOLD (USELESS)
		Mat bin;
		threshold(bgr_planes[2], bin, 0, 255, THRESH_OTSU);
		
		imshow("otsu red", bin);
		*/
		
		//Check if the user wants to go back to the previous one
		if(i == 0){
			key = waitKey(0);
			while(key == 81){
				printf("Can't go further back, type something else\n");
				key = waitKey(0);
			}
		} else if(key = waitKey(0) == 81){
			i -= 2;
		}
		
	}
	
	
	//waitKey(0);
	/*
	for(KeyPoint k:keypoints){
		Mat kp_img = Mat::zeros(Size(k.size,k.size) , CV_8UC1);
		Point2f p = k.pt;
		
		minX = (int) (p.x - (k.size / 2));
		minY = (int) (p.y - (k.size / 2));
		
		for(int i = 0; i < kp_img.rows; i++){
			for(int j = 0; j < kp_img.cols; j++){
				int pxl_pos_x = minX + i;
				int pxl_pos_y = minY + j;
				kp_img.at<int>(i, j) = gs_src.at<int>(pxl_pos_x, pxl_pos_y);
			}
		}
		imshow("kp zoom", kp_img);
		waitKey(0);
	}
	
	waitKey(0);
	*/
}