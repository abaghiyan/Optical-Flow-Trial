#ifndef REALTIMEOPTICALFLOW_H
#define REALTIMEOPTICALFLOW_H

#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"

#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"

/** USAGE EXAMPLE **/
/**
 *	cv::VideoCapture cap("C:/Users/Improvis/Downloads/videoplayback.mp4");
 *	cv::Mat frame;
 *	cap >> frame;
 *	//cv::imwrite("fr1.png", frame);
 *
 *
 *	//cv::Point(118,217), cv::Point(113,221), cv::Point(111,224), cv::Point(112,235), cv::Point(117,240),
 *	//cv::Point(124,244), cv::Point(131,242), cv::Point(137,233), cv::Point(139,226), cv::Point(135,219),
 *	//								   cv::Point(125,217)};
 *
 *	std::vector<cv::Point> polygon = { cv::Point(372,243), cv::Point(390,236), cv::Point(404,232), cv::Point(412,232), cv::Point(427,235),
 *									   cv::Point(438,245), cv::Point(445,263), cv::Point(444,287), cv::Point(444,306), cv::Point(442,325),
 *									   cv::Point(436,332), cv::Point(426,342), cv::Point(416,347), cv::Point(403,343), cv::Point(393,335),
 *									   cv::Point(393,327), cv::Point(390,321), cv::Point(389,309), cv::Point(384,298), cv::Point(383,285),
 *									   cv::Point(383,277), cv::Point(380,259) };
 *
 *
 *	//RealTimeOpticalFlow optFlow; //GPU Version
 *	RealTimeOpticalFlow optFlow(false); //CPU Version
 *
 *	if (optFlow.init(frame, polygon)) {
 *
 *		bool running = true;
 *		int64 t, t0 = 0, t1 = 1, tc0, tc1;
 *		cv::Mat image;
 *
 *		std::cout << "Use 'm' for CPU/GPU toggling\n";
 *
 *		while (running)
 *		{
 *			t = cv::getTickCount();
 *			tc0 = cv::getTickCount();
 *			cap >> frame;
 *			if (frame.empty())
 *				break;
 *			std::vector<cv::Point2f> predictionVector;
 *
 *			frame.copyTo(image);
 *
 *			if (optFlow.process(frame, predictionVector)) {
 *				tc1 = cv::getTickCount();
 *				size_t i;
 *				for (i = 0; i < predictionVector.size(); i++)
 *				{
 *					circle(image, predictionVector[i], 3, cv::Scalar(0, 0, 255), -1, 8);
 *				}
 *			}
 *			else {
 *				break;
 *			}
 *
 *
 *			std::stringstream s;
 *			s << "mode: " << ("CPU");
 *			cv::putText(image, s.str(), cv::Point(5, 25), cv::FONT_HERSHEY_SIMPLEX, 1., cv::Scalar(255, 0, 255), 2);
 *
 *			s.str("");
 *			s << "opt. flow FPS: " << cvRound((cv::getTickFrequency() / (tc1 - tc0)));
 *			cv::putText(image, s.str(), cv::Point(5, 65), cv::FONT_HERSHEY_SIMPLEX, 1., cv::Scalar(255, 0, 255), 2);
 *
 *			s.str("");
 *			s << "total FPS: " << cvRound((cv::getTickFrequency() / (t1 - t0)));
 *			putText(image, s.str(), cv::Point(5, 105), cv::FONT_HERSHEY_SIMPLEX, 1., cv::Scalar(255, 0, 255), 2);
 *
 *			imshow("flow", image);
 *
 *			char ch = (char)cv::waitKey(3);
 *			if (ch == 27)
 *				running = false;
 *
 *			t0 = t;
 *			t1 = cv::getTickCount();
 *		}
 *	}
**/



class RealTimeOpticalFlow
{
public:
	RealTimeOpticalFlow();
	RealTimeOpticalFlow(bool gpuMode);
	RealTimeOpticalFlow(bool gpuMode, int iterNum, double prec, int pyrLvl, double eigThr);
	~RealTimeOpticalFlow();

	bool init(const cv::Mat& frm, const std::vector<cv::Point>& pts);
	bool init(const cv::Mat& frm, const std::vector<cv::Point2f>& pts);
	bool process(const cv::Mat& frame, std::vector<cv::Point2f>& predictedPoints);	

private:

	bool m_gpuMode;
	cv::Mat m_initFrame;
	std::vector<cv::Point2f> m_initPts;
	cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> m_optFlowGPU;
	cv::Ptr<cv::cuda::CornersDetector> m_goodFtrsGPU;
	int m_numIters;
	double m_prec;
	int m_maxLevel;
	double m_eigThr;
};

#endif