#include "RealTimeOpticalFlow.h"


RealTimeOpticalFlow::RealTimeOpticalFlow()
	: m_gpuMode(true)
{
	m_optFlowGPU = cv::cuda::SparsePyrLKOpticalFlow::create();
	m_goodFtrsGPU = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, 21);
}

RealTimeOpticalFlow::RealTimeOpticalFlow(bool gpuMode)
	: m_gpuMode(gpuMode)
{
	if (m_gpuMode) {
		m_optFlowGPU = cv::cuda::SparsePyrLKOpticalFlow::create();
		m_goodFtrsGPU = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, 21);
	}
	else {
		m_numIters = 20;
		m_prec = 0.03;
		m_maxLevel = 3;
		m_eigThr = 0.001;
	}
}

RealTimeOpticalFlow::RealTimeOpticalFlow(bool gpuMode, int iterNum, double prec, int pyrLvl, double eigThr)
	: m_gpuMode(gpuMode)
	, m_numIters(iterNum)
	, m_prec(prec)
	, m_maxLevel(pyrLvl)
	, m_eigThr(eigThr)
{
	if (m_gpuMode) {
		m_optFlowGPU = cv::cuda::SparsePyrLKOpticalFlow::create();
		m_goodFtrsGPU = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, 21);
	}
}


RealTimeOpticalFlow::~RealTimeOpticalFlow()
{
}

bool RealTimeOpticalFlow::init(const cv::Mat& frm, const std::vector<cv::Point>& pts) {
	if (frm.empty() || pts.empty()) {
		return false;
	}

	std::vector<cv::Point2f> ptsF(pts.size());

	const cv::Point* pts_ = &pts[0];
	cv::Point2f* ptsF_ = &ptsF[0];
	size_t i;
	for (i = 0; i < pts.size(); ++i) {
		ptsF_[i].x = (float)pts_[i].x;
		ptsF_[i].y = (float)pts_[i].y;
	}

	return(init(frm, ptsF));
}

bool RealTimeOpticalFlow::init(const cv::Mat& frm, const std::vector<cv::Point2f>& pts)
{
	if (frm.empty() || pts.empty()) {
		return false;
	}
	
	m_initPts = pts;

	int t = frm.type();

	uchar depth = t & CV_MAT_DEPTH_MASK;

	if (depth == CV_8U) {
		if (frm.channels() > 1) {
			cv::cvtColor(frm, m_initFrame, cv::COLOR_BGR2GRAY);
		}
		else {
			m_initFrame = frm;
		}
	}
	else {
		cv::Mat tmp;
		frm.convertTo(tmp, CV_8U);
		if (tmp.channels() > 1) {
			cv::cvtColor(tmp, m_initFrame, cv::COLOR_BGR2GRAY);
		}
		else {
			m_initFrame = tmp;
		}
	}	

	return true;
	
}

bool RealTimeOpticalFlow::process(const cv::Mat & frame, std::vector<cv::Point2f>& predictedPoints)
{
	if (frame.empty()) {
		return false;
	}
	cv::Mat frameNext;
	int t = frame.type();

	uchar depth = t & CV_MAT_DEPTH_MASK;

	if (depth == CV_8U) {
		if (frame.channels() > 1) {
			cv::cvtColor(frame, frameNext, cv::COLOR_BGR2GRAY);
		}
		else {
			frameNext = frame;
		}
	}
	else {
		cv::Mat tmp;
		frame.convertTo(tmp, CV_8U);
		if (tmp.channels() > 1) {
			cv::cvtColor(tmp, frameNext, cv::COLOR_BGR2GRAY);
		}
		else {
			frameNext = tmp;
		}
	}

	if (m_gpuMode) {
		cv::cuda::GpuMat status;
		cv::cuda::GpuMat prevPts(m_initPts);
		cv::cuda::GpuMat nextPts;
		cv::cuda::GpuMat prevFrame(m_initFrame);
		cv::cuda::GpuMat nextFrame(frameNext);
		m_optFlowGPU->calc(prevFrame, nextFrame, prevPts, nextPts, status);

		if (!nextPts.empty()) {
			nextPts.download(predictedPoints);
			m_initFrame = frameNext;
			m_initPts = predictedPoints;
		}
		else {
			cv::Mat mask = cv::Mat::zeros(m_initFrame.rows, m_initFrame.cols, CV_8UC1);
			std::vector<cv::Point> tmpPts(m_initPts.size());
			cv::Point* tmpPts_ = &tmpPts[0];
			cv::Point2f* initPts_ = &m_initPts[0];
			for (size_t i = 0; i < tmpPts.size(); ++i) {
				tmpPts_[i].x = int(initPts_[i].x);
				tmpPts_[i].y = int(initPts_[i].y);
			}
			const cv::Point *ptsp = (const cv::Point*) cv::Mat(tmpPts).data;
			int nptsp = cv::Mat(tmpPts).rows;
			cv::fillConvexPoly(mask, ptsp, nptsp, cv::Scalar(255));
			cv::cuda::GpuMat maskGpu(mask);
			m_goodFtrsGPU->detect(prevFrame, prevPts, maskGpu);
			m_optFlowGPU->calc(prevFrame, nextFrame, prevPts, nextPts, status);
			if (!nextPts.empty()) {
				nextPts.download(predictedPoints);
				m_initFrame = frameNext;
				m_initPts = predictedPoints;
			}
			else {
				return false;
			}
		}		
	}
	else {

		std::vector<uchar> status;
		std::vector<float> err;

		std::vector<cv::Point> tmpPts(m_initPts.size());
		cv::Point* tmpPts_ = &tmpPts[0];
		cv::Point2f* initPts_ = &m_initPts[0];
		for (size_t i = 0; i < tmpPts.size(); ++i) {
			tmpPts_[i].x = int(initPts_[i].x);
			tmpPts_[i].y = int(initPts_[i].y);
		}

		cv::Rect r = cv::boundingRect(tmpPts);

		int sz = (r.height > r.width ? r.height : r.width);

		cv::Size subPixWinSize(sz/2, sz/2), winSize(sz, sz);

		cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, m_numIters, m_prec);

		cv::calcOpticalFlowPyrLK(m_initFrame, frameNext, m_initPts, predictedPoints, status, err, winSize,
			m_maxLevel, termcrit, 0, m_eigThr);
		
		if (predictedPoints.size() > 0) {
			m_initFrame = frameNext;
			m_initPts = predictedPoints;
		}
		else {
			cv::Mat mask = cv::Mat::zeros(m_initFrame.rows, m_initFrame.cols, CV_8UC1);
			const cv::Point *ptsp = (const cv::Point*) cv::Mat(tmpPts).data;
			int nptsp = cv::Mat(tmpPts).rows;
			cv::fillConvexPoly(mask, ptsp, nptsp, cv::Scalar(255));

			cv::goodFeaturesToTrack(m_initFrame, m_initPts, 21, 0.01, 10, mask);
			std::vector<cv::Point2f> prvPts = m_initPts;
			if (prvPts.size() > 0) {
				cv::cornerSubPix(frameNext, prvPts, subPixWinSize, cv::Size(-1, -1), termcrit);
				if (prvPts.size() > 0) {
					cv::calcOpticalFlowPyrLK(m_initFrame, frameNext, prvPts, predictedPoints, status, err, winSize,
						m_maxLevel, termcrit, 0, m_eigThr);
					if (predictedPoints.size() > 0) {
						m_initFrame = frameNext;
						m_initPts = predictedPoints;
					}
					else {
						return false;
					}
				}
				else {
					cv::calcOpticalFlowPyrLK(m_initFrame, frameNext, m_initPts, predictedPoints, status, err, winSize,
						m_maxLevel, termcrit, 0, m_eigThr);
					if (predictedPoints.size() > 0) {
						m_initFrame = frameNext;
						m_initPts = predictedPoints;
					}
					else {
						return false;
					}
				}
			}
			else {
				return false;
			}

		}
	}
	return true;
}


