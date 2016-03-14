#pragma once

#include "../../include/cv/cv.h"
#include "../../include/cv/highgui.h"
#include "../../common.h"
#include <list>
#include <blasxx.h>

#include "TactionCommon.h"

//////////////////////////////////////////////////////////////////////////

#define GRADIENT_MAGNITUDE_MIN	30
#define GRADIENT_MAGNITUDE_MAX	255
#define OPTFLOW_MAGNITUDE_MIN	0.5

#define SPM_HISTOGRAM_TYPE	2 // 0, 1, 2

//////////////////////////////////////////////////////////////////////////

class CAlgTemporalAction
{
public:
	CAlgTemporalAction(void);
	~CAlgTemporalAction(void);

public:
	inline double sqr(double x) {return x*x;}

	bool NormalizeFeatureInMatrix(FEATURE_MATRIX& feature_matrix);

	bool ColorImageGradient(const BYTE* color_img, int img_width, int img_height, vector<IplImage*>& gradient_img);
	bool ColorImageOpticalFlow(IplImage* cal_img, IplImage* ref_img, vector<IplImage*>& optflow_img);

	float VectorAngle(float vx, float vy);
	float VectorMagnitude(float vx, float vy);
	float FlowResponse(float magnitude, float angle, float magnitude_prefer, float angle_prefer, int tuning_width);
	float GradientResponse(float magnitude, float angle, float magnitude_prefer, float angle_prefer, float magnitude_sigma, float angle_sigma);

	bool TemporalCodebookFeature_ImageGradient(list< vector<IplImage*> >& gradient_buffer, const RECT sample, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix);
	bool TemporalCodebookFeature_OpticalFlow(list< vector<IplImage*> >& optflow_buffer, const RECT sample, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix);

	bool TemporalCodebookFeature_MultiScale_ImageGradient(list< vector<IplImage*> >& gradient_buffer, const RECT sample, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, FEATURE_MATRIX& feature_matrix);
	bool ResponseBlockFeature_MultiScale_ImageGradient(list< vector<IplImage*> >& gradient_buffer, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, const RECT sample, FEATURE_MATRIX& feature_matrix);
	bool TemporalCodebookFeature_MultiScale_OpticalFlow(list< vector<IplImage*> >& optflow_buffer, const RECT sample, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, FEATURE_MATRIX& feature_matrix);
	bool ResponseBlockFeature_MultiScale_OpticalFlow(list< vector<IplImage*> >& optflow_buffer, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, const RECT sample, FEATURE_MATRIX& feature_matrix);
	
	bool ImageGradientResponseBlockFeature_MaxMin(IplImage* ig_response, double& max_response, double& min_response);
	bool CalcImageGradientBlockStaticFeature(vector<float>& block_ig_feature, FEATURE_VECTOR& feature);
	bool CalcImageGradientBlockStaticFeature_Mean(vector<float>& block_ig_feature, double& mean_value);
	bool CalcImageGradientBlockStaticFeature_AbsDiff(vector<float>& block_ig_feature, double& diff_value);
	bool CalcImageGradientBlockStaticFeature_DiffPow2(vector<float>& block_ig_feature, double& diff_pow2_value);

	bool ImageGradientResponse(vector<IplImage*>& gradient_buffer, const RECT sample, float magnitude_prefer, float angle_prefer, IplImage* &gradient_response);
	bool OpticalFlowResponse(vector<IplImage*>& optflow_buffer, const RECT sample, float magnitude_prefer, float angle_prefer, IplImage* &optflow_response);

	bool ResponseBlockFeature_ImageGradient(list< vector<IplImage*> >& gradient_buffer, LOCAL_PARTITION& block_index, const RECT sample, FEATURE_MATRIX& feature_matrix);
	bool CalcImageGradientBlockResponseFeature(vector<IplImage*>& response_buffer, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix);
	bool CalcImageGradientBlockResponseFeature_MultiScale(vector<IplImage*>& response_buffer, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix);
	bool CalcCubeImageGradientResponse(list< vector<IplImage*> >& gradient_buffer, const RECT sample, float magnitude_prefer, float angle_prefer, int sample_rate, vector<IplImage*>& response_buffer);

	bool ResponseBlockFeature_OpticalFlow(list< vector<IplImage*> >& optflow_buffer, LOCAL_PARTITION& block_index, const RECT sample, FEATURE_MATRIX& feature_matrix);
	bool CalcOpticalFlowBlockResponseFeature(vector<IplImage*>& response_buffer, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix);
	bool CalcOpticalFlowBlockResponseFeature_MultiScale(vector<IplImage*>& response_buffer, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix);
	bool CalcCubeOpticalFlowResponse(list< vector<IplImage*> >& optflow_buffer, const RECT sample, float magnitude_prefer, float angle_prefer, int sample_rate, vector<IplImage*>& response_buffer);

	bool CalcBoFHistogramIndex(BOF_CODEBOOK& codebook, FEATURE_MATRIX& block_feature_matrix, vector<int>& index);
	bool CalcBoFHistogram(BOF_CODEBOOK& codebook, FEATURE_MATRIX& block_feature_matrix, FEATURE_VECTOR& feature);
	bool CalcBoFHistogram_SPM(BOF_CODEBOOK& codebook, vector<POINT_2D>& image_pos, int image_w, int image_h, FEATURE_MATRIX& block_feature_matrix, FEATURE_VECTOR& feature);
};
