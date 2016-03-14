#include "stdafx.h"
#include ".\algtemporalaction.h"
#include "fastflow.h"
#include "simplegradient.h"

#define ENABLE_TIMING	1
#include "timing.h"

//////////////////////////////////////////////////////////////////////////
#define GRADIENT_ANGLE_RESPONSE_PREFER		8
float m_gradient_angle_response_prefer[GRADIENT_ANGLE_RESPONSE_PREFER] = {0, 45, 90, 135, 180, 225, 270, 315};
#define GRADIENT_ANGLE_RESPONSE_SIGMA		45
#define GRADIENT_MAGNITUDE_RESPONSE_PREFER	2
float m_gradient_magnitude_response_prefer[GRADIENT_MAGNITUDE_RESPONSE_PREFER] = {75, 150};//{100, 200};
#define	GRADIENT_MAGNITUDE_RESPONSE_SIGMA	75//100

#define OPTFLOW_ANGLE_RESPONSE_PREFER		8
float m_optflow_angle_response_prefer[OPTFLOW_ANGLE_RESPONSE_PREFER] = {0, 45, 90, 135, 180, 225, 270, 315};
#define OPTFLOW_ANGLE_RESPONSE_SIGMA		45
#define OPTFLOW_MAGNITUDE_RESPONSE_PREFER	2
float m_optflow_magnitude_response_prefer[OPTFLOW_MAGNITUDE_RESPONSE_PREFER] = {2.5, 5.0};//{3, 6};
#define	OPTFLOW_MAGNITUDE_RESPONSE_SIGMA	2.5
//////////////////////////////////////////////////////////////////////////

CAlgTemporalAction::CAlgTemporalAction(void)
{
	fast_expn_init();
	fast_cosn_init();
}

CAlgTemporalAction::~CAlgTemporalAction(void)
{
}

// calculate gradient for color image, assume color image is RBG bit
bool CAlgTemporalAction::ColorImageGradient(const BYTE* color_img, int img_width, int img_height, vector<IplImage*>& gradient_img)
{
	int aperture_size = 3;

	if ( NULL == color_img || gradient_img.size() < 2 || img_width <= 0 || img_height <= 0 ||
         cvGetSize(gradient_img[0]).width  != img_width  || cvGetSize(gradient_img[1]).width  != img_width || 
		 cvGetSize(gradient_img[0]).height != img_height || cvGetSize(gradient_img[1]).height != img_height )
	{
		AfxMessageBox("Parameter error in \"ColorImageGradient\"");
		return false;
	}

	// convert to IplImage format
	IplImage* src_img = cvCreateImageHeader(cvSize(img_width, img_height), IPL_DEPTH_8U, 3);
	src_img->imageData = (char*)color_img;

	// convert to grayscale image
	IplImage* gray_img = cvCreateImage(cvSize(img_width, img_height), IPL_DEPTH_8U, 1);
	cvCvtColor(src_img, gray_img, CV_BGR2GRAY); // CV_RGB2GRAY); // assume color image is BGR bit
	cvFlip(gray_img, NULL, 0);

	// calculate gradient for x and y
	cvSobel(gray_img, gradient_img[0], 1, 0, aperture_size); // x-gradient
	cvSobel(gray_img, gradient_img[1], 0, 1, aperture_size); // y-gradient

	// release
	cvReleaseImageHeader(&src_img);
	cvReleaseImage(&gray_img);

	return true;
}

bool CAlgTemporalAction::ColorImageOpticalFlow(IplImage* cal_img, IplImage* ref_img, vector<IplImage*>& optflow_img)
{
	if (NULL == cal_img || NULL == ref_img || optflow_img.size() < 2)
	{
		AfxMessageBox("Parameter error in \"ColorImageOpticalFlow\"");
		return false;
	}

#define FAST_FLOW
//#define LUCAS_KANADE
//#define HORN_SCHUNCK

#ifdef FAST_FLOW
	fastflow(cal_img, ref_img, optflow_img[0], optflow_img[1]);
#endif

#ifdef LUCAS_KANADE
	// convert to grayscale image
	IplImage* cal_gray_img = cvCreateImage(cvGetSize(cal_img), IPL_DEPTH_8U, 1);
	cvCvtColor(cal_img, cal_gray_img, CV_RGB2GRAY);
	IplImage* ref_gray_img = cvCreateImage(cvGetSize(ref_img), IPL_DEPTH_8U, 1);
	cvCvtColor(ref_img, ref_gray_img, CV_RGB2GRAY);

	cvCalcOpticalFlowLK(cal_gray_img, ref_gray_img, cvSize(3,3), optflow_img[0], optflow_img[1]);

	// release
	cvReleaseImage(&cal_gray_img);
	cvReleaseImage(&ref_gray_img);
#endif

#ifdef HORN_SCHUNCK
	double			lambda		= 1.0;
	int				usePrevious	= 0;
	CvTermCriteria	criteria	= cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
														500,			 0.001);
	// convert to grayscale image
	IplImage* cal_gray_img = cvCreateImage(cvGetSize(cal_img), IPL_DEPTH_8U, 1);
	cvCvtColor(cal_img, cal_gray_img, CV_RGB2GRAY);
	IplImage* ref_gray_img = cvCreateImage(cvGetSize(ref_img), IPL_DEPTH_8U, 1);
	cvCvtColor(ref_img, ref_gray_img, CV_RGB2GRAY);

	cvCalcOpticalFlowHS(cal_gray_img, ref_gray_img, optflow_img[0], optflow_img[1], lambda, criteria);

	// release
	cvReleaseImage(&cal_gray_img);
	cvReleaseImage(&ref_gray_img);
#endif

	return true;
}

// calculate magnitude of a vector(vx,vy)
float CAlgTemporalAction::VectorMagnitude(float vx, float vy)
{
	return (float)(sqrt(sqr(vx)+sqr(vy)));
}

// calculate angle of a vector(vx,vy), return value range [0,360]
float CAlgTemporalAction::VectorAngle(float vx, float vy)
{
	double angle = atan2(vy, vx) * 180 / PI; // return [-180,+180]
	if ( angle >= 0 )
		return (float)angle;
	else
		return (float)(angle + 360);
}

// assume the input angle is in the range of [0,360]
float CAlgTemporalAction::FlowResponse(float magnitude, float angle, float magnitude_prefer, float angle_prefer, int tuning_width=2)
{
	double mag_response, ang_response;

	if ( tuning_width <= 2 )
	{
//		ang_response = sqr( 0.5 * ( 1 + cos(angle - angle_prefer) ) );
		ang_response = sqr( 0.5 * ( 1 + fast_cosn(angle - angle_prefer) ) );
	}
	else
	{
//		ang_response = pow( 0.5 * ( 1 + cos(angle - angle_prefer)), tuning_width);
		ang_response = pow( 0.5 * ( 1 + fast_cosn(angle - angle_prefer)), tuning_width);
	}
//	mag_response = exp(-abs(magnitude - magnitude_prefer));
	mag_response = fast_expn(abs(magnitude - magnitude_prefer));

	return (float)(mag_response * ang_response);
}

// assume the input angle is in the range of [0,360]
float CAlgTemporalAction::GradientResponse(float magnitude, float angle, float magnitude_prefer, float angle_prefer, float magnitude_sigma, float angle_sigma)
{
	if (magnitude < 0 || angle < 0 || angle > 360)
	{
		AfxMessageBox("Parameter error in \"GradientResponse\"");
		return -1;
	}

	// round situation handle
	if ( (angle_prefer + angle_sigma > 360) && (angle >= 0) && (angle <= angle_prefer + angle_sigma - 360) )
	{
		angle = angle + 360;
	}
	else if ( (angle_prefer - angle_sigma < 0) && (angle >= angle_prefer - angle_sigma + 360) && (angle <= 360) )
	{
		angle = angle - 360;
	}
	else
	{
		angle = angle;
	}

	// calculate
	double magnitude_base = abs( (magnitude - magnitude_prefer) / magnitude_sigma );
	double angle_base = abs( (angle - angle_prefer) / angle_sigma );

	double mag_response = magnitude_base < 1 ? 1 - magnitude_base : 0;
	double ang_response = angle_base < 1 ? 1 - angle_base : 0;

	return (float)(mag_response * ang_response);
}

//////////////////////////////////////////////////////////////////////////
// Image Gradient Functions
//////////////////////////////////////////////////////////////////////////

// main function for spm temporal codebook feature extraction of image gradient
bool CAlgTemporalAction::TemporalCodebookFeature_MultiScale_ImageGradient(list< vector<IplImage*> >& gradient_buffer, const RECT sample, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, FEATURE_MATRIX& feature_matrix)
{
	int i_scale;

	// get info. & check parameter
	int cnt_scale_num = multiscale_block_index.size();
	if ( cnt_scale_num <= 0 )
	{
		AfxMessageBox("Parameter error in \"TemporalCodebookFeature_MultiScale_ImageGradient\"");
		return false;
	}

	int cnt_block_num = multiscale_block_index[0].local_index.size();
	for (i_scale = 1; i_scale < cnt_scale_num; i_scale++)
	{
		if ( cnt_block_num != (int)multiscale_block_index[i_scale].local_index.size() )
		{
			AfxMessageBox("Parameter error in \"TemporalCodebookFeature_MultiScale_ImageGradient\"");
			return false;
		}
	}

	// response block feature & normalize
	feature_matrix.clear();
	ResponseBlockFeature_MultiScale_ImageGradient(gradient_buffer, multiscale_block_index, sample, feature_matrix);	// response block feature
	NormalizeFeatureInMatrix(feature_matrix);

	return true;
}

bool CAlgTemporalAction::ResponseBlockFeature_MultiScale_ImageGradient(list< vector<IplImage*> >& gradient_buffer, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, const RECT sample, FEATURE_MATRIX& feature_matrix)
{
	// suppose multiscale_block_index has been set properly
	int cnt_scale_num = multiscale_block_index.size();
	int cnt_block_num = multiscale_block_index[0].local_index.size();

	feature_matrix.resize(cnt_scale_num * cnt_block_num);

	for (int i_magnitude = 0; i_magnitude < GRADIENT_MAGNITUDE_RESPONSE_PREFER; i_magnitude++)
	{
		float magnitude_prefer = m_gradient_magnitude_response_prefer[i_magnitude];

		for (int i_angle = 0; i_angle < GRADIENT_ANGLE_RESPONSE_PREFER; i_angle++)
		{
			float angle_prefer = m_gradient_angle_response_prefer[i_angle];		

			// calculate cube gradient response with sample rate
			int cube_sample_rate = 3;
			vector<IplImage*> response_buffer;	
			CalcCubeImageGradientResponse(gradient_buffer, sample, magnitude_prefer, angle_prefer, cube_sample_rate, response_buffer);				

			// calculate 8x8x8 cubic feature
			for (int i_scale = 0; i_scale < cnt_scale_num; i_scale++)
			{
				FEATURE_MATRIX sub_feature_matrix;		sub_feature_matrix.resize(multiscale_block_index[i_scale].local_index.size());
				CalcImageGradientBlockResponseFeature_MultiScale(response_buffer, multiscale_block_index[i_scale], sub_feature_matrix);

				if (cnt_block_num != (int)sub_feature_matrix.size())
				{
					AfxMessageBox("Parameter error in \"ResponseBlockFeature_MultiScale_ImageGradient\"");
					return false;
				}

				for (int i_fea = 0; i_fea < (int)sub_feature_matrix.size(); i_fea++)
				{
					for (int i_value = 0; i_value < (int)sub_feature_matrix[i_fea].size(); i_value++)
					{
						feature_matrix[i_fea * cnt_scale_num + i_scale].push_back(sub_feature_matrix[i_fea][i_value]);
					}
				}
			}

			// release memory
			if (response_buffer.size() > 0)
			{
				for (int it = 0; it < (int)response_buffer.size(); ++it)
				{
					IplImage* img_buffer = response_buffer[it];
					cvReleaseImage(&img_buffer);
				}
				response_buffer.clear();
			}			
		}
	}

	return true;
}

// main function for spm temporal codebook feature extraction of optical flow
bool CAlgTemporalAction::TemporalCodebookFeature_MultiScale_OpticalFlow(list< vector<IplImage*> >& optflow_buffer, const RECT sample, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, FEATURE_MATRIX& feature_matrix)
{
	int i_scale;

	// get info. & check parameter
	int cnt_scale_num = multiscale_block_index.size();
	if ( cnt_scale_num <= 0 )
	{
		AfxMessageBox("Parameter error in \"TemporalCodebookFeature_MultiScale_OpticalFlow\"");
		return false;
	}

	int cnt_block_num = multiscale_block_index[0].local_index.size();
	for (i_scale = 1; i_scale < cnt_scale_num; i_scale++)
	{
		if ( cnt_block_num != (int)multiscale_block_index[i_scale].local_index.size() )
		{
			AfxMessageBox("Parameter error in \"TemporalCodebookFeature_MultiScale_OpticalFlow\"");
			return false;
		}
	}

	// response block feature & normalize
	feature_matrix.clear();
	ResponseBlockFeature_MultiScale_OpticalFlow(optflow_buffer, multiscale_block_index, sample, feature_matrix);
	NormalizeFeatureInMatrix(feature_matrix);

	return true;
}

bool CAlgTemporalAction::ResponseBlockFeature_MultiScale_OpticalFlow(list< vector<IplImage*> >& optflow_buffer, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, const RECT sample, FEATURE_MATRIX& feature_matrix)
{
	// suppose multiscale_block_index has been set properly
	int cnt_scale_num = multiscale_block_index.size();
	int cnt_block_num = multiscale_block_index[0].local_index.size();

	feature_matrix.resize(cnt_scale_num * cnt_block_num);

	for (int i_magnitude = 0; i_magnitude < OPTFLOW_MAGNITUDE_RESPONSE_PREFER; i_magnitude++)
	{
		float magnitude_prefer = m_optflow_magnitude_response_prefer[i_magnitude];

		for (int i_angle = 0; i_angle < OPTFLOW_ANGLE_RESPONSE_PREFER; i_angle++)
		{
			float angle_prefer = m_optflow_angle_response_prefer[i_angle];		

			// calculate cube optical flow response with sample rate
			int cube_sample_rate = 3;
			vector<IplImage*> response_buffer;	
			CalcCubeOpticalFlowResponse(optflow_buffer, sample, magnitude_prefer, angle_prefer, cube_sample_rate, response_buffer);				

			// calculate 8x8x8 cubic feature
			for (int i_scale = 0; i_scale < cnt_scale_num; i_scale++)
			{
				FEATURE_MATRIX sub_feature_matrix;		sub_feature_matrix.resize(multiscale_block_index[i_scale].local_index.size());
				CalcOpticalFlowBlockResponseFeature_MultiScale(response_buffer, multiscale_block_index[i_scale], sub_feature_matrix);

				if (cnt_block_num != (int)sub_feature_matrix.size())
				{
					AfxMessageBox("Parameter error in \"ResponseBlockFeature_MultiScale_OpticalFlow\"");
					return false;
				}

				for (int i_fea = 0; i_fea < (int)sub_feature_matrix.size(); i_fea++)
				{
					for (int i_value = 0; i_value < (int)sub_feature_matrix[i_fea].size(); i_value++)
					{
						feature_matrix[i_fea * cnt_scale_num + i_scale].push_back(sub_feature_matrix[i_fea][i_value]);
					}
				}
			}

			// release memory
			if (response_buffer.size() > 0)
			{
				for (int it = 0; it < (int)response_buffer.size(); ++it)
				{
					IplImage* img_buffer = response_buffer[it];
					cvReleaseImage(&img_buffer);
				}
				response_buffer.clear();
			}			
		}
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////

// main function for temporal codebook feature extraction of image gradient
bool CAlgTemporalAction::TemporalCodebookFeature_ImageGradient(list< vector<IplImage*> >& gradient_buffer, const RECT sample, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix)
{
	feature_matrix.clear();
	feature_matrix.resize(block_index.local_index.size());

//	BEGIN_TIMING(ResponseBlockFeature_ImageGradient);

	ResponseBlockFeature_ImageGradient(gradient_buffer, block_index, sample, feature_matrix);	// response block feature
	NormalizeFeatureInMatrix(feature_matrix);

//	END_TIMING(ResponseBlockFeature_ImageGradient);
//	TimingShowStatistics();		printf("\n");

	return true;
}

bool CAlgTemporalAction::ResponseBlockFeature_ImageGradient(list< vector<IplImage*> >& gradient_buffer, LOCAL_PARTITION& block_index, const RECT sample, FEATURE_MATRIX& feature_matrix)
{
	for (int i_magnitude = 0; i_magnitude < GRADIENT_MAGNITUDE_RESPONSE_PREFER; i_magnitude++)
	{
		float magnitude_prefer = m_gradient_magnitude_response_prefer[i_magnitude];

		for (int i_angle = 0; i_angle < GRADIENT_ANGLE_RESPONSE_PREFER; i_angle++)
		{
			float angle_prefer = m_gradient_angle_response_prefer[i_angle];		

			// calculate cube gradient response with sample rate
			int cube_sample_rate = 3;
			vector<IplImage*> response_buffer;	
			CalcCubeImageGradientResponse(gradient_buffer, sample, magnitude_prefer, angle_prefer, cube_sample_rate, response_buffer);				

			// calculate 8x8x8 cubic feature
//			CalcImageGradientBlockResponseFeature(response_buffer, block_index, feature_matrix);
			CalcImageGradientBlockResponseFeature_MultiScale(response_buffer, block_index, feature_matrix);

			// release memory
			if (response_buffer.size() > 0)
			{
				for (int it = 0; it < (int)response_buffer.size(); ++it)
				{
					IplImage* img_buffer = response_buffer[it];
					cvReleaseImage(&img_buffer);
				}
				response_buffer.clear();
			}			
		}
	}

	return true;
}

bool CAlgTemporalAction::CalcCubeImageGradientResponse(list< vector<IplImage*> >& gradient_buffer, const RECT sample, float magnitude_prefer, float angle_prefer, int sample_rate, vector<IplImage*>& response_buffer)
{
	if (response_buffer.size() > 0)
	{
		AfxMessageBox("Parameter error in \"CalcCubeImageGradientResponse\"");
		response_buffer.clear();
	}

	list< vector<IplImage*> >::iterator i_gradient;		int im_index = 0;
	for (i_gradient = gradient_buffer.begin(); i_gradient != gradient_buffer.end(); ++i_gradient, im_index++)
	{
		if (0 == im_index % sample_rate) // sample image gradient for response calculation
		{
			vector<IplImage*>& cur_gradient = *i_gradient;

			IplImage* response = NULL;
			ImageGradientResponse(cur_gradient, sample, magnitude_prefer, angle_prefer, response);
			response_buffer.push_back(response);
		}
	}
	
	return true;
}

bool CAlgTemporalAction::ImageGradientResponse(vector<IplImage*>& gradient_buffer, const RECT sample, float magnitude_prefer, float angle_prefer, IplImage* &gradient_response)
{
	if (gradient_buffer.size() < 2 || NULL != gradient_response)
	{
		AfxMessageBox("Parameter error in \"ImageGradientResponse\"");
		return false;
	}

	int img_wid = sample.right - sample.left + 1;
	int img_hei = sample.bottom - sample.top + 1;
	gradient_response = cvCreateImage(cvSize(img_wid, img_hei), IPL_DEPTH_32F, 1);
	cvSetZero(gradient_response);

	for (int i_x = sample.left; i_x <= sample.right; i_x++)
	{
		for (int i_y = sample.top; i_y <= sample.bottom; i_y++)
		{
			float vx = ((float*)(gradient_buffer[0]->imageData + i_y * gradient_buffer[0]->widthStep))[i_x];
			float vy = ((float*)(gradient_buffer[1]->imageData + i_y * gradient_buffer[1]->widthStep))[i_x];

			float rp;
			float mg = VectorMagnitude(vx, vy);
			if ( mg < GRADIENT_MAGNITUDE_MIN )
			{
				rp = 0;
			}
			else if ( mg > GRADIENT_MAGNITUDE_MAX )
			{
				rp = 0;//255;
			}
			else
			{
				float ag = VectorAngle(vx, vy);
				//////////////////////////////////////////////////////////////////////////
				rp = GradientResponse(mg, ag, magnitude_prefer, angle_prefer, (float)GRADIENT_MAGNITUDE_RESPONSE_SIGMA, (float)GRADIENT_ANGLE_RESPONSE_SIGMA);
//				rp = FlowResponse(mg, ag, magnitude_prefer, angle_prefer);
				//////////////////////////////////////////////////////////////////////////
			}
			((float*)(gradient_response->imageData + (i_y - sample.top) * gradient_response->widthStep))[i_x - sample.left] = rp;
		}
	}

	return true;
}

bool CAlgTemporalAction::CalcImageGradientBlockResponseFeature(vector<IplImage*>& response_buffer, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix)
{
	if (feature_matrix.size() != block_index.local_index.size())
	{
		AfxMessageBox("Parameter error in \"CalcImageGradientBlockResponseFeature\"");
		return false;
	}

	int patch_size = block_index.patch_size;
	IplImage* cur_response = cvCreateImage(cvSize(patch_size, patch_size), IPL_DEPTH_32F, 1);

	for (int i_block = 0; i_block < (int)block_index.local_index.size(); i_block++)
	{
		int cur_index_x = (int)block_index.local_index[i_block].x;
		int cur_index_y = (int)block_index.local_index[i_block].y;

		vector<float> select_response;
		for (int i_response_image = 0; i_response_image < (int)response_buffer.size(); i_response_image++)
		{
			// 8x8x8 block feature
			cvSetZero(cur_response);
			for (int i_x = cur_index_x; i_x < cur_index_x + patch_size; i_x++)
			{
				for (int i_y = cur_index_y; i_y < cur_index_y + patch_size; i_y++)
				{
					((float*)(cur_response->imageData + (i_y - cur_index_y) * cur_response->widthStep))[i_x - cur_index_x] = 
						((float*)(response_buffer[i_response_image]->imageData + i_y * response_buffer[i_response_image]->widthStep))[i_x];
				}
			}

			double min_response, max_response;
			ImageGradientResponseBlockFeature_MaxMin(cur_response, max_response, min_response);
			select_response.push_back((float)max_response);
		}

		FEATURE_VECTOR sample_feature;
		CalcImageGradientBlockStaticFeature(select_response, sample_feature);
		for (int i_fea = 0; i_fea < (int)sample_feature.size(); i_fea++)
		{
			feature_matrix[i_block].push_back(sample_feature[i_fea]);
		}
	}

	cvReleaseImage(&cur_response);
	return true;
}

bool CAlgTemporalAction::CalcImageGradientBlockResponseFeature_MultiScale(vector<IplImage*>& response_buffer, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix)
{
	if (feature_matrix.size() != block_index.local_index.size())
	{
		AfxMessageBox("Parameter error in \"CalcImageGradientBlockResponseFeature_MultiScale\"");
		return false;
	}

	for (int i_block = 0; i_block < (int)block_index.local_index.size(); i_block++)
	{
		// get multi-scale block index
		RECT cur_roi;
		cur_roi.left = 0,	cur_roi.right  = block_index.patch_size - 1;
		cur_roi.top  = 0,	cur_roi.bottom = block_index.patch_size - 1;

		LOCAL_PARTITION sub_block_index;
		sub_block_index.patch_size = block_index.patch_size / 2;		
		sub_block_index.overlap_percent = 0;
		if ( !GetBlockIndex(cur_roi, sub_block_index) )
			continue;

		// calculate sub-block feature
		IplImage* cur_response = cvCreateImage(cvSize(sub_block_index.patch_size, sub_block_index.patch_size), IPL_DEPTH_32F, 1);
		for (int i_sub_block = 0; i_sub_block < (int)sub_block_index.local_index.size(); i_sub_block++)
		{
			int cur_index_x = (int)block_index.local_index[i_block].x + (int)sub_block_index.local_index[i_sub_block].x;
			int cur_index_y = (int)block_index.local_index[i_block].y + (int)sub_block_index.local_index[i_sub_block].y;

			vector<float> select_response;		select_response.clear();
			for (int i_response_image = 0; i_response_image < (int)response_buffer.size(); i_response_image++)
			{
				cvSetZero(cur_response);
				for (int i_x = cur_index_x; i_x < cur_index_x + sub_block_index.patch_size; i_x++)
				{
					for (int i_y = cur_index_y; i_y < cur_index_y + sub_block_index.patch_size; i_y++)
					{
						((float*)(cur_response->imageData + (i_y - cur_index_y) * cur_response->widthStep))[i_x - cur_index_x] = 
							((float*)(response_buffer[i_response_image]->imageData + i_y * response_buffer[i_response_image]->widthStep))[i_x];
					}
				}

				double min_response, max_response;
				ImageGradientResponseBlockFeature_MaxMin(cur_response, max_response, min_response);
				select_response.push_back((float)max_response);
			}

			FEATURE_VECTOR sample_feature;
			CalcImageGradientBlockStaticFeature(select_response, sample_feature);
			for (int i_fea = 0; i_fea < (int)sample_feature.size(); i_fea++)
			{
				feature_matrix[i_block].push_back(sample_feature[i_fea]);
			}
		}
		cvReleaseImage(&cur_response);
	}

	return true;
}

bool CAlgTemporalAction::ImageGradientResponseBlockFeature_MaxMin(IplImage* ig_response, double& max_response, double& min_response)
{
	if (NULL == ig_response)
	{
		AfxMessageBox("Parameter error in \"ImageGradientResponseBlockFeature_MaxMin\"");
		return false;
	}

	cvMinMaxLoc(ig_response, &min_response, &max_response);
	return true;
}

bool CAlgTemporalAction::CalcImageGradientBlockStaticFeature(vector<float>& block_ig_feature, FEATURE_VECTOR& feature)
{
	feature.clear();

	double mean_value;
	CalcImageGradientBlockStaticFeature_Mean(block_ig_feature, mean_value);
	feature.push_back((float)mean_value);

	double diff_value;
	CalcImageGradientBlockStaticFeature_AbsDiff(block_ig_feature, diff_value);
	feature.push_back((float)diff_value);

// 	double diff_pow2_value;
// 	CalcImageGradientBlockStaticFeature_DiffPow2(block_ig_feature, diff_pow2_value);
// 	feature.push_back((float)diff_pow2_value);

	return true;
}

bool CAlgTemporalAction::CalcImageGradientBlockStaticFeature_Mean(vector<float>& block_ig_feature, double& mean_value)
{
	mean_value = 0;

	int cnt_select_response = block_ig_feature.size();
	for (int i_rep = 0; i_rep < cnt_select_response; i_rep++)
	{
		mean_value += block_ig_feature[i_rep];
	}
	mean_value /= cnt_select_response;

	return true;
}

bool CAlgTemporalAction::CalcImageGradientBlockStaticFeature_AbsDiff(vector<float>& block_ig_feature, double& diff_value)
{
	diff_value = 0;

	int cnt_select_response = block_ig_feature.size();
	for (int i_rep = 1; i_rep < cnt_select_response; i_rep++)
	{
		diff_value += abs(block_ig_feature[i_rep] - block_ig_feature[i_rep-1]);
	}
	diff_value /= cnt_select_response;

	return true;
}

bool CAlgTemporalAction::CalcImageGradientBlockStaticFeature_DiffPow2(vector<float>& block_ig_feature, double& diff_pow2_value)
{
	diff_pow2_value = 0;

	int cnt_select_response = block_ig_feature.size();
	for (int i_rep = 1; i_rep < cnt_select_response; i_rep++)
	{
		diff_pow2_value += (block_ig_feature[i_rep] - block_ig_feature[i_rep-1]) * (block_ig_feature[i_rep] - block_ig_feature[i_rep-1]);
	}
	diff_pow2_value = sqrt(diff_pow2_value / (cnt_select_response - 1));

	return true;
}

//////////////////////////////////////////////////////////////////////////
// Optical Flow Functions
//////////////////////////////////////////////////////////////////////////

// main function for temporal codebook feature extraction of optical flow
bool CAlgTemporalAction::TemporalCodebookFeature_OpticalFlow(list< vector<IplImage*> >& optflow_buffer, const RECT sample, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix)
{
	feature_matrix.clear();
	feature_matrix.resize(block_index.local_index.size());

//	BEGIN_TIMING(ResponseBlockFeature_OpticalFlow);

	ResponseBlockFeature_OpticalFlow(optflow_buffer, block_index, sample, feature_matrix);
	NormalizeFeatureInMatrix(feature_matrix);

//	END_TIMING(ResponseBlockFeature_OpticalFlow);
//	TimingShowStatistics();		printf("\n");

	return true;
}

bool CAlgTemporalAction::ResponseBlockFeature_OpticalFlow(list< vector<IplImage*> >& optflow_buffer, LOCAL_PARTITION& block_index, const RECT sample, FEATURE_MATRIX& feature_matrix)
{
	for (int i_magnitude = 0; i_magnitude < OPTFLOW_MAGNITUDE_RESPONSE_PREFER; i_magnitude++)
	{
		float magnitude_prefer = m_optflow_magnitude_response_prefer[i_magnitude];

		for (int i_angle = 0; i_angle < OPTFLOW_ANGLE_RESPONSE_PREFER; i_angle++)
		{
			float angle_prefer = m_optflow_angle_response_prefer[i_angle];		

			// calculate cube optical flow response with sample rate
			int cube_sample_rate = 3;
			vector<IplImage*> response_buffer;	
			CalcCubeOpticalFlowResponse(optflow_buffer, sample, magnitude_prefer, angle_prefer, cube_sample_rate, response_buffer);				

			// calculate 8x8x8 cubic feature
//			CalcOpticalFlowBlockResponseFeature(response_buffer, block_index, feature_matrix);
			CalcOpticalFlowBlockResponseFeature_MultiScale(response_buffer, block_index, feature_matrix);

			// release memory
			if (response_buffer.size() > 0)
			{
				for (int it = 0; it < (int)response_buffer.size(); ++it)
				{
					IplImage* img_buffer = response_buffer[it];
					cvReleaseImage(&img_buffer);
				}
				response_buffer.clear();
			}			
		}
	}

	return true;
}

bool CAlgTemporalAction::CalcCubeOpticalFlowResponse(list< vector<IplImage*> >& optflow_buffer, const RECT sample, float magnitude_prefer, float angle_prefer, int sample_rate, vector<IplImage*>& response_buffer)
{
	if (response_buffer.size() > 0)
	{
		AfxMessageBox("Parameter error in \"CalcCubeOpticalFlowResponse\"");
		response_buffer.clear();
	}

	list< vector<IplImage*> >::iterator i_optflow;		int im_index = 0;
	for (i_optflow = optflow_buffer.begin(); i_optflow != optflow_buffer.end(); ++i_optflow, im_index++)
	{
		if (0 == im_index % sample_rate) // sample optical flow for response calculation
		{
			vector<IplImage*>& cur_optflow = *i_optflow;

			IplImage* response = NULL;
			OpticalFlowResponse(cur_optflow, sample, magnitude_prefer, angle_prefer, response);
			response_buffer.push_back(response);
		}
	}

	return true;
}

bool CAlgTemporalAction::OpticalFlowResponse(vector<IplImage*>& optflow_buffer, const RECT sample, float magnitude_prefer, float angle_prefer, IplImage* &optflow_response)
{
	if (optflow_buffer.size() < 2 || NULL != optflow_response)
	{
		AfxMessageBox("Parameter error in \"OpticalFlowResponse\"");
		return false;
	}

	int img_wid = sample.right - sample.left + 1;
	int img_hei = sample.bottom - sample.top + 1;
	optflow_response = cvCreateImage(cvSize(img_wid, img_hei), IPL_DEPTH_32F, 1);
	cvSetZero(optflow_response);

	for (int i_x = sample.left; i_x <= sample.right; i_x++)
	{
		for (int i_y = sample.top; i_y <= sample.bottom; i_y++)
		{
			float vx = ((float*)(optflow_buffer[0]->imageData + i_y * optflow_buffer[0]->widthStep))[i_x];
			float vy = ((float*)(optflow_buffer[1]->imageData + i_y * optflow_buffer[1]->widthStep))[i_x];

			float rp;
			float mg = VectorMagnitude(vx, vy);
			if (mg < OPTFLOW_MAGNITUDE_MIN)
			{
				rp = 0;
			}
			else
			{
				float ag = VectorAngle(vx, vy);
				//////////////////////////////////////////////////////////////////////////
				rp = FlowResponse(mg, ag, magnitude_prefer, angle_prefer);
//				rp = GradientResponse(mg, ag, magnitude_prefer, angle_prefer, (float)OPTFLOW_MAGNITUDE_RESPONSE_SIGMA, (float)OPTFLOW_ANGLE_RESPONSE_SIGMA);
				//////////////////////////////////////////////////////////////////////////
			}
			((float*)(optflow_response->imageData + (i_y - sample.top) * optflow_response->widthStep))[i_x - sample.left] = rp;
		}
	}

	return true;
}

bool CAlgTemporalAction::CalcOpticalFlowBlockResponseFeature(vector<IplImage*>& response_buffer, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix)
{
	if (feature_matrix.size() != block_index.local_index.size())
	{
		AfxMessageBox("Parameter error in \"CalcOpticalFlowBlockResponseFeature\"");
		return false;
	}

	int patch_size = block_index.patch_size;
	IplImage* cur_response = cvCreateImage(cvSize(patch_size, patch_size), IPL_DEPTH_32F, 1);

	for (int i_block = 0; i_block < (int)block_index.local_index.size(); i_block++)
	{
		int cur_index_x = (int)block_index.local_index[i_block].x;
		int cur_index_y = (int)block_index.local_index[i_block].y;

		vector<float> select_response;
		for (int i_response_image = 0; i_response_image < (int)response_buffer.size(); i_response_image++)
		{
			// 8x8x8 block feature
			cvSetZero(cur_response);
			for (int i_x = cur_index_x; i_x < cur_index_x + patch_size; i_x++)
			{
				for (int i_y = cur_index_y; i_y < cur_index_y + patch_size; i_y++)
				{
					((float*)(cur_response->imageData + (i_y - cur_index_y) * cur_response->widthStep))[i_x - cur_index_x] = 
						((float*)(response_buffer[i_response_image]->imageData + i_y * response_buffer[i_response_image]->widthStep))[i_x];
				}
			}

			double min_response, max_response;
			//////////////////////////////////////////////////////////////////////////
			ImageGradientResponseBlockFeature_MaxMin(cur_response, max_response, min_response);	// same with image gradient
			//////////////////////////////////////////////////////////////////////////
			select_response.push_back((float)max_response);
		}

		FEATURE_VECTOR sample_feature;
		//////////////////////////////////////////////////////////////////////////
		CalcImageGradientBlockStaticFeature(select_response, sample_feature);	// same with image gradient
		//////////////////////////////////////////////////////////////////////////
		for (int i_fea = 0; i_fea < (int)sample_feature.size(); i_fea++)
		{
			feature_matrix[i_block].push_back(sample_feature[i_fea]);
		}
	}

	cvReleaseImage(&cur_response);
	return true;
}

bool CAlgTemporalAction::CalcOpticalFlowBlockResponseFeature_MultiScale(vector<IplImage*>& response_buffer, LOCAL_PARTITION& block_index, FEATURE_MATRIX& feature_matrix)
{
	if (feature_matrix.size() != block_index.local_index.size())
	{
		AfxMessageBox("Parameter error in \"CalcOpticalFlowBlockResponseFeature_MultiScale\"");
		return false;
	}

	for (int i_block = 0; i_block < (int)block_index.local_index.size(); i_block++)
	{
		// get multi-scale block index
		RECT cur_roi;
		cur_roi.left = 0,	cur_roi.right  = block_index.patch_size - 1;
		cur_roi.top  = 0,	cur_roi.bottom = block_index.patch_size - 1;

		LOCAL_PARTITION sub_block_index;
		sub_block_index.patch_size = block_index.patch_size / 2;		
		sub_block_index.overlap_percent = 0;
		if ( !GetBlockIndex(cur_roi, sub_block_index) )
			continue;

		// calculate sub-block feature
		IplImage* cur_response = cvCreateImage(cvSize(sub_block_index.patch_size, sub_block_index.patch_size), IPL_DEPTH_32F, 1);
		for (int i_sub_block = 0; i_sub_block < (int)sub_block_index.local_index.size(); i_sub_block++)
		{
			int cur_index_x = (int)block_index.local_index[i_block].x + (int)sub_block_index.local_index[i_sub_block].x;
			int cur_index_y = (int)block_index.local_index[i_block].y + (int)sub_block_index.local_index[i_sub_block].y;

			vector<float> select_response;		select_response.clear();
			for (int i_response_image = 0; i_response_image < (int)response_buffer.size(); i_response_image++)
			{
				cvSetZero(cur_response);
				for (int i_x = cur_index_x; i_x < cur_index_x + sub_block_index.patch_size; i_x++)
				{
					for (int i_y = cur_index_y; i_y < cur_index_y + sub_block_index.patch_size; i_y++)
					{
						((float*)(cur_response->imageData + (i_y - cur_index_y) * cur_response->widthStep))[i_x - cur_index_x] = 
							((float*)(response_buffer[i_response_image]->imageData + i_y * response_buffer[i_response_image]->widthStep))[i_x];
					}
				}

				double min_response, max_response;
				//////////////////////////////////////////////////////////////////////////
				ImageGradientResponseBlockFeature_MaxMin(cur_response, max_response, min_response);	// same with image gradient
				//////////////////////////////////////////////////////////////////////////
				select_response.push_back((float)max_response);
			}

			FEATURE_VECTOR sample_feature;
			//////////////////////////////////////////////////////////////////////////
			CalcImageGradientBlockStaticFeature(select_response, sample_feature);		// same with image gradient
			//////////////////////////////////////////////////////////////////////////
			for (int i_fea = 0; i_fea < (int)sample_feature.size(); i_fea++)
			{
				feature_matrix[i_block].push_back(sample_feature[i_fea]);
			}
		}
		cvReleaseImage(&cur_response);
	}

	return true;
}

bool CAlgTemporalAction::NormalizeFeatureInMatrix(FEATURE_MATRIX& feature_matrix)
{
	if (feature_matrix.size() <= 0)
	{
		AfxMessageBox("Parameter error in \"NormalizeFeatureInMatrix\"");
		return false;
	}

	int i_fea, cnt_fea_num = feature_matrix.size();
	for (i_fea = 0; i_fea < cnt_fea_num; i_fea++)
	{
		NormalizeFeaVector(feature_matrix[i_fea]);
	}

	return true;
}

// multiscale bof spm histogram
bool CAlgTemporalAction::CalcBoFHistogram_SPM(BOF_CODEBOOK& codebook, vector<POINT_2D>& image_pos, int image_w, int image_h, FEATURE_MATRIX& block_feature_matrix, FEATURE_VECTOR& feature)
{
	int i;
	int code_id;
	int region_x, region_y;

	int fea_num = block_feature_matrix.size();
	int fea_dim = block_feature_matrix[0].size();

	if (fea_num <= 0 || fea_dim != codebook.codebook_dim)
	{
		AfxMessageBox("Parameter error in \"CalcBoFHistogram_SPM\"");
		return false;
	}

	// calculate index for feature
	vector<int> index;
	CalcBoFHistogramIndex(codebook, block_feature_matrix, index);

	// calculate histogram feature
	FEATURE_VECTOR feature_0;	feature_0.resize(codebook.codebook_size, 0);
	FEATURE_VECTOR feature_1;	feature_1.resize(codebook.codebook_size*4, 0);
	FEATURE_VECTOR feature_2;	feature_2.resize(codebook.codebook_size*16, 0);

	for (i = 0; i < (int)index.size(); i++)
	{
		code_id = index[i];

		// layout-0 histogram
		feature_0[code_id]++;

		// layout-1 histogram
		region_x = (int)(image_pos[i].x * 2.0 / image_w +1);
		region_x = (region_x > 2) ? 2 : region_x;
		region_y = (int)(image_pos[i].y * 2.0 / image_h + 1);
		region_y = (region_y > 2) ? 2 : region_y;
		feature_1[code_id * 4 + (region_y - 1) * 2 + region_x - 1]++;
		
		// layout-2 histogram
		region_x = (int)(image_pos[i].x * 4.0 / image_w + 1);
		region_x = (region_x > 4) ? 4 : region_x;
		region_y = (int)(image_pos[i].y * 4.0 / image_h + 1);
		region_y = (region_y > 4) ? 4 : region_y;
		feature_2[code_id * 16 + (region_y - 1) * 4 + region_x - 1]++;
	}

	// normalize feature
	bool bnormalize = true;
	if (bnormalize)
	{
		NormalizeFeaVector_SPM(feature_0, 1);
		NormalizeFeaVector_SPM(feature_1, 4);
		NormalizeFeaVector_SPM(feature_2, 16);
	}

	// select histogram for feature extraction				
	int fea_type = SPM_HISTOGRAM_TYPE;
	switch (fea_type)
	{
	case 0:
		for (i = 0; i < (int)feature_0.size(); i++)
			feature.push_back(feature_0[i]);
		break;
	case 1:
		for (i = 0; i < (int)feature_1.size(); i++)
			feature.push_back(feature_1[i]);
		break;
	case 2:
		for (i = 0; i < (int)feature_2.size(); i++)
			feature.push_back(feature_2[i]);
		break;
	}

	return true;
}

// ordinary bof histogram
bool CAlgTemporalAction::CalcBoFHistogram(BOF_CODEBOOK& codebook, FEATURE_MATRIX& block_feature_matrix, FEATURE_VECTOR& feature)
{
	int i_fea, i_dim;

	int fea_num = block_feature_matrix.size();
	int fea_dim = block_feature_matrix[0].size();

	if (fea_num <= 0 || fea_dim != codebook.codebook_dim)
	{
		AfxMessageBox("Parameter error in \"CalcBoFHistogram\"");
		return false;
	}

	// calculate index for feature
	vector<int> index;
	CalcBoFHistogramIndex(codebook, block_feature_matrix, index);

	// calculate histogram feature
	feature.clear();		feature.resize(codebook.codebook_size, 0);
	for (i_fea = 0; i_fea < fea_num; i_fea++)
		feature[index[i_fea]]++;

	// normalize feature
	bool bnormalize = true;
	if (bnormalize)
		NormalizeFeaVector(feature);
	
	return true;
}

bool CAlgTemporalAction::CalcBoFHistogramIndex(BOF_CODEBOOK& codebook, FEATURE_MATRIX& block_feature_matrix, vector<int>& index)
{
	int i_fea, i_dim;

	int fea_num = block_feature_matrix.size();
	int fea_dim = block_feature_matrix[0].size();

	float* f_block_fea = new float[fea_num * fea_dim];	
	int idx = 0;
	for (i_fea = 0; i_fea < fea_num; i_fea++)
	{
		for (i_dim = 0; i_dim < fea_dim; i_dim++)
		{
			f_block_fea[idx] = block_feature_matrix[i_fea][i_dim];
			idx++;
		}
	}

	xblas::SharedValMatrix<float>	mat_feature(f_block_fea, codebook.codebook_dim, fea_num);
	xblas::SharedValMatrix<float>	mat_codebook(codebook.codebook_data, codebook.codebook_dim, codebook.codebook_size);
	xblas::NewValMatrix<float>		mat_product(codebook.codebook_size, fea_num);

	xblas::mmmult(mat_codebook.transpose(), mat_feature, mat_product);
	
	int code_id;		double dist;	
	index.clear(),		index.resize(fea_num);	
	for (i_fea = 0; i_fea < fea_num; i_fea++)
	{
		double min_dist = 10e9;
		float* p = &mat_product(0, i_fea);

		for (int k = 0; k < codebook.codebook_size; k++)
		{
			dist = codebook.codebook_length[k] - 2 * p[k];
			if (dist < min_dist)
			{
				min_dist = dist;
				code_id = k;
			}
		}
		index[i_fea] = code_id;
	}

	delete [] f_block_fea;
	return true;
}