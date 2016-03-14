#pragma once

#include "../../common.h"

struct ID_MAP
{
	int old_id;
	int new_id;
};

struct TACTION_PARAM
{
	string train_type;
	string codebook_fn;
	string celltoear_classifier_fn;
	string objectput_classifier_fn;
	string pointing_classifier_fn;
};

struct LOCAL_PARTITION
{
	vector<POINT_2D> local_index;
	int patch_size;
	int step_size;
	double overlap_percent;

	LOCAL_PARTITION()
	{
		local_index.clear();
		patch_size = 0;
		step_size  = 0;
		overlap_percent = 0.0;
	}
};

typedef vector<LOCAL_PARTITION> LOCAL_PARTITION_MULTISCALE;

struct BOF_CODEBOOK
{
	int codebook_size;		// number of codebook word
	int codebook_dim;		// number of dimension of each word
	float* codebook_data;	// codebook data
	float* codebook_length;	// length of each codebook word

	BOF_CODEBOOK()
	{
		codebook_size = 0;		codebook_dim = 0;
		codebook_data = NULL;	codebook_length = NULL;
	}
};

typedef vector<float> FEATURE_VECTOR;
typedef vector<FEATURE_VECTOR> FEATURE_MATRIX;

struct TACTION_CLASSIFIER
{
	int model_type;
	int w_dim;
	float* classifier_w;
	float  classifier_b;
	float  sigmoid_a;
	float  sigmoid_c;

	TACTION_CLASSIFIER()
	{
		classifier_w = NULL;
	}
};

struct CLASSIFY_ATTRIB
{
	int event_label;
	float classify_dist;
	float classify_conf;
};
typedef vector<CLASSIFY_ATTRIB> LIST_CLASSIFY_ATTRIB;

//////////////////////////////////////////////////////////////////////////

#define taction_map_size 1 + 3
extern ID_MAP taction_label_map[taction_map_size];

//////////////////////////////////////////////////////////////////////////
// fast algorithm for exp() & cos() functions

#define EXPN_SZ  2560 
#define EXPN_MAX 256
#define COSN_SZ	 3600
#define COSN_MAX 360

void fast_expn_init();
double fast_expn(double x);
void fast_cosn_init();
double fast_cosn(double x);
int floor_d(double x);

//////////////////////////////////////////////////////////////////////////

bool LoadTactionParam(const char* param_fn, TACTION_PARAM& param);
bool LoadTactionBoFCodebook(const char* codebook_fn, BOF_CODEBOOK& codebook);
bool LoadTactionClassifier(const char* classifier_fn, TACTION_CLASSIFIER &classifier);
bool GetBlockIndex(const RECT sample, LOCAL_PARTITION& block_index);
bool GetBlockIndex_MultiScale(const RECT sample, LOCAL_PARTITION_MULTISCALE& block_index);
bool NormalizeFeaVector(vector<float>& fea_vec);
bool NormalizeFeaVector_SPM(vector<float>& fea_vec, int step);
double MaxElementInVector(vector<double>& data);
double MinElementInVector(vector<double>& data);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


// 	vector< vector<POINT_2D> > local_index;
// 	vector<int> patch_size;
// 	vector<double> overlap_percent;
// 
// 	LOCAL_PARTITION_MULTISCALE()
// 	{
// 		local_index.clear();
// 		patch_size.clear();
// 		overlap_percent.clear();
// 	}

//struct SAMPLE
//{
//	int		event_id;
//	int		inst_id;
//	int		obj_id;
//	int		cat_label;
//	int		cat_num_id;
//	int		get_id;	// 0: label, 1: detect, 2: track
//	RECT	region;
//};
//
//struct ID_MAP
//{
//	int old_id;
//	int new_id;
//};
//
//#define map_size 1 + 3
//extern ID_MAP label_map[map_size];
//
//struct SIFT_XY
//{
//	int x;
//	int y;
//};
//extern SIFT_XY sift_descr_xy[89][71];

//struct CLASSIFIER
//{
//	int model_type;
//	int w_dim;
//	float* classifier_w;
//	float classifier_b;
//	float sigmoid_a;
//	float sigmoid_c;
//};

/*
bool CAlgTemporalAction::CalcCubeImageGradientResponse(list< vector<IplImage*> >& gradient_buffer, float magnitude_prefer, float angle_prefer, int sample_rate, vector<IplImage*>& response_buffer)
{
response_buffer.clear();

list< vector<IplImage*> >::iterator i_gradient;		int im_index = 0;
for (i_gradient = gradient_buffer.begin(); i_gradient != gradient_buffer.end(); ++i_gradient, im_index++)
{
if (0 == im_index % sample_rate) // sample image gradient for response calculation
{
vector<IplImage*>& cur_gradient = *i_gradient;

IplImage* response = NULL;
ImageGradientResponse(cur_gradient, magnitude_prefer, angle_prefer, response);
response_buffer.push_back(response);
}
}

return true;
}

bool CAlgTemporalAction::ImageGradientResponse(vector<IplImage*>& gradient_buffer, float magnitude_prefer, float angle_prefer, IplImage* gradient_response)
{
if (gradient_buffer.size() < 2 || NULL != gradient_response)
{
AfxMessageBox("Parameter error in \"ImageGradientResponse\"");
return false;
}

int img_wid = cvGetSize(gradient_buffer[0]).width;
int img_hei = cvGetSize(gradient_buffer[0]).height;
gradient_response = cvCreateImage(cvSize(img_wid, img_hei), IPL_DEPTH_32F, 1);
cvSetZero(gradient_response);

for (int i_y = 0; i_y < img_hei; i_y++)
{
for (int i_x = 0; i_x < img_wid; i_x++)
{
float vx = ((float*)(gradient_buffer[0]->imageData + i_y * gradient_buffer[0]->widthStep))[i_x];
float vy = ((float*)(gradient_buffer[1]->imageData + i_y * gradient_buffer[1]->widthStep))[i_x];

float mg = VectorMagnitude(vx, vy);
float ag = VectorAngle(vx, vy);
float rp = FlowResponse(mg, ag, magnitude_prefer, angle_prefer);

((float*)(gradient_response->imageData + (i_y) * gradient_response->widthStep))[i_x] = rp;
}
}

return true;
}

bool CAlgTemporalAction::SampleImageGradientTemporalResponse(list< vector<IplImage*> >& gradient_buffer, const RECT sample, list<IplImage*>& response_buffer)
{
for (int i_angle = 0; i_angle < GRADIENT_ANGLE_RESPONSE_PREFER; i_angle++)
{
int angle_prefer = m_gradient_angle_response_prefer[i_angle];

for (int i_magnitude = 0; i_magnitude < GRADIENT_MAGNITUDE_RESPONSE_PREFER; i_magnitude++)
{
int magnitude_prefer = m_gradient_magnitude_response_prefer[i_magnitude];

list< vector<IplImage*> >::iterator i_gradient;
for (i_gradient = gradient_buffer.begin(); i_gradient != gradient_buffer.end(); ++i_gradient)
{
vector<IplImage*>& cur_gradient = *i_gradient;

IplImage* response = cvCreateImage(cvGetSize(cur_gradient[0]), IPL_DEPTH_32F, 1);
ImageGradientResponse(cur_gradient, sample, magnitude_prefer, angle_prefer, response);
response_buffer.push_back(response);
}
}
}

return true;
}

bool CModuleTemporalAction::CodebookFeature_ImageGradient(LIST_RECT& list_sample_region, LIST_INT&  list_sample_id, FEATURE_MATRIX& fea_matrix)
{
fea_matrix.clear();

int cnt_generate_sample = list_sample_region.size();
for (int i_sample = 0; i_sample < cnt_generate_sample; i_sample++)
{
RECT cur_roi, roi_exp;
cur_roi = list_sample_region[i_sample];
if ( !ROIExpand(cur_roi, roi_exp) )		// region expansion
continue;

FEATURE_VECTOR roi_feature;
if (!m_taction.TemporalCodebookFeature_ImageGradient(m_gradient_buffer, roi_exp, roi_feature)) // image gradient feature
continue;

fea_matrix.push_back(roi_feature);
}

return true;
}
*/

//////////////////////////////////////////////////////////////////////////
// for debug
// 	FILE *fp_test;
// 	fopen_s(&fp_test, "c:\\test.txt", "ab");
// 	for (int i = 0; i < cnt_fea_cop; i++)(cnt_fea_num-1)*cnt_fea_cop(cnt_fea_num)*cnt_fea_cop
// 	{
// 		fprintf(fp_test, "%f\t", feature_memory[i]);
// 	}
// 	fclose(fp_test);
// for debug
//////////////////////////////////////////////////////////////////////////
