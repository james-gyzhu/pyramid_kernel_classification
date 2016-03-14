#include "stdafx.h"
#include <stdio.h>
#include <cv.h>

#include "TactionCommon.h"

//////////////////////////////////////////////////////////////////////////
ID_MAP taction_label_map[taction_map_size] = { {-1,0}, {6,1}, {8,2}, {18,3} };

//////////////////////////////////////////////////////////////////////////

double expn_tab[EXPN_SZ];
double cosn_tab[COSN_SZ];

// init fast exp(-x) table
void fast_expn_init()
{
	int k  ;
	for(k = 0 ; k < EXPN_SZ + 1 ; ++ k) 
	{
		expn_tab[k] = exp (-(double)k * ((double)EXPN_MAX / (double)EXPN_SZ)) ;
	}
}

// input x: in the range [0, EXPN_MAX]
// return : exp(-x)
double fast_expn(double x)
{
	double a,b,r ;
	int i ;
	if ( x < 0 || x > EXPN_MAX )
	{
		AfxMessageBox("Parameter error in \"fast_expn\"");
		return -100;
	}

	x *= EXPN_SZ / EXPN_MAX ;
	i = floor_d (x) ;
	r = x - i ;
	a = expn_tab [i    ] ;
	b = expn_tab [i + 1] ;
	return a + r * (b - a) ;
}

// init fast cos(x) table
void fast_cosn_init()
{
	int k  ;
	for(k = 0 ; k < COSN_SZ + 1 ; ++ k) 
	{
		cosn_tab[k] = cos((double)k * ((double)COSN_MAX / (double)COSN_SZ)) ;
	}
}

// input x: in the range [0, COSN_MAX]
// return : cos(abs(x))
double fast_cosn(double x)
{
	double a,b,r ;
	int i ;

	if ( x < 0 )	x = -x;
	
	if ( x < 0 || x > COSN_MAX)
	{
		AfxMessageBox("Parameter error in \"fast_cosn\"");
		return -100;
	}

	x *= COSN_SZ / COSN_MAX ;
	i = floor_d (x) ;
	r = x - i ;
	a = cosn_tab [i    ] ;
	b = cosn_tab [i + 1] ;
	return a + r * (b - a) ;
}

int floor_d(double x)
{
	int xi = (int) x ;
	if (x >= 0 || (double) xi == x) return xi ;
	else return xi - 1 ;
}

//////////////////////////////////////////////////////////////////////////

bool LoadTactionParam(const char* param_fn, TACTION_PARAM& param)
{
	char param_term[128];
	char param_value[512];
	FILE* fp_param = NULL;

	fp_param = fopen(param_fn, "rt");

	fscanf(fp_param, "%s : %s\n", param_term, param_value);	// training -- type: codebook || feature
	param.train_type = param_value;
	fscanf(fp_param, "%s : %s\n", param_term, param_value);	// testing -- codebook filename
	param.codebook_fn = param_value;
	fscanf(fp_param, "%s : %s\n", param_term, param_value);	// testing -- classifier filename for celltoear event
	param.celltoear_classifier_fn = param_value;
	fscanf(fp_param, "%s : %s\n", param_term, param_value);	// testing -- classifier filename for objectput event
	param.objectput_classifier_fn = param_value;
	fscanf(fp_param, "%s : %s\n", param_term, param_value);	// testing -- classifier filename for pointing event
	param.pointing_classifier_fn = param_value;

	fclose(fp_param);

	return true;
}

bool LoadTactionBoFCodebook(const char* codebook_fn, BOF_CODEBOOK& codebook)
{
	int i_size, i_dim;

	if (NULL != codebook.codebook_data)	delete [] codebook.codebook_data;
	if (NULL != codebook.codebook_length)	delete [] codebook.codebook_length;

	FILE* fp_codebook = NULL;
	fopen_s(&fp_codebook, codebook_fn, "rb");

	// load codebook info.
	fscanf_s(fp_codebook, "%d,%d\n", &codebook.codebook_size, &codebook.codebook_dim);
	codebook.codebook_data = new float[codebook.codebook_dim * codebook.codebook_size];

	// load codebook data
	int idx = 0;
	for (i_size = 0; i_size < codebook.codebook_size; i_size++)
	{
		for (i_dim = 0; i_dim < codebook.codebook_dim; i_dim++)
		{
			double cur_data;
			fscanf_s(fp_codebook, "%lf,", &cur_data);
			codebook.codebook_data[idx] = (float)cur_data;
			idx++;
		}
		fscanf_s(fp_codebook, "\n");
	}
	fclose(fp_codebook);

	// calculate length of each codebook word
	codebook.codebook_length = new float[codebook.codebook_size];
	ZeroMemory(codebook.codebook_length, codebook.codebook_size*sizeof(float));	
	for (i_size = 0; i_size < codebook.codebook_size; i_size++)
	{
		for (i_dim = 0; i_dim < codebook.codebook_dim; i_dim++)
		{
			codebook.codebook_length[i_size] += codebook.codebook_data[i_size * codebook.codebook_dim + i_dim] * codebook.codebook_data[i_size * codebook.codebook_dim + i_dim];
		}
	}

	return true;
}

bool LoadTactionClassifier(const char* classifier_fn, TACTION_CLASSIFIER &classifier)
{
	FILE* fp_classifier = NULL;
	fp_classifier = fopen(classifier_fn, "rt");

	// model type: 0 - SVM only, 1 - SVM+Sigmoid
	fscanf(fp_classifier, "%d\n", &classifier.model_type);

	// w matrix (h = trans(w) * x + b)
	fscanf(fp_classifier, "%d\n", &classifier.w_dim);
	if (classifier.w_dim <= 0)
	{
		AfxMessageBox("parameter error in LoadClassifier");
		return false;
	}
	classifier.classifier_w = new float[classifier.w_dim];
	for (int i_dim = 0; i_dim < classifier.w_dim; i_dim++)
	{
		float cur_value;
		fscanf(fp_classifier, "%f\n", &cur_value);
		classifier.classifier_w[i_dim] = cur_value;
	}
//	fscanf(fp_classifier, "\n");

	// b value (h = trans(w) * x + b)
	fscanf(fp_classifier, "%f\n", &classifier.classifier_b);

	// a, c for sigmoid function (sig = 1 / (1 + exp(a*x+c))
	if (1 == classifier.model_type)
	{
		fscanf(fp_classifier, "%f\n", &classifier.sigmoid_a);
		fscanf(fp_classifier, "%f\n", &classifier.sigmoid_c);
	}
	else
	{
		classifier.sigmoid_a = 0;
		classifier.sigmoid_c = 0;
	}

	fclose(fp_classifier);
	return true;
}

bool GetBlockIndex(const RECT sample, LOCAL_PARTITION& block_index)
{
	block_index.local_index.clear();
	int sample_w = sample.right - sample.left + 1;
	int sample_h = sample.bottom - sample.top + 1;
	int patch_size = block_index.patch_size;
	double overlap_percent = block_index.overlap_percent;

	if (sample_w < patch_size || sample_h < patch_size || overlap_percent >= 1 || overlap_percent < 0)
	{
		AfxMessageBox("Parameter error in \"GetBlockIndex\"");
		return false;
	}

	int overlap_size = (int)(patch_size * overlap_percent);
	int block_w_num = (sample_w - overlap_size) / (patch_size - overlap_size);
	int block_h_num = (sample_h - overlap_size) / (patch_size - overlap_size);

	for (int i = 0; i < block_w_num; i++)
	{
		for (int j = 0; j < block_h_num; j++)
		{
			POINT_2D cur_block;
			cur_block.x = i * (patch_size - overlap_size); // + sample.left;
			cur_block.y = j * (patch_size - overlap_size); // + sample.top;
			block_index.local_index.push_back(cur_block);
//			if (cur_block.x + (patch_size - 1) >= sample_w /*sample.right*/ || cur_block.y + (patch_size - 1) >= sample_h /*sample.bottom*/)
//			{
//				AfxMessageBox("Block index error in \"GetBlockIndex\"");
//				return false;
//			}			
		}
	}

	return true;
}

bool GetBlockIndex_MultiScale(const RECT sample, LOCAL_PARTITION_MULTISCALE& multiscale_block_index)
{
	int i_scale;

	int cnt_scale_num = multiscale_block_index.size();
	if (cnt_scale_num <= 0)
	{
		AfxMessageBox("Parameter error in \"GetBlockIndex_MultiScale\"");
		return false;
	}

	int sample_w = sample.right  - sample.left + 1;
	int sample_h = sample.bottom - sample.top  + 1;

	vector<double> list_patch_size, list_step_size;
	for (i_scale = 0; i_scale < cnt_scale_num; i_scale++)
	{
		int patch_size = multiscale_block_index[i_scale].patch_size;
		int step_size  = multiscale_block_index[i_scale].step_size;

		if (patch_size <= 0 || step_size <= 0 || sample_w < patch_size || sample_h < patch_size)
		{
			AfxMessageBox("Parameter error in \"GetBlockIndex_MultiScale\"");
			return false;
		}

		multiscale_block_index[i_scale].local_index.clear();
		list_patch_size.push_back(patch_size);
		list_step_size.push_back(step_size);
	}

	int margin		= (int)( MaxElementInVector(list_patch_size) / 2.0 );
	int step_size	= (int)( MinElementInVector(list_step_size) );
	int grid_w		= (int)( (sample_w - margin * 2.0) / step_size + 1);
	int grid_h		= (int)( (sample_h - margin * 2.0) / step_size + 1);

	for (int i_w = 0; i_w < grid_w; i_w++)
	{
		for (int i_h = 0; i_h < grid_h; i_h++)
		{
			for (i_scale = 0; i_scale < cnt_scale_num; i_scale++)
			{
				POINT_2D cur_block;
				cur_block.x = margin + i_w * step_size - (int)((multiscale_block_index[i_scale].patch_size) / 2.0);
				cur_block.y = margin + i_h * step_size - (int)((multiscale_block_index[i_scale].patch_size) / 2.0);
				multiscale_block_index[i_scale].local_index.push_back(cur_block);
			}
		}
	}

	return true;
}

bool NormalizeFeaVector(vector<float>& fea_vec)
{
	if (fea_vec.size() <= 0)
		return false;

	int i_cop, cnt_cop_num = fea_vec.size();
	float vec_sum = 0;

	for (i_cop = 0; i_cop < cnt_cop_num; i_cop++)
		vec_sum += fea_vec[i_cop];

	if (vec_sum <= MATH_ZERO)
	{
		for (i_cop = 0; i_cop < cnt_cop_num; i_cop++)
			fea_vec[i_cop] = 0;
	}
	else
	{
		for (i_cop = 0; i_cop < cnt_cop_num; i_cop++)
			fea_vec[i_cop] /= vec_sum;
	}

	return true;
}

bool NormalizeFeaVector_SPM(vector<float>& fea_vec, int step)
{
	if (fea_vec.size() <= 0 || step <= 0)
		return false;

	int i_cop, cnt_cop_num = fea_vec.size();
	float vec_sum = 0;

	for (i_cop = 0; i_cop < cnt_cop_num; i_cop++)
		vec_sum += fea_vec[i_cop];

	vec_sum = vec_sum / step;

	if (vec_sum <= MATH_ZERO)
	{
		for (i_cop = 0; i_cop < cnt_cop_num; i_cop++)
			fea_vec[i_cop] = 0;
	}
	else
	{
		for (i_cop = 0; i_cop < cnt_cop_num; i_cop++)
			fea_vec[i_cop] /= vec_sum;
	}

	return true;
}

double MaxElementInVector(vector<double>& data)
{
	if (data.size() <= 0)
	{
		return -1000000;
	}

	double max_value = data[0];
	for(int i = 1; i < (int)data.size(); i++)
	{
		max_value = (data[i] > max_value) ? data[i] : max_value;
	}

	return max_value;
};

double MinElementInVector(vector<double>& data)
{
	if (data.size() <= 0)
	{
		return 1000000;
	}

	double min_value = data[0];
	for(int i = 1; i < (int)data.size(); i++)
	{
		min_value = (data[i] < min_value) ? data[i] : min_value;
	}

	return min_value;
};