#include "stdafx.h"
#include ".\moduletemporalaction.h"
#include "../../../include/cv/highgui.h"

#include "../trecvid/utilities.h"

#define ENABLE_TIMING 1
#include "timing.h"

//////////////////////////////////////////////////////////////////////////
string taction_folder = "\\taction\\";
string codebook_fea_fn = "_taction_codebook.fea";
string codebook_inf_fn = "_taction_codebook.inf";
string train_fea_fn = "_taction_train.fea";
string train_lab_fn = "_taction_train.lab";
string train_inf_fn = "_taction_train.inf";

//////////////////////////////////////////////////////////////////////////


CModuleTemporalAction::CModuleTemporalAction(string& par_fn, bool bEvaluation):CTrecvidAnalyst(par_fn, bEvaluation)
{
	m_bEvaluation = bEvaluation;
	LoadTactionParam(par_fn.c_str(), taction_param);
	InitParam();
}

CModuleTemporalAction::~CModuleTemporalAction(void)
{
	Release();
}

void CModuleTemporalAction::Reset(void)
{
	Release();		// release memory
	InitParam();	// initialize parameters
}

void CModuleTemporalAction::Release(void)
{
	ReleaseFrameBuffer();
	ReleaseBoFCodebook();
	ReleaseClassifier();
}

void CModuleTemporalAction::ReleaseBoFCodebook(void)
{
	if (NULL != m_codebook.codebook_data)	// release codebook data
	{
		delete [] m_codebook.codebook_data;
		m_codebook.codebook_data = NULL;
	}
	if (NULL != m_codebook.codebook_length)	// release codebook length buffer
	{
		delete [] m_codebook.codebook_length;
		m_codebook.codebook_length = NULL;
	}
}

void CModuleTemporalAction::ReleaseFrameBuffer(void)
{
	if ( m_frame_buffer.size() > 0 )	// release frame buffer
	{
		list<IplImage*>::iterator it;
		for (it = m_frame_buffer.begin(); it!=m_frame_buffer.end(); ++it)
		{
			IplImage* img_color = *it;
			cvReleaseImage(&img_color);
		}
		m_frame_buffer.clear();
	}

	if ( m_gradient_buffer.size() > 0 )	// release gradient buffer
	{
		list< vector<IplImage*> >::iterator it;
		for (it = m_gradient_buffer.begin(); it != m_gradient_buffer.end(); ++it)
		{
			vector<IplImage*>& cur_gradient = *it;
			vector<IplImage*>::iterator ig;
			for (ig = cur_gradient.begin(); ig != cur_gradient.end(); ++ig)
			{
				IplImage* gradient_img = *ig;
				cvReleaseImage(&gradient_img);
			}
			
		}
		m_gradient_buffer.clear();
	}

	if ( m_optflow_buffer.size() > 0 )	// release optical flow buffer
	{
		list< vector<IplImage*> >::iterator it;
		for (it = m_optflow_buffer.begin(); it != m_optflow_buffer.end(); ++it)
		{
			vector<IplImage*>& cur_optflow = *it;
			vector<IplImage*>::iterator ig;
			for (ig = cur_optflow.begin(); ig != cur_optflow.end(); ++ig)
			{
				IplImage* optflow_img = *ig;
				cvReleaseImage(&optflow_img);
			}

		}
		m_optflow_buffer.clear();
	}

	m_bTemporalInit = false;
	m_bTemporalReady = false;
}

void CModuleTemporalAction::ReleaseClassifier(void)
{
	if (NULL != m_celltoear_classifier.classifier_w)
	{
		delete [] m_celltoear_classifier.classifier_w;
		m_celltoear_classifier.classifier_w = NULL;
	}
	if (NULL != m_objectput_classifier.classifier_w)
	{
		delete [] m_objectput_classifier.classifier_w;
		m_objectput_classifier.classifier_w = NULL;
	}
	if (NULL != m_pointing_classifier.classifier_w)
	{
		delete [] m_pointing_classifier.classifier_w;
		m_pointing_classifier.classifier_w = NULL;
	}
}

void CModuleTemporalAction::InitParam(void)
{
	m_bGradientBuffer    = true;	// if using/computing gradient feature
	m_bOpticalFlowBuffer = true;	// if using/computing optical flow feature

	m_bTemporalInit = false;	// init flag for buffering
	m_bTemporalReady = false;	// ready flag for buffering	

	// init training type -- codebook training || feature extraction
	if (!m_bEvaluation)
	{
		if ("codebook" == taction_param.train_type)
		{
			m_TrainType = TRAIN_TACTION_CODEBOOK;
		}
		else if ("trainfea" == taction_param.train_type)
		{
			m_TrainType = TRAIN_TACTION_FEATURE;
			InitCodebook(taction_param.codebook_fn.c_str());
		}
		else
		{
			AfxMessageBox("Parameter error for training type in \"InitParam\"");
			return;
		}
	}

	// init test type
	if (m_bEvaluation)
	{
		InitCodebook(taction_param.codebook_fn.c_str());	// init codebook
		InitClassifier(taction_param);
	}
}

bool CModuleTemporalAction::InitCodebook(const char* codebook_fn)
{	
	// load codebook data
	LoadTactionBoFCodebook(codebook_fn, m_codebook);	
	return true;
}

bool CModuleTemporalAction::InitClassifier(TACTION_PARAM& taction_param)
{	
	// load taction classifier
	LoadTactionClassifier(taction_param.celltoear_classifier_fn.c_str(), m_celltoear_classifier);
	LoadTactionClassifier(taction_param.objectput_classifier_fn.c_str(), m_objectput_classifier);
	LoadTactionClassifier(taction_param.pointing_classifier_fn.c_str(), m_pointing_classifier);
	return true;
}

void CModuleTemporalAction::SetVideoFn(const string& video_fn)
{
	CTrecvidAnalyst::SetVideoFn(video_fn);

	// Init output data & files
	Reset();
	InitOutput();
}

void CModuleTemporalAction::InitOutput(void)
{
	string video_path, video_name;
	GetVideoPath(video_path);		GetVideoName(video_name);

	string file_path = video_path;
	file_path.append(taction_folder);
	CreateDirectory(file_path.c_str(), NULL);

	if (!m_bEvaluation)		// training process
	{
		if (TRAIN_TACTION_CODEBOOK == m_TrainType)	// codebook
		{
			m_codebook_fea_fn = file_path;
			m_codebook_fea_fn.append(video_name);	m_codebook_fea_fn.append(codebook_fea_fn);
			m_codebook_inf_fn = file_path;
			m_codebook_inf_fn.append(video_name);	m_codebook_inf_fn.append(codebook_inf_fn);

			if (FileExist(m_codebook_fea_fn.c_str()))	DeleteFile(m_codebook_fea_fn.c_str());
			if (FileExist(m_codebook_inf_fn.c_str()))	DeleteFile(m_codebook_inf_fn.c_str());

			m_output_feature_codebook = 0;
		}

		if (TRAIN_TACTION_FEATURE == m_TrainType)	// train feature
		{
			m_train_fea_fn = file_path;
			m_train_fea_fn.append(video_name);		m_train_fea_fn.append(train_fea_fn);
			m_train_lab_fn = file_path;
			m_train_lab_fn.append(video_name);		m_train_lab_fn.append(train_lab_fn);
			m_train_inf_fn = file_path;
			m_train_inf_fn.append(video_name);		m_train_inf_fn.append(train_inf_fn);

			if (FileExist(m_train_fea_fn.c_str()))		DeleteFile(m_train_fea_fn.c_str());
			if (FileExist(m_train_lab_fn.c_str()))		DeleteFile(m_train_lab_fn.c_str());
			if (FileExist(m_train_inf_fn.c_str()))		DeleteFile(m_train_inf_fn.c_str());

			m_output_train_feature = 0;
		}
	}
}

bool CModuleTemporalAction::Analyze(bool		         bEvaluation,            // true: evaluation; false: training
									const int            frm_no,                 // current frame number
									LIST_INT             param_event_id,         // event id to be analyzed
									LIST_INT&            list_event_id,          // list of event id for the candidate regions
									LIST_INT&            list_event_inst_id,     // list of event instance id for training regions
									LIST_INT&            list_event_inst_obj_id, // list of object id in an event instance for training regions 
									LIST_FLOAT&          list_event_prob,        // list of event prob for the candidate regions
									LIST2_FLOAT&         list_event_all_prob,    // list of prob for all the events detected
									LIST_RECT&           list_event_region,      // list of candidate regions
									const BYTE*          img_color,              // the input frame
									const LIST_BLOB&     list_detection,         // people detection results given by nms if available
									const LIST_TRACK_2D& list_track,             // head tracking results given by tracking_2d if available
									const LIST_RECT&     fg_blob)				   // foreground blobs if available
{
	m_frm_no = frm_no;	// set current frame no

	PreProcess(m_frm_no, img_color);	// preprocessing

	if (!bEvaluation) // training process
	{
		if (m_bTemporalReady)
		{
			Training(param_event_id, list_event_id, list_event_inst_id, list_event_inst_obj_id, list_event_prob, list_event_region, list_detection, list_track, fg_blob);
		}
	}

	if (bEvaluation) // testing process
	{
		if (m_bTemporalReady)
		{
			Testing(param_event_id, list_event_id, list_event_prob, list_event_all_prob, list_event_region, list_detection, list_track, fg_blob);
		}		
	}

	return true;
}

bool CModuleTemporalAction::PreProcess(int frm_no, const BYTE* img_color)
{
	if (frm_no < 0 || NULL == img_color)
	{
		AfxMessageBox("Parameter error in \"PreProcess\"");
		return false;
	}

	TemporalBuffer(frm_no, img_color);	// temporal (frame/gradient/optical flow) buffering

	return true;
}

bool CModuleTemporalAction::TemporalBuffer(int frm_no, const BYTE* img_color)
{
	if (frm_no < 0 || NULL == img_color)
	{
		AfxMessageBox("Parameter error in \"FrameBuffer\"");
		return false;
	}

	int frm_wid = CVideoSource::GetImageWidth();
	int frm_hei = CVideoSource::GetImageHeight();

	if ( !m_bTemporalInit )
	{
		// original frame buffering
		FrameBufferInit(frm_wid, frm_hei);

		// gradient frame buffering		
		if (m_bGradientBuffer)
			GradientBufferInit(frm_wid, frm_hei);

		// optical flow buffering
		if (m_bOpticalFlowBuffer)
			OpticalFlowBufferInit(frm_wid, frm_hei);

		m_cur_buffer_len = 0;
		m_bTemporalInit = true;
	}

	if (m_bTemporalInit)
	{
		// frame buffering
		FrameBuffer(frm_wid, frm_hei, img_color);

		// gradient buffering
		if (m_bGradientBuffer)
			GradientBuffer(frm_wid, frm_hei, img_color);

		// optical flow buffering
		if (m_bOpticalFlowBuffer)
			OpticalFlowBuffer(frm_wid, frm_hei, img_color);

		//////////////////////////////////////////////////////////////////////////
		// for debug
		//int Scale = 1;
		//int Thickness = 1;
		//CvFont Font1=cvFont(Scale, Thickness);
		//CvPoint TextPosition1 = cvPoint(200,300);
		//CvScalar Color = CV_RGB(255,0,0);

		//char img_text[128];		sprintf(img_text, "Frame %d", frm_no);
		//cvPutText(*(m_frame_buffer.begin()), img_text, TextPosition1, &Font1, Color);
		// for debug
		//////////////////////////////////////////////////////////////////////////		
		m_cur_buffer_len++;
		if ( 1 == frm_no - m_last_frm_no )
		{
			if (FRAME_BUFFER_LEN == m_cur_buffer_len)
			{
				m_cur_buffer_len--;
				m_bTemporalReady = true;
			} 
		}
		else
		{
			m_cur_buffer_len = 1;
			m_bTemporalReady = false;
		}
	}

	m_last_frm_no = frm_no;
	//////////////////////////////////////////////////////////////////////////
	// for debug
	//list<IplImage*>::iterator it = m_frame_buffer.begin();
	//list< vector<IplImage*> >::iterator it = m_gradient_buffer.begin();
	//list< vector<IplImage*> >::iterator it = m_optflow_buffer.begin();

	//cvNormalize((*it)[0], (*it)[0], 0, 1, CV_MINMAX);
	//cvNamedWindow("frame 1", 1);
	//cvShowImage("frame 1", (*it)[0]);

	//it++;	it++;	it++;
	//cvNormalize((*it)[1], (*it)[1], 0, 1, CV_MINMAX);
	//cvNamedWindow("frame 2", 1);
	//cvShowImage("frame 2", (*it)[1]);

	//it++;	it++;	it++;
	//cvNamedWindow("frame 3", 1);
	//cvShowImage("frame 3", (*it)[0]);

	//cvWaitKey(1);
	// for debug
	//////////////////////////////////////////////////////////////////////////	 
	return true;
}

bool CModuleTemporalAction::FrameBufferInit(int frm_wid, int frm_hei)
{
	if (frm_wid <= 0 || frm_hei <= 0)
	{
		AfxMessageBox("Parameter error in \"FrameBufferInit\"");
		return false;
	}

	for (int i_buf = 0; i_buf < FRAME_BUFFER_LEN; i_buf++)
	{
		IplImage* ipl_color_img = cvCreateImage(cvSize(frm_wid, frm_hei), IPL_DEPTH_8U, 3);
		m_frame_buffer.push_back(ipl_color_img);
	}

	return true;
}

bool CModuleTemporalAction::FrameBuffer(int frm_wid, int frm_hei, const BYTE* img_color)
{
	if (frm_wid <= 0 || frm_hei <= 0 || NULL == img_color)
	{
		AfxMessageBox("Parameter error in \"FrameBuffer\"");
		return false;
	}

	IplImage* color_frame = cvCreateImageHeader(cvSize(frm_wid, frm_hei), IPL_DEPTH_8U, 3);
	color_frame->imageData = (char*)img_color;

	IplImage* color_img = (IplImage*)(*m_frame_buffer.rbegin());
	m_frame_buffer.pop_back();
	m_frame_buffer.push_front(color_img);
	cvCopyImage(color_frame, color_img);
	cvFlip(color_img, NULL, 0);
	cvReleaseImageHeader(&color_frame);

	return true;
}

bool CModuleTemporalAction::GradientBufferInit(int frm_wid, int frm_hei)
{
	if (frm_wid <= 0 || frm_hei <= 0)
	{
		AfxMessageBox("Parameter error in \"GradientBufferInit\"");
		return false;
	}

	for (int i_buf = 0; i_buf < FRAME_BUFFER_LEN; i_buf++)
	{
		vector<IplImage*> gradient_img;
		IplImage* ipl_gradient_x_img = cvCreateImage(cvSize(frm_wid, frm_hei), IPL_DEPTH_32F, 1);
		IplImage* ipl_gradient_y_img = cvCreateImage(cvSize(frm_wid, frm_hei), IPL_DEPTH_32F, 1);
		gradient_img.push_back(ipl_gradient_x_img);		gradient_img.push_back(ipl_gradient_y_img);
		m_gradient_buffer.push_back(gradient_img);
	}

	return true;
}

bool CModuleTemporalAction::GradientBuffer(int frm_wid, int frm_hei, const BYTE* img_color)
{
	if (frm_wid <= 0 || frm_hei <= 0 || NULL == img_color)
	{
		AfxMessageBox("Parameter error in \"GradientBuffer\"");
		return false;
	}

	vector<IplImage*> cur_gradient_img = (vector<IplImage*>)(*m_gradient_buffer.rbegin());
	m_gradient_buffer.pop_back();
	m_gradient_buffer.push_front(cur_gradient_img);

	m_taction.ColorImageGradient(img_color, frm_wid, frm_hei, cur_gradient_img);

	return true;
}

bool CModuleTemporalAction::OpticalFlowBufferInit(int frm_wid, int frm_hei)
{
	if (frm_wid <= 0 || frm_hei <= 0)
	{
		AfxMessageBox("Parameter error in \"OpticalFlowBufferInit\"");
		return false;
	}

	for (int i_buf = 0; i_buf < FRAME_BUFFER_LEN; i_buf++)
	{
		vector<IplImage*> optflow_img;
		IplImage* ipl_optflow_x_img = cvCreateImage(cvSize(frm_wid, frm_hei), IPL_DEPTH_32F, 1);
		IplImage* ipl_optflow_y_img = cvCreateImage(cvSize(frm_wid, frm_hei), IPL_DEPTH_32F, 1);
		optflow_img.push_back(ipl_optflow_x_img);		optflow_img.push_back(ipl_optflow_y_img);
		m_optflow_buffer.push_back(optflow_img);
	}

	return true;
}

bool CModuleTemporalAction::OpticalFlowBuffer(int frm_wid, int frm_hei, const BYTE* img_color)
{
	if (frm_wid <= 0 || frm_hei <= 0 || NULL == img_color)
	{
		AfxMessageBox("Parameter error in \"OpticalFlowBuffer\"");
		return false;
	}

	vector<IplImage*> cur_optflow_img = (vector<IplImage*>)(*m_optflow_buffer.rbegin());
	m_optflow_buffer.pop_back();
	m_optflow_buffer.push_front(cur_optflow_img);

	if (0 == m_cur_buffer_len)
	{
		cvZero(cur_optflow_img[0]);		cvZero(cur_optflow_img[1]);
	}
	else
	{
		list<IplImage*>::iterator it = m_frame_buffer.begin();
		IplImage* cal_img = (IplImage*)(*it);	IplImage* ref_img = (IplImage*)(*(++it));	
		m_taction.ColorImageOpticalFlow(cal_img, ref_img, cur_optflow_img);
	}

	return true;
}

bool CModuleTemporalAction::Training(LIST_INT&				param_event_id,    
									 LIST_INT&				list_event_id,
									 LIST_INT&				list_event_inst_id,    
									 LIST_INT&				list_event_inst_obj_id,
									 LIST_FLOAT&			list_event_prob,    
									 LIST_RECT&				list_event_region,  
									 const LIST_BLOB&		list_detection, 
									 const LIST_TRACK_2D&	list_track,
									 const LIST_RECT&		fg_blob)
{
	LIST_INT  train_event_id;
	LIST_RECT train_event_region;
	LIST_CHAR train_event_source;
    
	if (TRAIN_TACTION_CODEBOOK == m_TrainType)
	{
		bool detection_used = false;		
		bool tracking_used  = false;
		int cnt_positive = 0, cnt_negative = 0;
		GetTrainSample(param_event_id, list_event_id, list_event_region, list_detection, list_track, 
					   train_event_id, train_event_region, train_event_source, cnt_positive, cnt_negative, detection_used, tracking_used);

		// extract features and save data for codebook training
		if (cnt_positive || cnt_negative)
		{
//			ExtractCodebookFeature(param_event_id, train_event_id, train_event_region);
			ExtractCodebookFeature_MultiScale(param_event_id, train_event_id, train_event_region);
		}
	}

	if (TRAIN_TACTION_FEATURE == m_TrainType)
	{
		bool detection_used = true;		
		bool tracking_used  = true; // false;
		int cnt_positive = 0, cnt_negative = 0;
		GetTrainSample(param_event_id, list_event_id, list_event_region, list_detection, list_track, 
					   train_event_id, train_event_region, train_event_source, cnt_positive, cnt_negative, detection_used, tracking_used);

		// extract features and save data for model training
		if (cnt_positive || cnt_negative)
		{
//			ExtractTrainFeature(param_event_id, train_event_id, train_event_region);
			ExtractTrainFeature_MultiScale(param_event_id, train_event_id, train_event_region);
		}
	}

	return true;
}

bool CModuleTemporalAction::ExtractTrainFeature_MultiScale(const LIST_INT& param_event_id, const LIST_INT& train_event_id, const LIST_RECT& train_event_region)
{
	int cnt_event = param_event_id.size();		int cnt_region = train_event_region.size();		int cnt_id = train_event_id.size();
	if (cnt_event <= 0 || cnt_region <= 0 || cnt_region != cnt_id)
	{
		AfxMessageBox("parameter error in \"ExtractTrainFeature\"");
		return false;
	}

	FEATURE_MATRIX feature_matrix;		feature_matrix.clear();
	LIST_INT	   feature_label;		feature_label.clear();

	for (int i_region = 0; i_region < cnt_region; i_region++)
	{
		RECT cur_roi = train_event_region[i_region];
		int	 cur_id  = train_event_id[i_region];

		// do perturbation for samples
		bool pos_perburbation = true;
		bool neg_perburbation = false;
		LIST_RECT list_generate_sample;		LIST_INT  list_generate_id;	
		GeneratePerturbationSample(cur_roi, cur_id, pos_perburbation, neg_perburbation, list_generate_sample, list_generate_id);

		// compute & output features for each sample
		for (int i_sample = 0; i_sample < (int)list_generate_sample.size(); i_sample++)
		{
			RECT roi_exp;
			if ( !ROIExpand(list_generate_sample[i_sample], roi_exp) )	// region expansion
				continue;

			// get block index
			LOCAL_PARTITION_MULTISCALE multiscale_block_index;
			SetBlockIndexParam_MultiScale(multiscale_block_index);
			if ( !GetBlockIndex_MultiScale(roi_exp, multiscale_block_index) )
				continue;

			// get feature for block-cubes
			FEATURE_VECTOR feature_vector;			feature_vector.clear();
			ExtractActionFeature_MultiScale(roi_exp, multiscale_block_index, feature_vector);

			// gather feature vectors
			feature_matrix.push_back(feature_vector);
			feature_label.push_back(list_generate_id[i_sample]);
		}
	}

	// output feature to file
	if (feature_matrix.size() > 0)
		OutputFeature_Train(feature_matrix, feature_label);

	return true;
}

bool CModuleTemporalAction::ExtractActionFeature_MultiScale(const RECT roi_region, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, FEATURE_VECTOR& sample_feature)
{
	sample_feature.clear();

	vector<FEATURE_MATRIX*> list_feature;
	FEATURE_MATRIX gradient_block_feature;		gradient_block_feature.clear();
	FEATURE_MATRIX optflow_block_feature;		optflow_block_feature.clear();

	// extract block-wise feature
	if (m_bGradientBuffer)
	{
		CodebookFeature_MultiScale_ImageGradient(roi_region, multiscale_block_index, gradient_block_feature);
		list_feature.push_back(&gradient_block_feature);
	}

	if (m_bOpticalFlowBuffer)
	{
		CodebookFeature_MultiScale_OpticalFlow(roi_region, multiscale_block_index, optflow_block_feature);
		list_feature.push_back(&optflow_block_feature);
	}

	// form combined feature vector
	FEATURE_MATRIX block_feature_matrix;
	CombineMultiFeature(list_feature, block_feature_matrix);

	// calculate multiscale spm histograms
	CalcFeatureRepresentation_SPM(roi_region, multiscale_block_index, block_feature_matrix, sample_feature);

	return true;
}

bool CModuleTemporalAction::CalcFeatureRepresentation_SPM(const RECT roi_region, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, FEATURE_MATRIX& block_feature_matrix, FEATURE_VECTOR& feature)
{
	int cnt_scale_num = multiscale_block_index.size();
	if (cnt_scale_num <= 0)
	{
		AfxMessageBox("Parameter error in \"CalcFeatureRepresentation_SPM\"");
		return false;
	}

	int cnt_point_num = multiscale_block_index[0].local_index.size();

	vector<POINT_2D> idx_pos;
	for (int i_pos = 0; i_pos < cnt_point_num; i_pos++)
	{
		for (int i_scale = 0; i_scale < cnt_scale_num; i_scale++)
		{
			POINT_2D cur_idx_pos;
			cur_idx_pos.x = multiscale_block_index[i_scale].local_index[i_pos].x + (int)((multiscale_block_index[i_scale].patch_size) / 2.0);
			cur_idx_pos.y = multiscale_block_index[i_scale].local_index[i_pos].y + (int)((multiscale_block_index[i_scale].patch_size) / 2.0);
			idx_pos.push_back(cur_idx_pos);
		}
	}
	int image_w = roi_region.right - roi_region.left + 1;
	int image_h = roi_region.bottom - roi_region.top + 1;
	
	feature.clear();
	m_taction.CalcBoFHistogram_SPM(m_codebook, idx_pos, image_w, image_h, block_feature_matrix, feature);
	return true;
}

bool CModuleTemporalAction::ExtractTrainFeature(const LIST_INT& param_event_id, const LIST_INT& train_event_id, const LIST_RECT& train_event_region)
{
	int cnt_event = param_event_id.size();		int cnt_region = train_event_region.size();		int cnt_id = train_event_id.size();
	if (cnt_event <= 0 || cnt_region <= 0 || cnt_region != cnt_id)
	{
		AfxMessageBox("parameter error in \"ExtractTrainFeature\"");
		return false;
	}

	FEATURE_MATRIX feature_matrix;		feature_matrix.clear();
	LIST_INT	   feature_label;		feature_label.clear();

	for (int i_region = 0; i_region < cnt_region; i_region++)
	{
		RECT cur_roi = train_event_region[i_region];
		int	 cur_id  = train_event_id[i_region];

		// do perturbation for samples
		bool pos_perburbation = true;
		bool neg_perburbation = false;
		LIST_RECT list_generate_sample;		LIST_INT  list_generate_id;	
		GeneratePerturbationSample(cur_roi, cur_id, pos_perburbation, neg_perburbation, list_generate_sample, list_generate_id);

		// compute & output features for each sample
		for (int i_sample = 0; i_sample < (int)list_generate_sample.size(); i_sample++)
		{
			RECT roi_exp;
			if ( !ROIExpand(list_generate_sample[i_sample], roi_exp) )	// region expansion
				continue;

			// get block index
			LOCAL_PARTITION block_index;
			block_index.patch_size = TACTION_BLOCK_PATCH_SIZE;		
			block_index.overlap_percent = TACTION_BLOCK_PATCH_OVERLAP;
			if ( !GetBlockIndex(roi_exp, block_index) )
				continue;

			// get feature for block-cubes
			FEATURE_VECTOR feature_vector;			feature_vector.clear();
			ExtractActionFeature(roi_exp, block_index, feature_vector);

			// gather feature vectors
			feature_matrix.push_back(feature_vector);
			feature_label.push_back(list_generate_id[i_sample]);
		}
	}

	// output feature to file
	if (feature_matrix.size() > 0)
		OutputFeature_Train(feature_matrix, feature_label);

	return true;
}

bool CModuleTemporalAction::ExtractActionFeature(const RECT roi_region, LOCAL_PARTITION& block_index, FEATURE_VECTOR& sample_feature)
{
	sample_feature.clear();

	vector<FEATURE_MATRIX*> list_feature;
	FEATURE_MATRIX gradient_block_feature;		gradient_block_feature.clear();
	FEATURE_MATRIX optflow_block_feature;		optflow_block_feature.clear();

	// extract block-wise feature
	if (m_bGradientBuffer)
	{
		CodebookFeature_ImageGradient(roi_region, block_index, gradient_block_feature);
		list_feature.push_back(&gradient_block_feature);
	}

	if (m_bOpticalFlowBuffer)
	{
		CodebookFeature_OpticalFlow(roi_region, block_index, optflow_block_feature);
		list_feature.push_back(&optflow_block_feature);
	}

	// form combined feature vector
	FEATURE_MATRIX block_feature_matrix;
	CombineMultiFeature(list_feature, block_feature_matrix);

	// output codebook feature to file
	CalcFeatureRepresentation(block_feature_matrix, sample_feature);

	return true;
}

bool CModuleTemporalAction::CalcFeatureRepresentation(FEATURE_MATRIX& block_feature_matrix, FEATURE_VECTOR& feature)
{
	feature.clear();
	m_taction.CalcBoFHistogram(m_codebook, block_feature_matrix, feature);
	return true;
}

bool CModuleTemporalAction::ExtractCodebookFeature(const LIST_INT& param_event_id, const LIST_INT& train_event_id, const LIST_RECT& train_event_region)
{
	int cnt_event = param_event_id.size();		int cnt_region = train_event_region.size();		int cnt_id = train_event_id.size();
	if (cnt_event <= 0 || cnt_region <= 0 || cnt_region != cnt_id)
	{
		AfxMessageBox("parameter error in \"ExtractCodebookFeature\"");
		return false;
	}

	for (int i_region = 0; i_region < cnt_region; i_region++)
	{
		RECT cur_roi = train_event_region[i_region];
		int	 cur_id  = train_event_id[i_region];

		// do perturbation for samples
		bool pos_perburbation = false;
		bool neg_perburbation = false;
		LIST_RECT list_generate_sample;		LIST_INT  list_generate_id;	
		GeneratePerturbationSample(cur_roi, cur_id, pos_perburbation, neg_perburbation, list_generate_sample, list_generate_id);

		// compute & output features for each sample
		for (int i_sample = 0; i_sample < (int)list_generate_sample.size(); i_sample++)
		{
			RECT roi_exp;
			if ( !ROIExpand(list_generate_sample[i_sample], roi_exp) )	// region expansion
				continue;

			// get block index
			LOCAL_PARTITION block_index;
			block_index.patch_size = TACTION_BLOCK_PATCH_SIZE;		
			block_index.overlap_percent = 0.5;
			if ( !GetBlockIndex(roi_exp, block_index) )
				continue;

			// get feature for block-cubes
			vector<FEATURE_MATRIX*> list_feature;

			FEATURE_MATRIX gradient_feature;		gradient_feature.clear();
			FEATURE_MATRIX optflow_feature;			optflow_feature.clear();

			if (m_bGradientBuffer)	// extract feature from image gradient
			{
				CodebookFeature_ImageGradient(roi_exp, block_index, gradient_feature);
				list_feature.push_back(&gradient_feature);
			}

			if (m_bOpticalFlowBuffer) // extract feature from optical flow
			{
				CodebookFeature_OpticalFlow(roi_exp, block_index, optflow_feature);
				list_feature.push_back(&optflow_feature);
			}

			// output codebook feature to file
			OutputFeature_Codebook(list_feature);
		}
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////

// codebook feature for image gradient
bool CModuleTemporalAction::CodebookFeature_ImageGradient(const RECT roi_region, LOCAL_PARTITION& block_index, FEATURE_MATRIX& sample_feature)
{
	return m_taction.TemporalCodebookFeature_ImageGradient(m_gradient_buffer, roi_region, block_index, sample_feature);
}

// codebook feature for optical flow
bool CModuleTemporalAction::CodebookFeature_OpticalFlow(const RECT roi_region, LOCAL_PARTITION& block_index, FEATURE_MATRIX& sample_feature)
{
	return m_taction.TemporalCodebookFeature_OpticalFlow(m_optflow_buffer, roi_region, block_index, sample_feature);
}

//////////////////////////////////////////////////////////////////////////

// multi-scale codebook feature for image gradient
bool CModuleTemporalAction::CodebookFeature_MultiScale_ImageGradient(const RECT roi_region, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, FEATURE_MATRIX& sample_feature)
{
	return m_taction.TemporalCodebookFeature_MultiScale_ImageGradient(m_gradient_buffer, roi_region, multiscale_block_index, sample_feature);
}

// multi-scale codebook feature for optical flow
bool CModuleTemporalAction::CodebookFeature_MultiScale_OpticalFlow(const RECT roi_region, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, FEATURE_MATRIX& sample_feature)
{
	return m_taction.TemporalCodebookFeature_MultiScale_OpticalFlow(m_optflow_buffer, roi_region, multiscale_block_index, sample_feature);
}

//////////////////////////////////////////////////////////////////////////

bool CModuleTemporalAction::ExtractCodebookFeature_MultiScale(const LIST_INT& param_event_id, const LIST_INT& train_event_id, const LIST_RECT& train_event_region)
{
	int cnt_event = param_event_id.size();		int cnt_region = train_event_region.size();		int cnt_id = train_event_id.size();
	if (cnt_event <= 0 || cnt_region <= 0 || cnt_region != cnt_id)
	{
		AfxMessageBox("parameter error in \"ExtractCodebookFeature\"");
		return false;
	}

	for (int i_region = 0; i_region < cnt_region; i_region++)
	{
		RECT cur_roi = train_event_region[i_region];
		int	 cur_id  = train_event_id[i_region];

		// do perturbation for samples
		bool pos_perburbation = false;
		bool neg_perburbation = false;
		LIST_RECT list_generate_sample;		LIST_INT  list_generate_id;	
		GeneratePerturbationSample(cur_roi, cur_id, pos_perburbation, neg_perburbation, list_generate_sample, list_generate_id);

		// compute & output features for each sample
		for (int i_sample = 0; i_sample < (int)list_generate_sample.size(); i_sample++)
		{
			RECT roi_exp;
			if ( !ROIExpand(list_generate_sample[i_sample], roi_exp) )	// region expansion
				continue;

			// get block index
			LOCAL_PARTITION_MULTISCALE multiscale_block_index;
			SetBlockIndexParam_MultiScale(multiscale_block_index);
			if ( !GetBlockIndex_MultiScale(roi_exp, multiscale_block_index) )
				continue;

			// get feature for block-cubes
			vector<FEATURE_MATRIX*> list_feature;

			FEATURE_MATRIX gradient_feature;		gradient_feature.clear();
			FEATURE_MATRIX optflow_feature;			optflow_feature.clear();

			if (m_bGradientBuffer)	// extract feature from image gradient
			{
				CodebookFeature_MultiScale_ImageGradient(roi_exp, multiscale_block_index, gradient_feature);
				list_feature.push_back(&gradient_feature);
			}

			if (m_bOpticalFlowBuffer) // extract feature from optical flow
			{
				CodebookFeature_MultiScale_OpticalFlow(roi_exp, multiscale_block_index, optflow_feature);
				list_feature.push_back(&optflow_feature);
			}

			// output codebook feature to file
			OutputFeature_MultiScale_Codebook(list_feature, (int)multiscale_block_index.size());
		}
	}

	return true;
}

bool CModuleTemporalAction::SetBlockIndexParam_MultiScale(LOCAL_PARTITION_MULTISCALE& multiscale_block_index)
{
	LOCAL_PARTITION scale_1;
	scale_1.patch_size = TACTION_BLOCK_PATCH_SIZE;
	scale_1.step_size  = TACTION_BLOCK_PATCH_SIZE;
	multiscale_block_index.push_back(scale_1);

	LOCAL_PARTITION scale_2;
	scale_2.patch_size = TACTION_BLOCK_PATCH_SIZE * 2;
	scale_2.step_size  = TACTION_BLOCK_PATCH_SIZE;
	multiscale_block_index.push_back(scale_2);

	return true;
}

bool CModuleTemporalAction::GetTrainSample(LIST_INT&			param_event_id,
										   LIST_INT&			list_event_id,
										   LIST_RECT&			list_event_region,  
										   const LIST_BLOB&		list_detection, 
										   const LIST_TRACK_2D&	list_track,
										   LIST_INT&			train_event_id, 
										   LIST_RECT&			train_event_region,
										   LIST_CHAR&			train_event_source,
										   int& cnt_gt_positive, int& cnt_gt_negative, bool detection_used, bool tracking_used)
{
	train_event_id.clear();		train_event_region.clear();

	cnt_gt_positive = 0;		cnt_gt_negative = 0;
	SampleFromGroundTruth(param_event_id, list_event_id, list_event_region, train_event_id, train_event_region, train_event_source, cnt_gt_positive, cnt_gt_negative);

	if (cnt_gt_positive)
	{
		if (detection_used)
		{
			SampleFromDetection(list_event_id, list_event_region, list_detection, train_event_id, train_event_region, train_event_source);
		}

		if (tracking_used)
		{
			SampleFromTracking(list_event_id, list_event_region, list_detection, list_track, train_event_id, train_event_region, train_event_source);
		}
	}

	return true;
}

bool CModuleTemporalAction::SampleFromGroundTruth(LIST_INT&		param_event_id,
												  LIST_INT&		list_event_id,
												  LIST_RECT&	list_event_region,												 
												  LIST_INT&		train_event_id, 
												  LIST_RECT&	train_event_region,
												  LIST_CHAR&	train_event_source,
												  int& cnt_positive, int& cnt_negative)
{
	cnt_positive = 0;	cnt_negative = 0;

	int cnt_event  = param_event_id.size();
	int cnt_region = list_event_region.size();
	for(int i_region = 0; i_region < cnt_region; i_region++)
	{
		if (list_event_id[i_region] < 0)	// labeled negative samples
		{
			train_event_id.push_back(list_event_id[i_region]);
			train_event_region.push_back(list_event_region[i_region]);
			train_event_source.push_back('L');
			cnt_negative++;	
		}
		else	// labeled positive samples
		{
			for(int i_event = 0; i_event < cnt_event; i_event++)
			{
				if (list_event_id[i_region] == param_event_id[i_event])
				{
					train_event_id.push_back(list_event_id[i_region]);
					train_event_region.push_back(list_event_region[i_region]);
					train_event_source.push_back('L');
				}
			}
			cnt_positive++;	
		}
	}

	return true;
}

bool CModuleTemporalAction::SampleFromDetection(LIST_INT&			list_event_id,
												LIST_RECT&			list_event_region,
												const LIST_BLOB&	list_detection,
												LIST_INT&			train_event_id, 
												LIST_RECT&			train_event_region,
												LIST_CHAR&			train_event_source)
{
	bool bFarEnough, bCloseEnough, bDuplicate = false;
	int cnt_region = list_event_region.size();

	//test head detection results
	int cnt_detection = list_detection.size();
	for(int i_detection = 0; i_detection < cnt_detection; i_detection++)
	{
		// prior-knowledge evaluation
		if (!list_detection[i_detection].active) continue;

		const Blob& detect_blob = list_detection[i_detection];
		RECT headRect;
		Blob2Head(detect_blob,headRect);

		bFarEnough = true;
		for(int i_region = 0; i_region < cnt_region; i_region++)
		{
			if (list_event_id[i_region] > 0)
			{
				int dx = ((list_event_region[i_region].left+list_event_region[i_region].right)/2 - (headRect.left+headRect.right)/2);
				int dy = ((list_event_region[i_region].top+list_event_region[i_region].bottom)/2 - (headRect.top+headRect.bottom)/2);
				double dist = sqrt((double)(dx*dx+dy*dy));
				if (dist<5*max(list_event_region[i_region].right-list_event_region[i_region].left,list_event_region[i_region].bottom-list_event_region[i_region].top))
				{
					bFarEnough = false;
					break;
				}
			}
		}

		if (bFarEnough)
		{
			train_event_id.push_back(-1);
			train_event_region.push_back(headRect);
			train_event_source.push_back('D');
		}

		bCloseEnough = false;
		int closest_event_id = -1;
		float largest_overlap_ratio = 0;
		for(int i_region = 0; i_region < cnt_region; i_region++)
		{
			if (list_event_id[i_region] > 0)
			{
				float area_ratioA, area_ratioB;
				if (IsIntersectRect(list_event_region[i_region],area_ratioA, headRect, area_ratioB))
					if (area_ratioA > 0.7 && area_ratioB > 0.7 && area_ratioA + area_ratioB > 1.49)
					{
						if (area_ratioA + area_ratioB > largest_overlap_ratio)
						{
							largest_overlap_ratio = area_ratioA + area_ratioB;
							closest_event_id = list_event_id[i_region];
						}
						bCloseEnough = true;							 
					}
			}
		}

		if (bCloseEnough && closest_event_id > 0)
		{
			train_event_id.push_back(closest_event_id);
			train_event_region.push_back(headRect);	
			train_event_source.push_back('D');
		}
	}

	return true;
}

bool CModuleTemporalAction::SampleFromTracking(LIST_INT&			list_event_id,
											   LIST_RECT&			list_event_region,
											   const LIST_BLOB&		list_detection,
											   const LIST_TRACK_2D&	list_track,
											   LIST_INT&			train_event_id, 
											   LIST_RECT&			train_event_region,
											   LIST_CHAR&			train_event_source)
{
	bool bFarEnough, bDuplicate, bCloseEnough;
	int cnt_region = list_event_region.size();

	int cnt_track = list_track.size();
	for(int i_track = 0; i_track < cnt_track; i_track++)
	{
		// prior-knowledge evaluation
		if (!list_track[i_track]->IsActive()) continue;

		bFarEnough = true;

		TRACK_2D *pTrack = list_track[i_track];
		if (pTrack->IsCurrent())
		{
			RECT headRect;
			Traj2Head(pTrack->GetTail(),headRect);

			for(int i_region = 0; i_region < cnt_region; i_region++)
			{
				if (list_event_id[i_region] > 0)
				{
					int dx = ((list_event_region[i_region].left+list_event_region[i_region].right)/2 - (headRect.left+headRect.right)/2);
					int dy = ((list_event_region[i_region].top+list_event_region[i_region].bottom)/2 - (headRect.top+headRect.bottom)/2);
					double dist = sqrt((double)(dx*dx+dy*dy));
					if (dist<5*max(list_event_region[i_region].right-list_event_region[i_region].left, list_event_region[i_region].bottom-list_event_region[i_region].top))
					{
						bFarEnough = false;
						break;
					}

				}
			}

			bDuplicate = false;
			if (bFarEnough)
			{
				int cnt_detection = list_detection.size();
				for(int i_detection = 0; i_detection < cnt_detection; i_detection++)
				{
					if (!list_detection[i_detection].active) continue;

					const Blob& detect_blob = list_detection[i_detection];

					RECT dHeadRect;
					Blob2Head(detect_blob,dHeadRect);

					float area_ratioA, area_ratioB;
					if (IsIntersectRect(headRect,area_ratioA, dHeadRect,area_ratioB))
						if (area_ratioA + area_ratioB > 1.49)
						{
							bDuplicate = true;
							break;
						}
				}
			}

			if (bFarEnough && !bDuplicate)
			{
				//extract features as negative samples?
				train_event_id.push_back(-1);
				train_event_region.push_back(headRect);
				train_event_source.push_back('T');
			}

			bCloseEnough = false;
			int closest_event_id = -1;
			float largest_overlap_ratio = 0;
			for(int i_region = 0; i_region < cnt_region; i_region++)
			{
				if (list_event_id[i_region] > 0)
				{
					float area_ratioA, area_ratioB;
					if (IsIntersectRect(list_event_region[i_region],area_ratioA, headRect, area_ratioB))
						if (area_ratioA > 0.7 && area_ratioB > 0.7 && area_ratioA+area_ratioB > 1.49)
						{
							if (area_ratioA + area_ratioB > largest_overlap_ratio)
							{
								largest_overlap_ratio = area_ratioA + area_ratioB;
								closest_event_id = list_event_id[i_region];
							}
							bCloseEnough = true;							 
						}
				}
			}

			bDuplicate = false;
			if (bCloseEnough)
			{
				int cnt_detection = list_detection.size();
				for(int i_detection = 0; i_detection < cnt_detection; i_detection++)
				{
					if (!list_detection[i_detection].active) continue;
					const Blob& detect_blob = list_detection[i_detection];

					RECT dHeadRect;
					Blob2Head(detect_blob,dHeadRect);

					float area_ratioA, area_ratioB;
					if (IsIntersectRect(headRect,area_ratioA, dHeadRect,area_ratioB))
						if (area_ratioA + area_ratioB > 1.49)
						{
							bDuplicate = true;
							break;
						}
				}
			}

			if (bCloseEnough && closest_event_id > 0 && !bDuplicate)
			{
				//extract features as positive samples?
				train_event_id.push_back(closest_event_id);
				train_event_region.push_back(headRect);	
				train_event_source.push_back('T');
			}

		} // if (pTrack->IsCurrent())
	} //for(int i_track = 0; i_track < cnt_track; i_track++)

	return true;
}

bool CModuleTemporalAction::Testing(LIST_INT&				param_event_id,    
									LIST_INT&				list_event_id,
									LIST_FLOAT&				list_event_prob,   
									LIST2_FLOAT&			list_event_all_prob,
									LIST_RECT&				list_event_region,  
									const LIST_BLOB&		list_detection, 
									const LIST_TRACK_2D&	list_track,
									const LIST_RECT&		fg_blob)
{
	// give each candidate region likelihood for each event analyzed
	int cnt_event = param_event_id.size();
	int cnt_region = list_event_region.size();

	for (int i_region = 0; i_region < cnt_region; i_region++)
	{
		// get roi expended
		RECT roi_exp;
		if ( !ROIExpand(list_event_region[i_region], roi_exp) )
		{
			list_event_id[i_region] = -100;
			list_event_prob[i_region] = -1;
			continue;
		}

		// get block index
		LOCAL_PARTITION_MULTISCALE multiscale_block_index;
		SetBlockIndexParam_MultiScale(multiscale_block_index);
		if ( !GetBlockIndex_MultiScale(roi_exp, multiscale_block_index) )
			continue;

		// extract test feature per region
		FEATURE_VECTOR feature_vector;			feature_vector.clear();
		ExtractActionFeature_MultiScale(roi_exp, multiscale_block_index, feature_vector);

		// classify roi region
		int classify_label  = -1;		float classify_dist = -1;		float classify_conf = -1;
		LIST_CLASSIFY_ATTRIB list_classify_info;
		ClassifyROI(feature_vector, classify_label, classify_dist, classify_conf, list_classify_info);

		// output recognization results
		int cnt_class = list_classify_info.size();
		if (cnt_class == cnt_event)
		{
			list_event_id[i_region] = classify_label;
			list_event_prob[i_region] = classify_conf;

			for (int i_event = 0; i_event < cnt_event; i_event++)
			{
				for (int i_class = 0; i_class < cnt_class; i_class++)
				{
					if (param_event_id[i_event] == list_classify_info[i_class].event_label)
					{
						list_event_all_prob[i_region].push_back(list_classify_info[i_class].classify_conf);
						break;
					}
				}
			}
		}
		else
		{
			AfxMessageBox("event number wrong in Testing");
			return false;
		}
	} // i_region

	return true;
}

bool CModuleTemporalAction::ClassifyROI(vector<float> &fea_vector, int &classify_label, float &classify_dist, float &classify_conf, LIST_CLASSIFY_ATTRIB &list_classify_info)
{
	if (fea_vector.size() <= 0)
	{
		AfxMessageBox("Parameter error in \"ClassifyROI\"");
		return false;
	}

	// initialize
	classify_label = -1;	classify_dist  = -1;	classify_conf  = -1;	list_classify_info.clear();

	float max_conf = (float)SMALL_NUMBER, total_conf = 0;
	int label;
	float dist, conf;
	CLASSIFY_ATTRIB cur_classify;

	// celltoear
	ClassifierDecision(fea_vector, m_celltoear_classifier, dist, conf);		total_conf += conf;
	cur_classify.event_label = name_to_id("CellToEar");
	cur_classify.classify_dist = dist;
	cur_classify.classify_conf = conf;
	list_classify_info.push_back(cur_classify);
	if (conf > max_conf)
	{
		max_conf = conf;
		classify_dist = dist;	classify_conf = conf;
		classify_label = name_to_id("CellToEar");	//1; // celltoear
	}

	// objectput
	ClassifierDecision(fea_vector, m_objectput_classifier, dist, conf);		total_conf += conf;
	cur_classify.event_label = name_to_id("ObjectPut");
	cur_classify.classify_dist = dist;
	cur_classify.classify_conf = conf;
	list_classify_info.push_back(cur_classify);
	if (conf > max_conf)
	{
		max_conf = conf;
		classify_dist = dist;	classify_conf = conf;
		classify_label = name_to_id("ObjectPut");	//2; // objectput
	}

	// pointing
	ClassifierDecision(fea_vector, m_pointing_classifier, dist, conf);		total_conf += conf;
	cur_classify.event_label = name_to_id("Pointing");
	cur_classify.classify_dist = dist;
	cur_classify.classify_conf = conf;
	list_classify_info.push_back(cur_classify);
	if (conf > max_conf)
	{
		max_conf = conf;
		classify_dist = dist;	classify_conf = conf;
		classify_label = name_to_id("Pointing");	//3; // pointing
	}

	// normalize confidence or not?
	bool bnormalize = false;
	if (bnormalize)
	{
		classify_conf = classify_conf / total_conf;
	}

	return true;
}

bool CModuleTemporalAction::ClassifierDecision(vector<float> &fea_vector, TACTION_CLASSIFIER& classifier, float &distance, float &confidence)
{
	ClassifierDistance(fea_vector, classifier, distance);
	if (1 == classifier.model_type)
	{
		ClassifierSigmoidConfidence(distance, classifier.sigmoid_a, classifier.sigmoid_c, confidence);
	}
	else
	{
		confidence = distance;
	}

	return true;
}

bool CModuleTemporalAction::ClassifierDistance(vector<float> &fea_vector, TACTION_CLASSIFIER& classifier, float &distance)
{
	int cnt_fea_dim = fea_vector.size();
	if (cnt_fea_dim != classifier.w_dim)
	{
		AfxMessageBox("vector dim error in \"ClassifierDistance\"");
		distance = -1;
		return false;
	}

	float* fea = new float[cnt_fea_dim];
	for (int i_fea = 0; i_fea < cnt_fea_dim; i_fea++)
	{
		fea[i_fea] = fea_vector[i_fea];
	}

	xblas::SharedValVector<float> vec_wnc(classifier.classifier_w, classifier.w_dim);
	xblas::SharedValVector<float> vec_fea(fea, cnt_fea_dim);
	float dot_mul = xblas::dot(vec_wnc, vec_fea);

	distance = dot_mul + classifier.classifier_b;

	delete [] fea;
	return true;
}

bool CModuleTemporalAction::ClassifierSigmoidConfidence(float distance, float sigmoid_a, float sigmoid_c, float &confidence)
{
	float interim = 1 + exp(sigmoid_a * distance + sigmoid_c);
	confidence = 1 / interim;
	return true;
}

bool CModuleTemporalAction::ROIExpand(RECT roi, RECT& roi_expand)
{
	roi_expand.left = roi_expand.right = roi_expand.top = roi_expand.bottom = 0;

	int frm_wid = CVideoSource::GetImageWidth();
	int frm_hei = CVideoSource::GetImageHeight();

	if ( roi.left < 0 || roi.top < 0 || roi.right > frm_wid-1 || roi.bottom > frm_hei-1 ||
		roi.left > roi.right || roi.top > roi.bottom ||
		roi.right - roi.left < 1 || roi.bottom - roi.top < 1 )
	{
//		AfxMessageBox("original roi error");
		return false;
	}

	int roi_wid = roi.right  - roi.left;
	int roi_hei = roi.bottom - roi.top;
	roi_expand.left   = cvRound(roi.left   - 1.5 * roi_wid);
	roi_expand.right  = cvRound(roi.right  + 1.5 * roi_wid);
	roi_expand.top	  = cvRound(roi.top    - 0.5 * roi_hei);
	roi_expand.bottom = cvRound(roi.bottom + 4.0 * roi_hei);

	if (roi_expand.left < 0 )
		roi_expand.left = 0;
	if (roi_expand.right > frm_wid - 1)
		roi_expand.right = frm_wid - 1;
	if (roi_expand.top < 0)
		roi_expand.top = 0;
	if (roi_expand.bottom > frm_hei - 1)
		roi_expand.bottom = frm_hei;
	if (roi_expand.left > roi_expand.right || roi_expand.top > roi_expand.bottom)
	{
//		AfxMessageBox("roi expand error");
		return false;
	}

	return true;
}

bool CModuleTemporalAction::GeneratePerturbationSample(RECT seed_sample_region, int seed_sample_id, bool positive_perturbation, bool negative_perturbation, LIST_RECT& list_generate_sample, LIST_INT& list_generate_id)
{
	list_generate_sample.clear();
	list_generate_id.clear();

	if (seed_sample_id > 0)	// positive sample
	{
		GenerateSample(list_generate_sample, list_generate_id, seed_sample_region, seed_sample_id, positive_perturbation);	// false: NO perturbation, true: perturbation
	}
	else // negative sample
	{
		GenerateSample(list_generate_sample, list_generate_id, seed_sample_region, seed_sample_id, negative_perturbation);	// false: NO perturbation, true: perturbation
	}

	return true;
}

bool CModuleTemporalAction::OutputFeature_MultiScale_Codebook(vector<FEATURE_MATRIX*>& list_feature, int scale_num)
{
	int i_cat, i_fea, i_cop;
	float* feature_memory = NULL;

	// check input parameter
	int cnt_fea_cat = list_feature.size();
	if (cnt_fea_cat <= 0)
	{
		AfxMessageBox("Parameter error in \"OutputFeature_MultiScale_Codebook\"");
		return false;
	}

	for (i_cat = 0; i_cat < cnt_fea_cat; i_cat++)
	{
		if ( (*list_feature[i_cat]).size() <= 0 )
		{
			AfxMessageBox("Parameter error in \"OutputFeature_MultiScale_Codebook\"");
			return false;
		}
	}

	int cnt_fea_num = (*list_feature[0]).size();
	for (i_cat = 1; i_cat < cnt_fea_cat; i_cat++)
	{
		if (cnt_fea_num != (*list_feature[i_cat]).size())
		{
			AfxMessageBox("Parameter error in \"OutputFeature_MultiScale_Codebook\"");
			return false;
		}
	}

	if (0 != cnt_fea_num % scale_num)
	{
		AfxMessageBox("Parameter error in \"OutputFeature_MultiScale_Codebook\"");
		return false;
	}

	for (i_cat = 0; i_cat < cnt_fea_cat; i_cat++)
	{
		for (i_fea = 1; i_fea < cnt_fea_num; i_fea++)
		{
			if ( (*list_feature[i_cat])[i_fea].size() != (*list_feature[i_cat])[i_fea-1].size() )
			{
				AfxMessageBox("Parameter error in \"OutputFeature_MultiScale_Codebook\"");
				return false;
			}
		}
	}

	// re-arrange feature matrix
	int cnt_fea_cop = 0;
	for (i_cat = 0; i_cat < cnt_fea_cat; i_cat++)
	{
		cnt_fea_cop += (*list_feature[i_cat])[0].size();
	}

	feature_memory = new float[cnt_fea_num * cnt_fea_cop];
	int idx_fea = 0;
	for (i_fea = 0; i_fea < cnt_fea_num; i_fea++)
	{
		for (i_cat = 0; i_cat < cnt_fea_cat; i_cat++)
		{
			int cur_cnt_fea_cop = (*list_feature[i_cat])[0].size();
			for (i_cop = 0; i_cop < cur_cnt_fea_cop; i_cop++)
			{
				feature_memory[idx_fea] = (*list_feature[i_cat])[i_fea][i_cop];
				idx_fea++;
			}
		}
	}

	// output 
	FILE *fp_fea, *fp_inf;

	fopen_s(&fp_fea, m_codebook_fea_fn.c_str(), "ab");
	fwrite(feature_memory, sizeof(float), cnt_fea_num*cnt_fea_cop, fp_fea);
	fclose(fp_fea);

	fopen_s(&fp_inf, m_codebook_inf_fn.c_str(), "ab");
	fprintf(fp_inf, "%f\t%f\t%f\t%f\t%f\n", (float)m_frm_no, (float)m_output_feature_codebook, (float)(cnt_fea_num/scale_num), (float)cnt_fea_cop, (float)scale_num);
	fclose(fp_inf);

	// done
	if (NULL != feature_memory)		delete [] feature_memory;
	m_output_feature_codebook += cnt_fea_num/scale_num;
	return true;
}

bool CModuleTemporalAction::OutputFeature_Codebook(vector<FEATURE_MATRIX*>& list_feature)
{
	int i_cat, i_fea, i_cop;
	float* feature_memory = NULL;

	// check input parameter
	int cnt_fea_cat = list_feature.size();
	if (cnt_fea_cat <= 0)
	{
		AfxMessageBox("Parameter error in \"OutputFeature_Codebook\"");
		return false;
	}

	for (i_cat = 0; i_cat < cnt_fea_cat; i_cat++)
	{
		if ( (*list_feature[i_cat]).size() <= 0 )
		{
			AfxMessageBox("Parameter error in \"OutputFeature_Codebook\"");
			return false;
		}
	}

	int cnt_fea_num = (*list_feature[0]).size();
	for (i_cat = 1; i_cat < cnt_fea_cat; i_cat++)
	{
		if (cnt_fea_num != (*list_feature[i_cat]).size())
		{
			AfxMessageBox("Parameter error in \"OutputFeature_Codebook\"");
			return false;
		}
	}

	for (i_cat = 0; i_cat < cnt_fea_cat; i_cat++)
	{
		for (i_fea = 1; i_fea < cnt_fea_num; i_fea++)
		{
			if ( (*list_feature[i_cat])[i_fea].size() != (*list_feature[i_cat])[i_fea-1].size() )
			{
				AfxMessageBox("Parameter error in \"OutputFeature_Codebook\"");
				return false;
			}
		}
	}

	// re-arrange feature matrix
	int cnt_fea_cop = 0;
	for (i_cat = 0; i_cat < cnt_fea_cat; i_cat++)
	{
		cnt_fea_cop += (*list_feature[i_cat])[0].size();
	}

	feature_memory = new float[cnt_fea_num * cnt_fea_cop];
	int idx_fea = 0;
	for (i_fea = 0; i_fea < cnt_fea_num; i_fea++)
	{
		for (i_cat = 0; i_cat < cnt_fea_cat; i_cat++)
		{
			int cur_cnt_fea_cop = (*list_feature[i_cat])[0].size();
			for (i_cop = 0; i_cop < cur_cnt_fea_cop; i_cop++)
			{
				feature_memory[idx_fea] = (*list_feature[i_cat])[i_fea][i_cop];
				idx_fea++;
			}
		}
	}

	// output 
	FILE *fp_fea, *fp_inf;

	fopen_s(&fp_fea, m_codebook_fea_fn.c_str(), "ab");
	fwrite(feature_memory, sizeof(float), cnt_fea_num*cnt_fea_cop, fp_fea);
	fclose(fp_fea);

	fopen_s(&fp_inf, m_codebook_inf_fn.c_str(), "ab");
	fprintf(fp_inf, "%f\t%f\t%f\t%f\n", (float)m_frm_no, (float)m_output_feature_codebook, (float)cnt_fea_num, (float)cnt_fea_cop);
	fclose(fp_inf);

	// done
	if (NULL != feature_memory)		delete [] feature_memory;
	m_output_feature_codebook += cnt_fea_num;
	return true;
}

bool CModuleTemporalAction::OutputFeature_Train(FEATURE_MATRIX& feature_matrix, LIST_INT& feature_label)
{
	FILE *fp_fea, *fp_inf, *fp_label;
	float *feature_memory = NULL;

	int fea_num = feature_matrix.size();
	if (fea_num <= 0 || fea_num != (int)feature_label.size())
	{
//		AfxMessageBox("Parameter error in \"OutputFeature_Train\"");
		return false;
	}
	int fea_dim = feature_matrix[0].size();

	// output feature data
	feature_memory = new float[fea_num * fea_dim];		int idx = 0;
	for (int i_fea = 0; i_fea < fea_num; i_fea++)
	{
		for (int i_dim = 0; i_dim < fea_dim; i_dim++)
		{
			feature_memory[idx] = feature_matrix[i_fea][i_dim];
			idx++;
		}
	}
	fopen_s(&fp_fea, m_train_fea_fn.c_str(), "ab");
	fwrite(feature_memory, sizeof(float), fea_num*fea_dim, fp_fea);
	fclose(fp_fea);

	// output feature label
	fopen_s(&fp_label, m_train_lab_fn.c_str(), "ab");
	for (int i_fea = 0; i_fea < fea_num; i_fea++)
	{
		int label_id = feature_label[i_fea];
		if (label_id < 0)
		{
			label_id = 0;
		}
		else
		{
			bool b_label = false;
			for (int i_label = 0; i_label < taction_map_size; i_label++)
			{
				if (label_id == taction_label_map[i_label].old_id)
				{
					label_id = taction_label_map[i_label].new_id;
					break;
				}
			}
		}
		fprintf(fp_label, "%d\n", label_id);
	}
	fclose(fp_label);

	// output feature info
	m_output_train_feature += fea_num;
	fopen_s(&fp_inf, m_train_inf_fn.c_str(), "wb");
	fprintf(fp_inf, "%.1f %d\n", m_output_train_feature, fea_dim);
	fclose(fp_inf);

	// done
	if (NULL != feature_memory)		delete [] feature_memory;
	return true;
}

bool CModuleTemporalAction::CombineMultiFeature(vector<FEATURE_MATRIX*>& list_feature, FEATURE_MATRIX& combined_feature)
{
	int i_cat, i_fea, i_cop;

	// check input parameter
	int cnt_fea_cat = list_feature.size();
	if (cnt_fea_cat <= 0)
	{
//		AfxMessageBox("Parameter error in \"OutputFeature_Codebook\"");
		return false;
	}

	int cnt_fea_num = (*list_feature[0]).size();
	for (i_cat = 1; i_cat < cnt_fea_cat; i_cat++)
	{
		if (cnt_fea_num != (*list_feature[i_cat]).size())
		{
			AfxMessageBox("Parameter error in \"CombineMultiFeature\"");
			return false;
		}
	}

	for (i_cat = 0; i_cat < cnt_fea_cat; i_cat++)
	{
		for (i_fea = 1; i_fea < cnt_fea_num; i_fea++)
		{
			if ( (*list_feature[i_cat])[i_fea].size() != (*list_feature[i_cat])[i_fea-1].size() )
			{
				AfxMessageBox("Parameter error in \"CombineMultiFeature\"");
				return false;
			}
		}
	}

	// re-arrange feature matrix
	int cnt_fea_cop = 0;
	for (i_cat = 0; i_cat < cnt_fea_cat; i_cat++)
	{
		cnt_fea_cop += (*list_feature[i_cat])[0].size();
	}

	combined_feature.clear();	combined_feature.resize(cnt_fea_num);
	for (i_fea = 0; i_fea < cnt_fea_num; i_fea++)
	{
		for (i_cat = 0; i_cat < cnt_fea_cat; i_cat++)
		{
			int cur_cnt_fea_cop = (*list_feature[i_cat])[0].size();
			for (i_cop = 0; i_cop < cur_cnt_fea_cop; i_cop++)
			{
				combined_feature[i_fea].push_back((*list_feature[i_cat])[i_fea][i_cop]);
			}
		}
	}

	return true;
}
