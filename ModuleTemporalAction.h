#pragma once

#include "..\trecvid\trecvidanalyst.h"
#include "../../include/cv/cv.h"
//#include "utilities.h"
#include <list>
#include <blasxx.h>

#include "AlgTemporalAction.h"

//////////////////////////////////////////////////////////////////////////
#define FRAME_BUFFER_LEN	22

#define TRAIN_TACTION_CODEBOOK	1
#define TRAIN_TACTION_FEATURE	2

#define TACTION_BLOCK_PATCH_SIZE	8
#define TACTION_BLOCK_PATCH_OVERLAP	0

//////////////////////////////////////////////////////////////////////////


class CModuleTemporalAction : public CTrecvidAnalyst
{
public:
	CModuleTemporalAction(string& par_fn, bool bEvaluation);
	virtual ~CModuleTemporalAction(void);

	virtual void Reset(void);
	virtual bool Analyze(bool		          bEvaluation,            // true: evaluation; false: training
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
						 const LIST_RECT&     fg_blob);				  // foreground blobs if available

protected:
	bool m_bEvaluation;	// type of process (train or test)
	int	 m_TrainType;	// training type -- codebook training || feature extraction
	TACTION_PARAM taction_param;	// parameters loaded from file

	CAlgTemporalAction m_taction;	// instance of temporal action algorithms class

	// common attributes for video frame
	int m_frm_no;	// current frame index no.
	int m_frm_hei;	// frame height
	int m_frm_wid;	// frame width

	// variables for frame buffering
	bool m_bTemporalReady;	// ready flag for buffering
	bool m_bTemporalInit;	// init flag for buffering
	int m_last_frm_no;		// last frame index for buffering
	int m_cur_buffer_len;	// current frame index for buffering
	list<IplImage*> m_frame_buffer;	// frame buffering memory

	// variables for frame gradient buffering
	bool m_bGradientBuffer;		// flag for whether using/computing gradient
	list< vector<IplImage*> > m_gradient_buffer;	// gradient buffering memory

	// variables for frame optical flow buffering
	bool m_bOpticalFlowBuffer;	// flag for whether using/computing optical flow
	list< vector<IplImage*> > m_optflow_buffer;		// optical flow buffering memory

	// variables for file output
	string m_codebook_fea_fn, m_codebook_inf_fn;
	string m_train_fea_fn, m_train_inf_fn, m_train_lab_fn;
	float m_output_feature_codebook;

	// variables for feature extraction (for train & test)
	BOF_CODEBOOK m_codebook;
	float m_output_train_feature;

	// variables for testing/evaluation
	TACTION_CLASSIFIER	m_celltoear_classifier;	// celltoear classifier
	TACTION_CLASSIFIER	m_objectput_classifier;	// objectput classifier
	TACTION_CLASSIFIER	m_pointing_classifier;	// pointing classifier

public:
	bool Training(LIST_INT&				param_event_id,    
		          LIST_INT&             list_event_id,
				  LIST_INT&             list_event_inst_id,    
				  LIST_INT&             list_event_inst_obj_id,
				  LIST_FLOAT&           list_event_prob,    
				  LIST_RECT&            list_event_region,  
				  const LIST_BLOB&      list_detection, 
				  const LIST_TRACK_2D&  list_track,
				  const LIST_RECT&      fg_blob);

	bool Testing(LIST_INT&				param_event_id,    
				 LIST_INT&              list_event_id,
				 LIST_FLOAT&            list_event_prob,   
				 LIST2_FLOAT&           list_event_all_prob,
				 LIST_RECT&             list_event_region,  
				 const LIST_BLOB&       list_detection, 
				 const LIST_TRACK_2D&   list_track,
				 const LIST_RECT&       fg_blob);

	bool PreProcess(int frm_no, const BYTE* img_color);		// preprocessing routine
	bool TemporalBuffer(int frm_no, const BYTE* img_color);	// temporal (frame) buffering
	void InitParam(void);
	void ReleaseBoFCodebook(void);
	void ReleaseFrameBuffer(void);
	void ReleaseClassifier(void);
	void Release(void);

	bool FrameBufferInit(int frm_wid, int frm_hei);
	bool FrameBuffer(int frm_wid, int frm_hei, const BYTE* img_color);
	bool GradientBufferInit(int frm_wid, int frm_hei);
	bool GradientBuffer(int frm_wid, int frm_hei, const BYTE* img_color);
	bool OpticalFlowBufferInit(int frm_wid, int frm_hei);
	bool OpticalFlowBuffer(int frm_wid, int frm_hei, const BYTE* img_color);

	bool SampleFromGroundTruth(LIST_INT&	param_event_id,
							   LIST_INT&	list_event_id,
							   LIST_RECT&	list_event_region,												 
							   LIST_INT&	train_event_id, 
							   LIST_RECT&	train_event_region,
							   LIST_CHAR&	train_event_source,
							   int& cnt_positive, int& cnt_negative);
	bool SampleFromDetection(LIST_INT&			list_event_id,
					   		 LIST_RECT&			list_event_region,
							 const LIST_BLOB&	list_detection,
							 LIST_INT&			train_event_id, 
							 LIST_RECT&			train_event_region,
							 LIST_CHAR&			train_event_source);
	bool SampleFromTracking(LIST_INT&				list_event_id,
						    LIST_RECT&				list_event_region,
							const LIST_BLOB&		list_detection,
							const LIST_TRACK_2D&	list_track,
							LIST_INT&				train_event_id, 
							LIST_RECT&				train_event_region,
							LIST_CHAR&				train_event_source);
	bool GetTrainSample(LIST_INT&				param_event_id,
						LIST_INT&				list_event_id,
						LIST_RECT&				list_event_region,  
						const LIST_BLOB&		list_detection, 
						const LIST_TRACK_2D&	list_track,
						LIST_INT&				train_event_id, 
						LIST_RECT&				train_event_region,
						LIST_CHAR&				train_event_source,
						int& cnt_gt_positive, int& cnt_gt_negative, bool detection_used, bool tracking_used);

	bool ExtractCodebookFeature(const LIST_INT& param_event_id, const LIST_INT& train_event_id, const LIST_RECT& train_event_region);
	bool CodebookFeature_ImageGradient(const RECT roi_region, LOCAL_PARTITION& block_index, FEATURE_MATRIX& sample_feature);
	bool CodebookFeature_OpticalFlow(const RECT roi_region, LOCAL_PARTITION& block_index, FEATURE_MATRIX& sample_feature);

	bool ExtractCodebookFeature_MultiScale(const LIST_INT& param_event_id, const LIST_INT& train_event_id, const LIST_RECT& train_event_region);
	bool CodebookFeature_MultiScale_ImageGradient(const RECT roi_region, LOCAL_PARTITION_MULTISCALE& block_index, FEATURE_MATRIX& sample_feature);
	bool CodebookFeature_MultiScale_OpticalFlow(const RECT roi_region, LOCAL_PARTITION_MULTISCALE& block_index, FEATURE_MATRIX& sample_feature);

	bool OutputFeature_MultiScale_Codebook(vector<FEATURE_MATRIX*>& list_feature, int scale_num);

	bool ROIExpand(RECT roi, RECT& roi_expand);
	bool GeneratePerturbationSample(RECT seed_sample_region, int seed_sample_id, bool positive_perturbation, bool negative_perturbation, LIST_RECT& list_generate_sample, LIST_INT& list_generate_id);

	void SetVideoFn(const string& video_fn);
	void InitOutput(void);
	bool CombineMultiFeature(vector<FEATURE_MATRIX*>& list_feature, FEATURE_MATRIX& combined_feature);
	bool OutputFeature_Codebook(vector<FEATURE_MATRIX*>& list_feature);
	bool InitCodebook(const char* codebook_fn);	

	bool ExtractTrainFeature(const LIST_INT& param_event_id, const LIST_INT& train_event_id, const LIST_RECT& train_event_region);
	bool ExtractActionFeature(const RECT roi_region, LOCAL_PARTITION& block_index, FEATURE_VECTOR& sample_feature);	
	bool CalcFeatureRepresentation(FEATURE_MATRIX& block_feature_matrix, FEATURE_VECTOR& feature);
	bool OutputFeature_Train(FEATURE_MATRIX& feature_matrix, LIST_INT& feature_label);

	bool ExtractTrainFeature_MultiScale(const LIST_INT& param_event_id, const LIST_INT& train_event_id, const LIST_RECT& train_event_region);
	bool ExtractActionFeature_MultiScale(const RECT roi_region, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, FEATURE_VECTOR& sample_feature);
	bool CalcFeatureRepresentation_SPM(const RECT roi_region, LOCAL_PARTITION_MULTISCALE& multiscale_block_index, FEATURE_MATRIX& block_feature_matrix, FEATURE_VECTOR& feature);

	bool SetBlockIndexParam_MultiScale(LOCAL_PARTITION_MULTISCALE& block_index);

	bool InitClassifier(TACTION_PARAM& taction_param);
	bool ClassifyROI(vector<float> &fea_vector, int &classify_label, float &classify_dist, float &classify_conf, LIST_CLASSIFY_ATTRIB &list_classify_info);
	bool ClassifierDecision(vector<float> &fea_vector, TACTION_CLASSIFIER& classifier, float &distance, float &confidence);
	bool ClassifierDistance(vector<float> &fea_vector, TACTION_CLASSIFIER& classifier, float &distance);
	bool ClassifierSigmoidConfidence(float distance, float sigmoid_a, float sigmoid_c, float &confidence);
};
