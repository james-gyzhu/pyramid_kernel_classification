#include "stdafx.h"

#include <stdio.h>
#define ENABLE_TIMING 0//1
#include "timing.h"

#include "simplegradient.h"

// simple gradient calculation using kernel [-1 0 +1] for x-cord and [-1 0 +1]' for y-cord
bool simplegradient(IplImage* gray_img, IplImage* &gradient_x, IplImage* &gradient_y)
{
	float ratio = 1.0;

	if ( NULL == gray_img || NULL == gradient_x || NULL == gradient_y )
		return false;

	int img_wid = cvGetSize(gray_img).width;		int img_hei = cvGetSize(gray_img).height;

	if ( cvGetSize(gradient_x).width != img_wid || cvGetSize(gradient_x).height != img_hei || cvGetSize(gradient_y).width != img_wid || cvGetSize(gradient_y).height != img_hei )
		return false;

	cvSetZero(gradient_x);		cvSetZero(gradient_y);

	// x-gradient
	for (int hei = 0; hei < img_hei; hei++)
	{
		for (int wid = 1; wid < img_wid - 1; wid++)
		{
			((float*)(gradient_x->imageData + gradient_x->widthStep*hei))[wid] = 
				ratio * ( (float)(((uchar*)(gray_img->imageData + gray_img->widthStep*hei))[wid + 1]) - (float)(((uchar*)(gray_img->imageData + gray_img->widthStep*hei))[wid - 1]) );
		}
	}

	// y-gradient
	for (int hei = 1; hei < img_hei - 1; hei++)
	{
		for (int wid = 0; wid < img_wid; wid++)
		{
			((float*)(gradient_y->imageData + gradient_y->widthStep*hei))[wid] = 
				ratio * ( (float)(((uchar*)(gray_img->imageData + gray_img->widthStep*(hei + 1)))[wid]) - (float)(((uchar*)(gray_img->imageData + gray_img->widthStep*(hei - 1)))[wid]) );
		}
	}

	return true;
}