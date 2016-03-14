#include "stdafx.h"

#include <stdio.h>
#include <cv.h>
#define ENABLE_TIMING 0//1
#include "timing.h"

void calcflow(float* dxdx,float* dydy,float* dxdy,float* dxdt,float* dydt,float* gx,float* gy,int width,int height,int widthstepf,float regularization)
{
	int x,y;
	for (y=0;y<height;y++) {
#pragma ivdep
		for (x=0;x<width;x++) {
			float xx=dxdx[y*widthstepf+x]+regularization;
			float yy=dydy[y*widthstepf+x]+regularization;
			float xy=dxdy[y*widthstepf+x];
			float xt=dxdt[y*widthstepf+x];
			float yt=dydt[y*widthstepf+x];
			float det=1/(xx*yy-xy*xy);

			gx[y*widthstepf+x]=det*(xy*yt-yy*xt);
			gy[y*widthstepf+x]=det*(xy*xt-xx*yt);
		}
	}
}

void fastflow(IplImage* img_prev,IplImage* img_curr,IplImage* imgx,IplImage* imgy,int radius=5,float regularization=10)
{
	IplImage* image=img_prev;
	int num_channels=img_prev->nChannels;

	unsigned char* prev=(unsigned char*)img_prev->imageData;
	unsigned char* curr=(unsigned char*)img_curr->imageData;
	float* gx=(float*)imgx->imageData;
	float* gy=(float*)imgy->imageData;

	int widthstep=img_prev->widthStep;
	int width=img_prev->width;
	int height=img_prev->height;
	int widthstepf=imgx->widthStep/sizeof(float);
	int x,y;
	IplImage* img_dxdx=cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_32F, 1);
	IplImage* img_dydy=cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_32F, 1);
	IplImage* img_dxdy=cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_32F, 1);
	IplImage* img_dxdt=cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_32F, 1);
	IplImage* img_dydt=cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_32F, 1);

	IplImage* img_dxdx_smooth=cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_32F, 1);
	IplImage* img_dxdy_smooth=cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_32F, 1);
	IplImage* img_dydy_smooth=cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_32F, 1);
	IplImage* img_dxdt_smooth=cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_32F, 1);
	IplImage* img_dydt_smooth=cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_32F, 1);

	float* dxdx=(float*)img_dxdx->imageData;
	float* dxdy=(float*)img_dxdy->imageData;
	float* dydy=(float*)img_dydy->imageData;
	float* dxdt=(float*)img_dxdt->imageData;
	float* dydt=(float*)img_dydt->imageData;
	memset(dxdx,0,sizeof(float)*widthstepf*height);
	memset(dxdy,0,sizeof(float)*widthstepf*height);
	memset(dydy,0,sizeof(float)*widthstepf*height);
	memset(dxdt,0,sizeof(float)*widthstepf*height);
	memset(dydt,0,sizeof(float)*widthstepf*height);

	if (num_channels==1) {
		for (y=1;y<height-1;y++) {
#pragma ivdep
			for (x=1;x<width-1;x++) {
				int dt=curr[y*widthstep+x]-prev[y*widthstep+x];
				int dx=prev[y*widthstep+x+1]-prev[y*widthstep+x];
				int dy=prev[(y+1)*widthstep+x]-prev[(y)*widthstep+x];
				dxdt[y*widthstepf+x]=(float)dt*(float)dx;
				dydt[y*widthstepf+x]=(float)dt*(float)dy;
				dxdx[y*widthstepf+x]=(float)dx*(float)dx;
				dydy[y*widthstepf+x]=(float)dy*(float)dy;
				dxdy[y*widthstepf+x]=(float)dx*(float)dy;
			}
		}
	}
	else if (num_channels==3) {
		int c;
		IplImage* prev_imgs[3];
		IplImage* curr_imgs[3];
		for (c=0;c<3;c++) {
			prev_imgs[c]=cvCreateImage(cvSize(width,height),IPL_DEPTH_32F,1);
			curr_imgs[c]=cvCreateImage(cvSize(width,height),IPL_DEPTH_32F,1);
		}
		float* curr_c0=(float*)curr_imgs[0]->imageData;
		float* curr_c1=(float*)curr_imgs[1]->imageData;
		float* curr_c2=(float*)curr_imgs[2]->imageData;
		float* prev_c0=(float*)prev_imgs[0]->imageData;
		float* prev_c1=(float*)prev_imgs[1]->imageData;
		float* prev_c2=(float*)prev_imgs[2]->imageData;
		BEGIN_TIMING(separate);
		for (y=0;y<height;y++) {
			for (x=0;x<width;x++) {
				prev_c0[y*widthstepf+x]=prev[y*widthstep+3*x];
				prev_c1[y*widthstepf+x]=prev[y*widthstep+3*x+1];
				prev_c2[y*widthstepf+x]=prev[y*widthstep+3*x+2];
				curr_c0[y*widthstepf+x]=curr[y*widthstep+3*x];
				curr_c1[y*widthstepf+x]=curr[y*widthstep+3*x+1];
				curr_c2[y*widthstepf+x]=curr[y*widthstep+3*x+2];
			}
		}
		END_TIMING(separate);

		BEGIN_TIMING(gradient);
		for (c=0;c<num_channels;c++) {
			float* pr=(float*)prev_imgs[c]->imageData;
			float* cr=(float*)curr_imgs[c]->imageData;
			for (y=1;y<height-1;y++) {
#pragma ivdep
				for (x=1;x<width-1;x++) {
					float dt=cr[y*widthstepf+x]-pr[y*widthstepf+x];
					float dx=pr[y*widthstepf+x+1]-pr[y*widthstepf+x];
					float dy=pr[(y+1)*widthstepf+x]-pr[y*widthstepf+x];
					dxdt[y*widthstepf+x]+=dt*dx;
					dydt[y*widthstepf+x]+=dt*dy;
					dxdx[y*widthstepf+x]+=dx*dx;
					dydy[y*widthstepf+x]+=dy*dy;
					dxdy[y*widthstepf+x]+=dx*dy;
				}
			}
		}
		END_TIMING(gradient);

		for (int c=0;c<num_channels;c++) {
			cvReleaseImage(&prev_imgs[c]);
			cvReleaseImage(&curr_imgs[c]);
		}
	}
	else {
		BEGIN_TIMING(gradient);
		for (y=1;y<height-1;y++) {
			for (x=1;x<width-1;x++) {
				for (int c=0;c<num_channels;c++) {
					int dt=curr[y*widthstep+num_channels*x+c]-prev[y*widthstep+num_channels*x+c];
					int dx=prev[y*widthstep+num_channels*(x+1)+c]-prev[y*widthstep+num_channels*x+c];
					int dy=prev[(y+1)*widthstep+num_channels*x+c]-prev[y*widthstep+num_channels*x+c];
					dxdt[y*widthstepf+x]+=(float)dt*(float)dx;
					dydt[y*widthstepf+x]+=(float)dt*(float)dy;
					dxdx[y*widthstepf+x]+=(float)dx*(float)dx;
					dydy[y*widthstepf+x]+=(float)dy*(float)dy;
					dxdy[y*widthstepf+x]+=(float)dx*(float)dy;
				}
			}
		}
		END_TIMING(gradient);
	}
	BEGIN_TIMING(smooth);
	cvSmooth(img_dxdx,img_dxdx_smooth,CV_BLUR,radius,radius);
	cvSmooth(img_dxdy,img_dxdy_smooth,CV_BLUR,radius,radius);
	cvSmooth(img_dydy,img_dydy_smooth,CV_BLUR,radius,radius);
	cvSmooth(img_dxdt,img_dxdt_smooth,CV_BLUR,radius,radius);
	cvSmooth(img_dydt,img_dydt_smooth,CV_BLUR,radius,radius);
	END_TIMING(smooth);

	dxdx=(float*)img_dxdx_smooth->imageData;
	dxdy=(float*)img_dxdy_smooth->imageData;
	dydy=(float*)img_dydy_smooth->imageData;
	dxdt=(float*)img_dxdt_smooth->imageData;
	dydt=(float*)img_dydt_smooth->imageData;

	BEGIN_TIMING(calcflow);
	calcflow(dxdx,dydy,dxdy,dxdt,dydt,gx,gy,width,height,widthstepf,regularization);
	END_TIMING(calcflow);


	cvReleaseImage(&img_dxdx);
	cvReleaseImage(&img_dxdy);
	cvReleaseImage(&img_dydy);
	cvReleaseImage(&img_dxdt);
	cvReleaseImage(&img_dydt);
	cvReleaseImage(&img_dxdx_smooth);
	cvReleaseImage(&img_dxdy_smooth);
	cvReleaseImage(&img_dydy_smooth);
	cvReleaseImage(&img_dxdt_smooth);
	cvReleaseImage(&img_dydt_smooth);
}

