/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools  |  Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: kiev
 *
 * Created on July 4, 2017, 1:38 PM
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <fftw3.h>

#include "fftw_procedures.h"
#include "filters_procedures.h"

/*
 * 
 */
int main(int argc, char** argv)
{
	int WIDTH, HEIGHT, DIM;
	//fftw_complex * complex_fft = 0, * changed_fft = 0;

	fftw_plan plan = 0;

	//const char * path = "bw/gray.jpeg";
	//const char * path = "grad.png";
	//const char * path = "color/lenna.jpg";
	const char * path = "lenna.png";
	//const char * path = "xadrez.png";
	//const char * path = "tel.jpg";
	//const char * path = "bw/telb.png";
	//const char * path = "collor.jpg";
	//const char * path = "bw/tinyb.jpg";

	IplImage * image_in = 0, * image_out = 0;

	image_in = cvLoadImage(path, 1);

	if (image_in == 0) {
		printf("\nERROR! Image not loaded. Check in main function\n");
		exit(-1);
	}

	WIDTH = image_in->width;
	HEIGHT = image_in->height;
	DIM = image_in->nChannels;

	printf("(%d, %d, %d)\n", WIDTH, HEIGHT, DIM);
	

    cvDFT(src, dst, DFT_COMPLEX_OUTPUT, 0);

	 = laplaceFFTW(, WIDTH, HEIGHT);
	


    cvDFT(src, dst, DFT_REAL_OUTPUT, 0);


	printf("Finishing the program\n");
	
	printf("\tClearing the plans\n");
	// free memory
	fftw_destroy_plan(plan);

	
	printf("\tCleaning the fftw_complex\n");
	fftw_free(complex_fft);
	fftw_free(changed_fft);

	printf("\tShowing the images\n");
	cvShowImage("IN",  image_in);
	cvShowImage("OUT", image_out);
	/*cvSaveImage("In.png", image_in, 0);
	cvSaveImage("Out.png", image_out, 0);*/
	
	cvWaitKey(0);
    /*
	cvShowImage("iplimage_dft(): mag", image_mag);
	cvShowImage("iplimage_dft(): phase", image_phase);
	//cvShowImage("iplimage_dft(): spectrum", image_spectrum);
	cvSaveImage("iplimage_dft_mag.png", image_mag, 0);
	cvSaveImage("iplimage_dft_phase.png", image_phase, 0);
	cvWaitKey(0);
	 * 
	 *

	// Free memory in the end
	fftw_destroy_plan(plan_forward);
	fftw_free(in_complex);
	fftw_free(dft_complex);
	cvReleaseImage(&image_in);
	cvReleaseImage(&image_spectrum);
	 */
	
	
	
	printf("\tCleaning the images\n");
	cvReleaseImage(&image_in);
	cvReleaseImage(&image_out);
	/*cvReleaseImage(&image_mag);
	cvReleaseImage(&image_phase);
	cvReleaseImage(&image_spectrum);*/

	return 0;
}
