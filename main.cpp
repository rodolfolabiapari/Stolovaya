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
double* conv2_(double* d, double* kernel, double* result)
{
	 register double acc;
	 register int data_row_point; 
	 register int data_col_point;
	 register int kernel_row, k2;
	 register int kernel_col, l2;
	 register int data_position_first_col, kernel_position_first_col;
	 char FILTER_DIM = 3;

	 for(data_row_point = 0; data_row_point < HEIGHT; data_row_point++) {
       
		  data_position_first_col = data_row_point * HEIGHT; 
	
		  for(data_col_point = 0; data_col_point < WIDTH; data_col_point++) {   
	   
				acc = 0.0;
	    
				for(kernel_row = FILTER_DIM - 1, k2 = 0; kernel_row >= 0; kernel_row--, k2++) {
	       
					 kernel_position_first_col = kernel_row * FILTER_DIM;  // loop invariants
		
					 for(kernel_col = FILTER_DIM - 1, l2 = 0; kernel_col >= 0; kernel_col--, l2++) {
						  acc += kernel[kernel_position_first_col + kernel_col] 
 * 
				 d[((data_row_point + k2) * IMG_DIM) + (data_col_point + l2)];
					 }
				}
	    
				result[data_position_first_col + data_col_point] = acc;
		  }
	 }

	 return result;
}


void sobel_kernel(char * data, char isHorizontal) 
{
	char data_out [WIDTH * HEIGHT * DIM];
   
	CvMat * k =
   
	cvFilter2D(data, data_out, );
}
 */


/*
 * Print fftw_complex Vector
 */
void print_complex(fftw_complex * c, int WIDTH, int HEIGHT)
{
	int i;

	for (i = 0; i < WIDTH * HEIGHT; i++) {
		printf("%5d(%8.1f, %8.1f)  \n", i, c[i][0], c[i][1]);
	}
}

/*
 * Print IplImage
 */
void print_ipl(IplImage ipl, int WIDTH, int HEIGHT)
{
	int i, j;
	char * d = ipl.imageData;

	for (i = 0; i < HEIGHT + 10; i++) {
		printf("%d:  ", i + 1);
		for (j = 0; j < WIDTH; j++) {
			printf("%3d ", (int) (unsigned char) d[i * 3 * WIDTH + j * 3]);
		}
		printf("\n");
	}

	exit(-1);
}


/*
 * Procedure that copies the value of fftw_complex vector to a mag and phase char vectors
 */
void complex_to_magPhase(fftw_complex * fft_complex_outside, char * mag, char * phase, int WIDTH, int HEIGHT)
{
	int i;

	double mag_b, phase_b;

	for (i = 0; i < WIDTH * HEIGHT; i++) {
		mag_b = mag[i * 3];
		phase_b = ((phase[i * 3] / (double) MAXBGR) * 2 * M_PI) - M_PI;
		fft_complex_outside[i][0] = (mag_b * cos(phase_b));
		fft_complex_outside[i][1] = (mag_b * sin(phase_b));
	}
}

/*
 * Procedure that gets the real and imaginary values of the fftw_complex vector.
 */
void get_real_and_imaginary(fftw_complex * fft_complex_outside, fftw_complex * fft_complex_inside,
		  int position, double * r, double * i, char method, char isIn, int WIDTH, int HEIGHT)
{
	if (isIn) {
		if (method) {
			// Normalize values
			*r = fft_complex_inside[position * 3 + 0][0] / (double) (WIDTH * HEIGHT);
			*i = fft_complex_inside[position * 3 + 0][1] / (double) (WIDTH * HEIGHT);

		} else {
			// Normalize values
			*r = fft_complex_inside[position * 3 + 0][0];
			*i = fft_complex_inside[position * 3 + 0][1];
		}

	} else {
		if (method) {
			// Normalize values
			*r = fft_complex_outside[position * 3 + 0][0] / (double) (WIDTH * HEIGHT);
			*i = fft_complex_outside[position * 3 + 0][1] / (double) (WIDTH * HEIGHT);

		} else {
			// Normalize values
			*r = fft_complex_outside[position * 3 + 0][0];
			*i = fft_complex_outside[position * 3 + 0][1];
		}
	}
}

/*
 * Procedure that copy the real and imaginary value to a position in a char vector.
 */
void phase_to_ipl(int pos, double r, double i, char * data)
{
	double _Complex complex;
	unsigned char buffer_uc = 0;
	double phase_r = 0, phase_g = 0, phase_b = 0;
	char DEBUG = 0;

	// Calcule the phase
	complex = r + i * _Complex_I;

	phase_b = carg(complex) + M_PI;

	// scale and write to output
	data[pos * 3 + 0] = (unsigned char) (int) (MAXBGR * (phase_b / (double) (2 * M_PI)));

	if (DEBUG) printf("  |  Ph %d", (unsigned char) data[pos * 3 + 0]);

	data[pos * 3 + 1] = data[pos * 3 + 0];
	data[pos * 3 + 2] = data[pos * 3 + 0];

	if (DEBUG) printf("\n");
}

/*
 * Swap pixels from a Ipls char vector.
 */
void swap_pixels(char * base_position, int upper_offset, int lower_offset)
{
	char buffer;

	//B
	buffer = *(base_position + upper_offset + 0);
	*(base_position + upper_offset + 0) = *(base_position + lower_offset + 0);
	*(base_position + lower_offset + 0) = buffer;

	//G
	buffer = *(base_position + upper_offset + 1);
	*(base_position + upper_offset + 1) = *(base_position + lower_offset + 1);
	*(base_position + lower_offset + 1) = buffer;

	//R
	buffer = *(base_position + upper_offset + 2);
	*(base_position + upper_offset + 2) = *(base_position + lower_offset + 2);
	*(base_position + lower_offset + 2) = buffer;
}

/*
 * Procedure that inverts the quadrants in diagonal form.
 * 
 * The quadrant 1 goes to 3 and 2 goes to 3.
 */
void swap_quadrants_gray_image(char * pixels, int WIDTH, int HEIGHT)
{
	int full_col, full_row, half_col, half_row;
	int upper_offset, lower_offset, row, col;

	full_col = WIDTH;
	full_row = HEIGHT;

	half_col = floor(full_col * 3 / (double) 2);
	half_row = floor(full_row * 3 / (double) 2);

	// swap quadrants diagonally
	for (row = 0; row < half_row; row += 3) {
		for (col = 0; col < half_col; col += 3) {
			// Position of Second and First Quadrants.
			upper_offset = col + (full_col * row);
			// Position of Third and Fourth Quadrants
			lower_offset = upper_offset + // Current Position
					  half_col + // Jump to next Quadrants
					  (full_col * half_row);

			swap_pixels(pixels, upper_offset, lower_offset);

			swap_pixels(pixels, upper_offset + half_col, lower_offset - half_col);
		}
	}
}


void magnitude_pixel_logarithm(double max, double min, char * data, 
		  double * b, int WIDTH, int HEIGHT, int DIM)
{
	double c;
	int i, j;
	char DEBUG = 0;

	c = 255 / (log(1 + max));

	if (min == 0) {
		if (DEBUG) printf("%f\n", log(1 + max));
		if (DEBUG) printf("%f\n-\n", 255 / log(1 + max));

		j = 0;
		for (i = 0; i < WIDTH * HEIGHT * DIM; i += 3) {

			data[i + 0] = c * log(1 + b[j]);

			if (DEBUG) printf("(%08.3f, %08.3f) ", log(1 + b[j]), c * log(1 + b[j]));

			data[i + 1] = data[i];
			data[i + 2] = data[i];

			if (DEBUG) printf("\n");
			j++;

		}
	} else {

		j = 0;
		for (i = 0; i < WIDTH * HEIGHT * DIM; i += 3) {

			data[i + 0] = c * log(1 + normalize_interval(b[j], max, min));

			if (DEBUG) printf("(%08.3f, %08.3f) ", log(1 + normalize_interval(b[j], max, min)), c * log(1 + normalize_interval(b[j], max, min)));

			data[i + 1] = data[i];
			data[i + 2] = data[i];

			if (DEBUG) printf("\n");
			j++;
		}
	}
}

/*
 * Procedure that generates a spectrum of the complex array calculated by
 * Fourier Transform.
 */
/*
void do_spectrum(fftw_complex * out_complex_b, fftw_complex * out_complex_g,
		  fftw_complex * out_complex_r, IplImage out_mag, IplImage out_phase)
{
	double real_r = 0, imag_r = 0,
			  real_g = 0, imag_g = 0,
			  real_b = 0, imag_b = 0,
			  mag_r[WIDTH * HEIGHT], mag_g[WIDTH * HEIGHT], mag_b[WIDTH * HEIGHT];
	double max_mag_b = -1, max_mag_g = -1, max_mag_r = -1;
	double min_mag_b = -1, min_mag_g = -1, min_mag_r = -1;
	char DEBUG;
	int i;

	DEBUG = 0;
	// Get the values and transform its in magnitude and phase
	for (i = 0; i < WIDTH * HEIGHT; i++) {

		if (DEBUG) printf("i:%5d:%5d", i, WIDTH * HEIGHT);

		get_real_and_complex(&real_b, &imag_b, &real_g, &imag_g, &real_r,
				  &imag_r, out_complex_b[i], out_complex_g[i], out_complex_r[i], 0);

		// Calcule the magnitude
		mag_b[i] = sqrt((real_b * real_b) + (imag_b * imag_b));

		if (DEBUG) printf("  |  FFTW(%012.3f, %012.3f)", real_b, imag_b);
		if (DEBUG) printf("  |  Mg(%012.3f)", mag_b[i]);

		if (!isGray) {
			mag_g[i] = sqrt((real_g * real_g) + (imag_g * imag_g));
			mag_r[i] = sqrt((real_r * real_r) + (imag_r * imag_r));

			if (DEBUG) printf(" (%012.3f, %012.3f) (%012.3f, %012.3f)", real_g, imag_g, real_r, imag_r);
			if (DEBUG) printf(" (%012.3f) (%012.3f)", mag_g[i], mag_r[i]);
		}

		if (i == 0) {
			max_mag_b = mag_b[i];
			max_mag_g = mag_g[i];
			max_mag_r = mag_r[i];

			min_mag_b = mag_b[i];
			min_mag_g = mag_g[i];
			min_mag_r = mag_r[i];

		} else {
			if (mag_b[i] > max_mag_b) max_mag_b = mag_b[i];
			if (mag_g[i] > max_mag_g) max_mag_g = mag_g[i];
			if (mag_r[i] > max_mag_r) max_mag_r = mag_r[i];

			if (mag_b[i] < min_mag_b) min_mag_b = mag_b[i];
			if (mag_g[i] < min_mag_g) min_mag_g = mag_g[i];
			if (mag_r[i] < min_mag_r) min_mag_r = mag_r[i];
		}

		calcule_phase(i, real_b, imag_b, real_g, imag_g, real_r, imag_r, out_phase.imageData);

		fflush(stdout);
	}

	printf("M%f, m%f\n", max_mag_b, min_mag_b);

	magnitude_pixel_logarithm(max_mag_b, max_mag_g, max_mag_r,
			  min_mag_b, min_mag_g, min_mag_r, out_mag.imageData, mag_b, mag_g, mag_r);

	swap_quadrants_gray_image(out_mag.imageData);
	swap_quadrants_gray_image(out_phase.imageData);
}

void complex_to_image(fftw_complex * in_mag_b, fftw_complex * in_mag_g, fftw_complex * in_mag_r, char * data)
{
	double magR, magG, magB;
	int i;
	
	// save real parts to output
	for (i = 0; i < WIDTH * HEIGHT; i++) {
		magB = in_mag_b[i][0];

		// make sure it's capped at MAXBGR
		data[i * 3 + 0] = magR > MAXBGR ? MAXBGR : magR;

		if (!isGray) {
			magG = in_mag_g[i][0];
			magR = in_mag_r[i][0];
			
			data[i * 3 + 1] = magG > MAXBGR ? MAXBGR : magG;
			data[i * 3 + 2] = magB > MAXBGR ? MAXBGR : magB;
			
		} else {
			data[i * 3 + 1] = data[i * 3 + 0];
			data[i * 3 + 2] = data[i * 3 + 0];
		}
	}
}
 */



/*
 * FFTW function for the IplImage.
 */
/*
void ifft(fftw_complex * fft_complex_outside_b, fftw_complex * fft_complex_outside_g,
	fftw_complex * fft_complex_outside_r, fftw_complex * fft_complex_inside_b, fftw_complex * fft_complex_inside_g,
	fftw_complex * fft_complex_inside_r, IplImage * ipl_image_out, 
	fftw_plan * plan_b, fftw_plan * plan_g, fftw_plan * plan_r, 
	int WIDTH, int HEIGHT, int DIM)
{
	printf("Starting the IFFTW\n");

	swap_quadrants_gray_fftw(fft_complex_inside_b, WIDTH, HEIGHT);


	// Assign the values of image (BGR) to the real parts of the array (array[i][0])
	printf("\tCoping the datas of image\n");

   
	if (!plan_b) {
 * plan_b = fftw_plan_dft_2d(WIDTH, HEIGHT, fft_complex_inside_b, fft_complex_outside_b,
			FFTW_BACKWARD, FFTW_PATIENT);

		if (!isGray) {
 * plan_g = fftw_plan_dft_2d(WIDTH, HEIGHT, fft_complex_inside_g, fft_complex_outside_g,
		 FFTW_BACKWARD, FFTW_PATIENT);
 * plan_r = fftw_plan_dft_2d(WIDTH, HEIGHT, fft_complex_inside_r, fft_complex_outside_r,
		 FFTW_BACKWARD, FFTW_PATIENT);
		}
	} else {
      
		// free memory
		fftw_destroy_plan(* plan_b);
		if (!isGray) {
	 fftw_destroy_plan(* plan_g);
	 fftw_destroy_plan(* plan_r);
		}

 * plan_b = fftw_plan_dft_2d(WIDTH, HEIGHT, fft_complex_outside_b, fft_complex_inside_b,
			FFTW_BACKWARD, FFTW_PATIENT);

		if (!isGray) {
 * plan_g = fftw_plan_dft_2d(WIDTH, HEIGHT, fft_complex_outside_g, fft_complex_inside_g,
		 FFTW_BACKWARD, FFTW_PATIENT);
 * plan_r = fftw_plan_dft_2d(WIDTH, HEIGHT, fft_complex_outside_r, fft_complex_inside_r,
		 FFTW_BACKWARD, FFTW_PATIENT);
		}
	}

	printf("\tExecuting the FFTW\n");
	// Execute the forward FFT
	fftw_execute(* plan_b);

	if (!isGray) {
		fftw_execute(* plan_g);
		fftw_execute(* plan_r);
	}

	complex_to_ipl(fft_complex_outside_b, fft_complex_outside_g, fft_complex_outside_r, ipl_image_out->imageData, WIDTH, HEIGHT, DIM);
   
	fftw_cleanup();
}*/

/*
 * 
 */
int main(int argc, char** argv)
{
	int WIDTH, HEIGHT, DIM;
	char isGray = 1;
	fftw_complex * complex_fft = 0, * changed_fft = 0;

	fftw_plan plan = 0;

	//const char * path = "bw/gray.jpeg";
	//const char * path = "grad.png";
	//const char * path = "color/lenna.jpg";
	const char * path = "bw/lenna.png";
	//const char * path = "xadrez.png";
	//const char * path = "tel.jpg";
	//const char * path = "bw/telb.png";
	//const char * path = "collor.jpg";
	//const char * path = "bw/tinyb.jpg";

	isGray = 1;

	IplImage * image_in = 0, * image_out = 0;

	image_in = cvLoadImage(path, 1);

	if (image_in == 0) {
		printf("\nERROR! Image not loaded. Check in main function\n");
		exit(-1);
	}

	WIDTH = image_in->width;
	HEIGHT = image_in->height;
	DIM = image_in->nChannels;

	complex_fft = fft(image_in, WIDTH, HEIGHT, DIM);

	changed_fft = laplace(complex_fft, WIDTH, HEIGHT);
	
	image_out = ifft(changed_fft, WIDTH, HEIGHT, DIM);

	printf("Finishing the program\n");
	
	printf("\tClearing the plans\n");
	// free memory
	fftw_destroy_plan(plan);

	
	printf("\tCleaning the fftw_complex\n");
	fftw_free(complex_fft);

	printf("\tShowing the images\n");
	cvShowImage("IN",  image_in);
	cvShowImage("OUT", image_out);
	cvSaveImage("In.png", image_in, 0);
	cvSaveImage("Out.png", image_out, 0);
	/*
	cvShowImage("iplimage_dft(): mag", image_mag);
	cvShowImage("iplimage_dft(): phase", image_phase);
	//cvShowImage("iplimage_dft(): spectrum", image_spectrum);
	cvSaveImage("iplimage_dft_mag.png", image_mag, 0);
	cvSaveImage("iplimage_dft_phase.png", image_phase, 0);
	cvWaitKey(0);
	 * 
	 */

	/*
	//cvShowImage("iplimage_dft(): original", image_in);
	//
   
	//Find the maximum value among the magnitudes
	max = mag = 0;
	k = 0;
	for (i = 0; i < HEIGHT; i++){
		for (j = 0; j < WIDTH; j++){
			mag = sqrt(pow(dft_complex[k][0],2) + pow(dft_complex[k][1],2));
			if (max < mag)
				max = mag;
			k++;
		}
	}

	// Convert DFT result to output image
	k = 0;
	for (i = 0; i < HEIGHT; i++) {
		for (j = 0; j < WIDTH; j++) {
			mag = sqrt(pow(dft_complex[k][0],2) + pow(dft_complex[k][1],2));
			mag = 255*(mag/max);
			if (DEBUG)  printf("\tMAX: %.0f\tPix: %f\n", max, mag);
			((uchar*)(image_spectrum->imageData + i * image_spectrum->widthStep))[j] = mag;
			k++;
		}
	}

	cvShowImage("iplimage_dft(): original", image_in);
	cvShowImage("iplimage_dft(): result", image_spectrum);
	cvSaveImage("iplimage_dft.png", image_spectrum, 0 );
	cvWaitKey(0);

	// Free memory in the end
	fftw_destroy_plan(plan_forward);
	fftw_free(in_complex);
	fftw_free(dft_complex);
	cvReleaseImage(&image_in);
	cvReleaseImage(&image_spectrum);
	 */
	
	cvWaitKey();
	
	
	printf("\tCleaning the images\n");
	cvReleaseImage(&image_in);
	cvReleaseImage(&image_out);
	/*cvReleaseImage(&image_mag);
	cvReleaseImage(&image_phase);
	cvReleaseImage(&image_spectrum);*/

	return 0;
}