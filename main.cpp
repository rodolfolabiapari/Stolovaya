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


int WIDTH, HEIGHT, DIM;
const unsigned char MAXBGR = 255;


unsigned char at_gray_pixel(char * d, int row, int col) {
   unsigned char r;
   
   r = d[row * 3 * WIDTH + col * 3];
   
   return r;
}


void copy_to_complex(fftw_complex * in_complex_b, fftw_complex * in_complex_g, fftw_complex * in_complex_r, char * data, char BW) {
   int i, j, DEBUG;
   
   DEBUG = 1;
   
   j = 0;
   for (i = 0; i < WIDTH * HEIGHT * 2; i++) {
      
      in_complex_b[i][0] = (unsigned char) data[j];

      if (DEBUG)  printf("%d:%f\n",i, in_complex_b[i][0]);
      
      if (!BW) {
	 j++;
	 in_complex_g[i][0] = (unsigned char) data[j];  if (DEBUG)  printf("%f\n", in_complex_g[i][0]);  j++;
	 in_complex_r[i][0] = (unsigned char) data[j];  if (DEBUG)  printf("%f\n\n", in_complex_r[i][0]); j++;
	 
      } else
	 j += 3;
   }
}

void get_real_and_complex(double * r_b, double * i_b, double * r_g, double * i_g, double * r_r, double * i_r, fftw_complex out_complex_b, fftw_complex out_complex_g, fftw_complex out_complex_r, char method, char BW) {
   
      if (method) {
	 // Normalize values
	 *r_b = out_complex_b[0] / (double) (WIDTH * HEIGHT);
	 *i_b = out_complex_b[1] / (double) (WIDTH * HEIGHT);
	 if (!BW) {
	    
	    *r_g = out_complex_g[0] / (double) (WIDTH * HEIGHT);
	    *i_g = out_complex_g[1] / (double) (WIDTH * HEIGHT);

	    *r_r = out_complex_r[0] / (double) (WIDTH * HEIGHT);
	    *i_r = out_complex_r[1] / (double) (WIDTH * HEIGHT);
	 }
	  
      } else {
	 // Normalize values
	 *r_b = out_complex_b[0];
	 *i_b = out_complex_b[1];
	 
	 if (!BW) {
	    *r_g = out_complex_g[0];
	    *i_g = out_complex_g[1];

	    *r_r = out_complex_r[0];
	    *i_r = out_complex_r[1];
	 }
      }
}

void calcule_phase(int pos, double r_b, double i_b, double r_g, double i_g, double r_r, double i_r, char * data, char BW ) {
   
   double _Complex complex_b, complex_g, complex_r;
   unsigned char buffer_uc = 0;
   double phase_r = 0, phase_g = 0, phase_b = 0;
   char DEBUG = 1;
   
   // Calcule the phase
   complex_b = r_b + i_b * _Complex_I;

   if (!BW) {
      complex_g = r_g + i_g * _Complex_I;
      complex_r = r_r + i_r * _Complex_I;
   }

   phase_b = carg(complex_b) + M_PI;

   if (!BW) {
      phase_g = carg(complex_g) + M_PI;
      phase_r = carg(complex_r) + M_PI;
   }

   // scale and write to output
   data[pos * 3 + 0] = (unsigned char) (int) (MAXBGR * (phase_b / (double) (2 * M_PI)));

   if (DEBUG)  printf("  |  Ph %d", (unsigned char) data[pos * 3 + 0]);

   if (!BW) {
      data[pos * 3 + 1] = (unsigned char) (int) (MAXBGR * (phase_g / (double) (2 * M_PI)));
      data[pos * 3 + 2] = (unsigned char) (int) (MAXBGR * (phase_r / (double) (2 * M_PI)));

      if (DEBUG)  printf(" %d %d", (unsigned char) data[pos * 3 + 1], (unsigned char) data[pos * 3 + 2]);

   } else {
      buffer_uc = (unsigned char) (int) (MAXBGR * (phase_b / (double) (2 * M_PI)));

      data[pos * 3 + 1] = buffer_uc;
      data[pos * 3 + 2] = buffer_uc;
   }

   if (DEBUG)  printf("\n");
}

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

double normalize_interval(double x, double max, double min) {
   double scale, new_value;
   
   scale = (x - min) / (max - min);

   new_value = (max * scale);
   
   return new_value;
}

void magnitude_pixel_logarithm(double max_b, double max_g, double max_r, 
	double min_b, double min_g, double min_r, 
	char * data, double * b, double * g, double * r, char BW) 
{
   double c_b, c_g, c_r;
   int i, j;
   char DEBUG = 1;
   
   
   c_b = 255 / (log(1 + max_b));

   if (!BW) {
      c_g = 255 / (log(1 + max_g));
      c_r = 255 / (log(1 + max_r));
   }
   
   if (min_b == 0) {
      if (DEBUG)  printf("%f\n", log(1 + max_b));
      if (DEBUG)  printf("%f\n-\n", 255 / log(1 + max_b));

      j = 0;
      for (i = 0; i < WIDTH * HEIGHT * DIM; i += 3) {

	 if (!BW) {
	    data[i + 0] = c_b * log(1 + b[j]);
	    data[i + 1] = c_g * log(1 + g[j]);
	    data[i + 2] = c_r * log(1 + r[j]);

	    if (DEBUG)  printf("(%08.3f, %08.3f) ",  log(1 + b[j]), c_b * log(1 + b[j]));
	    if (DEBUG)  printf("(%08.3f, %08.3f) ",  log(1 + b[j]), c_g * log(1 + g[j]));
	    if (DEBUG)  printf("(%08.3f, %08.3f)", log(1 + b[j]), c_r * log(1 + r[j]));

	 } else {
	    data[i + 0] = c_b * log(1 + b[j]);

	    if (DEBUG)  printf("(%08.3f, %08.3f) ",  log(1 + b[j]), c_b * log(1 + b[j]));

	    data[i + 1] = data[i];
	    data[i + 2] = data[i];
	 }

	 if (DEBUG)  printf("\n");
	 j++;
	 
      }
   } else  {
      
      //if (DEBUG)  printf("%f\n", log(1 + max_b));
      //if (DEBUG)  printf("%f\n-\n", 255 / log(1 + max_b));
      
      j = 0;
      for (i = 0; i < WIDTH * HEIGHT * DIM; i += 3) {

	 if (!BW) {
	    data[i + 0] = c_b * log(1 + normalize_interval(b[j], max_b, min_b));
	    data[i + 1] = c_g * log(1 + normalize_interval(g[j], max_g, min_g));
	    data[i + 2] = c_r * log(1 + normalize_interval(r[j], max_r, min_r));

	    if (DEBUG)  printf("(%08.3f, %08.3f) ", log(1 + normalize_interval(b[j], max_b, min_b)), c_b * log(1 + normalize_interval(b[j], max_b, min_b)));
	    if (DEBUG)  printf("(%08.3f, %08.3f) ", log(1 + normalize_interval(g[j], max_g, min_g)), c_g * log(1 + normalize_interval(g[j], max_g, min_g)));
	    if (DEBUG)  printf("(%08.3f, %08.3f) ", log(1 + normalize_interval(r[j], max_r, min_r)), c_r * log(1 + normalize_interval(r[j], max_r, min_r)));

	 } else {
	    data[i + 0] = c_b * log(1 + normalize_interval(b[j], max_b, min_b));

	    if (DEBUG)  printf("(%08.3f, %08.3f) ",  log(1 + normalize_interval(b[j], max_b, min_b)), c_b * log(1 + normalize_interval(b[j], max_b, min_b)));

	    data[i + 1] = data[i];
	    data[i + 2] = data[i];
	 }

	 if (DEBUG)  printf("\n");
	 j++;
	 
      }
   }
}

void swap_quadrants_gray_image(char * pixels)
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

	 /*buffer1 = *(pixels + upper_offset + 0);
	  *(pixels + upper_offset + 0) = *(pixels + lower_offset + 0);
	  *(pixels + lower_offset + 0) = buffer1;
	  */
	 /*buffer1 = *(pixels + upper_offset + 1);
	  *(pixels + upper_offset + 1) = *(pixels + lower_offset + 1);
	  *(pixels + lower_offset + 1) = buffer1;

	 buffer1 = *(pixels + upper_offset + 2);
	  *(pixels + upper_offset + 2) = *(pixels + lower_offset + 2);
	  *(pixels + lower_offset + 2) = buffer1;
	  */


	 /*buffer2 = *(pixels + upper_offset + half_col);
	  *(pixels + upper_offset + half_col) = *(pixels + lower_offset - half_col);
	  *(pixels + lower_offset - half_col) = buffer2;
	  */
	 /*buffer2 = *(pixels + upper + half_col);
	  *(pixels + upper + half_col) = *(pixels + lower - half_col);
	  *(pixels + lower - half_col) = buffer2;*/
      }
   }
}

void fft(IplImage * image_in, IplImage * outMag,
	IplImage * outPhase, IplImage * outSpectrum, char BW)
{

   int i = 0, j = 0, k = 0;
   fftw_plan plan_r, plan_g, plan_b;
   fftw_complex * in_complex_r = 0, * in_complex_g = 0, * in_complex_b = 0,
	   * out_complex_r = 0, * out_complex_g = 0, * out_complex_b = 0;
   double real_r = 0, imag_r = 0,
	   real_g = 0, imag_g = 0,
	   real_b = 0, imag_b = 0,
	   mag_r[WIDTH * HEIGHT], mag_g[WIDTH * HEIGHT], mag_b[WIDTH * HEIGHT];
   double max_mag_b = -1, max_mag_g = -1, max_mag_r = -1;
   double min_mag_b = -1, min_mag_g = -1, min_mag_r = -1;
   unsigned char DEBUG = 1;

   // Allocate the input complex arrays 
   printf("Creating the Complex Vectors\n");
   
   in_complex_b = (fftw_complex*) fftw_malloc(sizeof (fftw_complex) * WIDTH * HEIGHT);
   if (!BW) {
      in_complex_g = (fftw_complex*) fftw_malloc(sizeof (fftw_complex) * WIDTH * HEIGHT);
      in_complex_r = (fftw_complex*) fftw_malloc(sizeof (fftw_complex) * WIDTH * HEIGHT);
   }

   // Allocate the output complex arrays
   out_complex_b = (fftw_complex*) fftw_malloc(sizeof (fftw_complex) * WIDTH * HEIGHT);
   if (!BW) {
      out_complex_g = (fftw_complex*) fftw_malloc(sizeof (fftw_complex) * WIDTH * HEIGHT);
      out_complex_r = (fftw_complex*) fftw_malloc(sizeof (fftw_complex) * WIDTH * HEIGHT);
   }

   // create plans
   printf("Creating the Plans\n");
   
   plan_b = fftw_plan_dft_2d(WIDTH, HEIGHT, in_complex_b, out_complex_b, FFTW_FORWARD, FFTW_PATIENT);
   if (!BW) {
      plan_g = fftw_plan_dft_2d(WIDTH, HEIGHT, in_complex_g, out_complex_g, FFTW_FORWARD, FFTW_PATIENT);
      plan_r = fftw_plan_dft_2d(WIDTH, HEIGHT, in_complex_r, out_complex_r, FFTW_FORWARD, FFTW_PATIENT);
   }

   // Assign the values of image (BGR) to the real parts of the array (array[i][0])
   j = 0;
   printf("Coping the datas of image\n");
   
   copy_to_complex(in_complex_b, in_complex_g, in_complex_r, (*image_in).imageData, BW);

   printf("Executing the FFTW\n");
   // Execute the forward FFT
   fftw_execute(plan_b);
   
   if (!BW) {
      fftw_execute(plan_g);
      fftw_execute(plan_r);
   }

   DEBUG = 1;
   // Get the values and transform its in magnitude and phase
   for (i = 0; i < WIDTH * HEIGHT; i++) {
      
      if (DEBUG)  printf("i:%5d:%5d", i, WIDTH * HEIGHT);
      
      get_real_and_complex(&real_b, &imag_b, &real_g, &imag_g, &real_r, &imag_r, out_complex_b[i], out_complex_g[i], out_complex_r[i], 0, BW);
      
      // Calcule the magnitude
      mag_b[i] = sqrt((real_b * real_b) + (imag_b * imag_b));
      
      if (DEBUG)  printf("  |  FFTW(%012.3f, %012.3f)", real_b, imag_b);
      if (DEBUG)  printf("  |  Mg(%012.3f)", mag_b[i]);

      if (!BW) {
	 mag_g[i] = sqrt((real_g * real_g) + (imag_g * imag_g));
	 mag_r[i] = sqrt((real_r * real_r) + (imag_r * imag_r));

	 if (DEBUG)  printf(" (%012.3f, %012.3f) (%012.3f, %012.3f)", real_g, imag_g, real_r, imag_r);
	 if (DEBUG)  printf(" (%012.3f) (%012.3f)", mag_g[i], mag_r[i]);
      }
      
      if (i == 0) {
	 max_mag_b = mag_b[i]; max_mag_g = mag_g[i]; max_mag_r = mag_r[i];
	 
	 min_mag_b = mag_b[i]; min_mag_g = mag_g[i];  min_mag_r = mag_r[i];
	 
      } else {
	 if (mag_b[i] > max_mag_b) max_mag_b = mag_b[i];
	 if (mag_g[i] > max_mag_g) max_mag_g = mag_g[i];
	 if (mag_r[i] > max_mag_r) max_mag_r = mag_r[i];
	 
	 if (mag_b[i] < min_mag_b) min_mag_b = mag_b[i];
	 if (mag_g[i] < min_mag_g) min_mag_g = mag_g[i];
	 if (mag_r[i] < min_mag_r) min_mag_r = mag_r[i];
      }
      
      fflush(stdout);
      
      calcule_phase(i, real_b, imag_b, real_g, imag_g, real_r, imag_r, (*outPhase).imageData, BW);

      fflush(stdout);
   }
   
   printf("M%f, m%f\n", max_mag_b, min_mag_b);
   
   magnitude_pixel_logarithm(max_mag_b, max_mag_g, max_mag_r, 
	   min_mag_b, min_mag_g, min_mag_r, (*outMag).imageData, mag_b, mag_g, mag_r, BW);
   
   swap_quadrants_gray_image((*outMag).imageData);
   swap_quadrants_gray_image((*outPhase).imageData);

   // free memory
   fftw_destroy_plan(plan_b);
   if (!BW) {
      fftw_destroy_plan(plan_g);
      fftw_destroy_plan(plan_r);
   }

   fftw_free(in_complex_b);
   fftw_free(out_complex_b);
   if (!BW) {
      fftw_free(in_complex_g);
      fftw_free(out_complex_g);
      fftw_free(in_complex_r);
      fftw_free(out_complex_r);
   }
}


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


void print(char * d) {
   int i, j;
   
   for (i = 0; i < HEIGHT + 10; i++) {
      printf("%d:  ", i + 1);
      for (j = 0; j < WIDTH; j++){
	 printf("%3d ",(int) (unsigned char) d[i * 3 * WIDTH + j * 3]);
      }
      printf("\n");
   }
   
   exit(-1);
}

/*
 * 
 */
int main(int argc, char** argv)
{
   double hScale = 1.0;
   double vScale = 1.0;
   int lineWidth = 1;
   double max = 0, mag = 0;
   int i, j, k;
   //const char * path = "bw/gray.jpeg";
   //const char * path = "grad.png";
   const char * path = "color/lenna.jpg";
   //const char * path = "xadrez.png";
   //const char * path = "tel.jpg";
   //const char * path = "telb.png";
   //const char * path = "collor.jpg";
   //const char * path = "bw/tinyb.jpg";
   char BW = 1;

   IplImage * image_in, * image_mag, * image_phase, * image_spectrum, * image_out;

   image_in = cvLoadImage(path, 1);
   
   if (image_in == 0) {
      exit(-1);
   }

   WIDTH  = image_in->width;
   HEIGHT = image_in->height;
   DIM    = image_in->nChannels;
   
   print(image_in->imageData);

   image_mag = cvCreateImage(cvSize(WIDTH, HEIGHT), 8, 3);
   image_phase = cvCreateImage(cvSize(WIDTH, HEIGHT), 8, 3);
   image_spectrum = cvCreateImage(cvSize(WIDTH, HEIGHT), 8, 3);

   fft(image_in, image_mag, image_phase, image_spectrum, BW);

   //cvShowImage("iplimage_dft(): original", image_in);
   cvShowImage("iplimage_dft(): mag", image_mag);
   cvShowImage("iplimage_dft(): phase", image_phase);
   //cvShowImage("iplimage_dft(): spectrum", image_spectrum);
   cvSaveImage("iplimage_dft_mag.png", image_mag, 0);
   cvSaveImage("iplimage_dft_phase.png", image_phase, 0);
   cvWaitKey(0);

   /*
   //cvShowImage("iplimage_dft(): original", image_in);
   //cvWaitKey();
   
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
   
   cvReleaseImage(&image_in);
   //cvReleaseImage(&image_out);
   cvReleaseImage(&image_mag);
   cvReleaseImage(&image_phase);
   cvReleaseImage(&image_spectrum);
   
   return 0;
}