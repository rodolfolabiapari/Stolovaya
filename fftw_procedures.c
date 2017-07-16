/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   fftw_functions.c
 * Author: kiev
 * 
 * Created on July 11, 2017, 2:09 PM
 */

#include "fftw_procedures.h"

void normalize_to_255 (fftw_complex * fftw, char type_data, int WIDTH, int HEIGHT) {
	double smax = 255, smin = 0;
	double max_current, min_current;
	int i;
	
	max_current = min_current = fftw[0][type_data];
	
	for (i = 1; i < WIDTH * HEIGHT; i++) {
		if (max_current < fftw[i][type_data]) max_current = fftw[i][type_data];
		
		if (min_current > fftw[i][type_data]) min_current = fftw[i][type_data];
	}
	
	
	for (i = 0; i < WIDTH * HEIGHT; i++)
		fftw[i][type_data] =  ( fftw[i][type_data] - min_current ) * (smax - smin) / ( max_current - min_current ) + smin;
}

double normalize_interval(double x, double max, double min)
{
	double scale, new_value;

	scale = (x - min) / (max - min);

	new_value = (max * scale);

	return new_value;
}

void logarithm_it(fftw_complex * fftw, char type_data, int WIDTH, int HEIGHT)
{
	double c;
	int i;
	char DEBUG = 0;
	double max, min;
	
	min = max = fftw[0][type_data];
	
	for (i = 1; i < WIDTH * HEIGHT; i++) {
		if (max < fftw[i][type_data]) max = fftw[i][type_data];
		
		if (min > fftw[i][type_data]) min = fftw[i][type_data];
	}

	c = 255 / (log(1 + max));

	for (i = 0; i < WIDTH * HEIGHT; i++) {

		fftw[i][type_data] = c * log(1 + normalize_interval(fftw[i][type_data], max, min));

		if (DEBUG) printf("(%08.3f, %08.3f)\n", log(1 + normalize_interval(fftw[i][type_data], max, min)), 
				  c * log(1 + normalize_interval(fftw[i][type_data], max, min)));
	}
}


void swap_fftw_pixels(fftw_complex * base_position, int upper_offset, int lower_offset)
{
	fftw_complex buffer;

	//B
	buffer[0] = base_position[upper_offset + 0][0];
	base_position[upper_offset + 0][0] = base_position[lower_offset + 0][0];
	base_position[lower_offset + 0][0] = buffer[0];

	buffer[1] = base_position[upper_offset + 0][1];
	base_position[upper_offset + 0][1] = base_position[lower_offset + 0][1];
	base_position[lower_offset + 0][1] = buffer[1];

	/*
	//G
	buffer[0] = base_position[upper_offset + 1][0];
	base_position[upper_offset + 1][0] = base_position[lower_offset + 1][0];
	base_position[lower_offset + 1][0] = buffer[0];

	buffer[1] = base_position[upper_offset + 1][1];
	base_position[upper_offset + 1][1] = base_position[lower_offset + 1][1];
	base_position[lower_offset + 1][1] = buffer[1];


	//R
	buffer[0] = base_position[upper_offset + 2][0];
	base_position[upper_offset + 2][0] = base_position[lower_offset + 2][0];
	base_position[lower_offset + 2][0] = buffer[0];

	buffer[1] = base_position[upper_offset + 2][1];
	base_position[upper_offset + 2][1] = base_position[lower_offset + 2][1];
	base_position[lower_offset + 2][1] = buffer[1];
	 * */
}

/*
 * Procedure that inverts the quadrants in diagonal form.
 * 
 * The quadrant 1 goes to 3 and 2 goes to 3.
 */
void swap_quadrants_gray_fftw(fftw_complex * c, int WIDTH, int HEIGHT)
{
	int full_col, full_row, half_col, half_row;
	int upper_offset, lower_offset, row, col;

	full_col = WIDTH;
	full_row = HEIGHT;

	half_col = full_col / 2;
	half_row = full_row / 2;

	// swap quadrants diagonally
	for (row = 0; row < half_row; row++) {
		
		for (col = 0; col < half_col; col++) {
			
			// Position of Second and First Quadrants.
			upper_offset = col + (full_col * row);
			
			// Position of Third and Fourth Quadrants
			lower_offset = upper_offset + // Current Position
					  half_col + // Jump to next Quadrants
					  (full_col * half_row); // bottom 

			swap_fftw_pixels(c, upper_offset, lower_offset);

			swap_fftw_pixels(c, upper_offset + half_col, lower_offset - half_col);
		}
	}
}


/*
 * Copy a fftw_complex to another one.
 * 
 * This method do not allocate nothing in memory. Only copy.
 */
void copy_complex(fftw_complex * from, fftw_complex ** to, int WIDTH, int HEIGHT) {
	int i = 0;
	
	for (i = 0; i < WIDTH * HEIGHT; i++) {
		(*to)[i][0] = from[i][0];
		(*to)[i][1] = from[i][1];
	}
}

/*
 * Procedure that allocates space to future process.
 * 
 * It verifies if the vectors has been allocated.
 * 
 * If allocated, do nothing. Else, allocate it.
 */
void create_fftw_complex(fftw_complex ** f, int WIDTH, int HEIGHT)
{
	if (! *f) {
		*f = (fftw_complex*) fftw_malloc(sizeof (fftw_complex) * WIDTH * HEIGHT);
	}
}

/*
 * Copy the values of image to the vector of complex values.
 * 
 * This procedure copy the datas of char's array to the another which name is
 * fftw_complex, from fftw3 library.
 * 
 * The data is colocated in position 0, local of the real numbers.
 */
void ipl_to_complex(fftw_complex * complex_fft, char * data, int WIDTH, int HEIGHT)
{
	int i, j, DEBUG;

	DEBUG = 0;

	j = 0;
	for (i = 0; i < WIDTH * HEIGHT; i++) {

		complex_fft[i][0] = (unsigned char) data[j];

		if (DEBUG) printf("%6d:%d\t%f\n", i, WIDTH * HEIGHT, complex_fft[i][0]);

		j += 3;
	}
}



/*
 * Convert the fftw_complex to a iplimage struct copying the data.
 * 
 * This procedure does the copy without allocate memory.
 */
void complex_to_ipl(fftw_complex * fft_complex_outside, char * data, int WIDTH, int HEIGHT, int DIM)
{
	int i, j, DEBUG;

	DEBUG = 0;

	normalize_to_255(fft_complex_outside, 0, WIDTH, HEIGHT);
	normalize_to_255(fft_complex_outside, 1, WIDTH, HEIGHT);
	
	if (1) {
		
		j = 0;
		for (i = 0; i < WIDTH * HEIGHT * DIM; i += 3) {

			data[i + 0] = (unsigned char) (fft_complex_outside[j][0]);

			if (DEBUG) printf("%6d:%d\t(R%6.3f:I%6.3f, %d)\n", i, WIDTH * HEIGHT, fft_complex_outside[j][0], 
					  fft_complex_outside[j][1], (unsigned char) data[i]);

			data[i + 1] = data[i];
			data[i + 2] = data[i];

			j++;
		}
	} else { 
		j = 0;
		for (i = 0; i < WIDTH * HEIGHT * DIM; i += 3) {

			data[i + 0] = (unsigned char) (fft_complex_outside[j][0] / (WIDTH * HEIGHT));

			if (DEBUG) printf("%6d:%d\t(R%6.3f:I%6.3f, %d)\n", i, WIDTH * HEIGHT, fft_complex_outside[j][0] / (double) (WIDTH * HEIGHT), 
					  fft_complex_outside[j][1] / (double) (WIDTH * HEIGHT), (unsigned char) data[i]);

			data[i + 1] = data[i];
			data[i + 2] = data[i];

			j++;
		}
	}
	
}

/*
 * FFTW function for the IplImage.
 */
fftw_complex * fft(IplImage * ipl_image_in, int WIDTH, int HEIGHT, int DIM)
{
	fftw_plan plan = 0;
	fftw_complex * complex_in = 0;
	fftw_complex * complex_out = 0;
	printf("Starting the FFTW\n");


	create_fftw_complex(& complex_in,  WIDTH, HEIGHT);
	create_fftw_complex(& complex_out, WIDTH, HEIGHT);
	
	// create plans
	printf("\tCreating the Plans for FORWARD\n");

	plan = fftw_plan_dft_2d(WIDTH, HEIGHT, complex_in, complex_out,
			  FFTW_FORWARD, FFTW_PATIENT);


	// Assign the values of image (BGR) to the real parts of the array (array[i][0])
	printf("\tCoping the datas of image\n");

	ipl_to_complex(complex_in, (*ipl_image_in).imageData, WIDTH, HEIGHT);


	printf("\tExecuting the FFTW\n");

	// Execute the forward FFT
	fftw_execute(plan);

	fftw_destroy_plan(plan);

	fftw_free(complex_in);
	
	//fftw_cleanup();

	swap_quadrants_gray_fftw(complex_out, WIDTH, HEIGHT);
	
	return complex_out;
}

/*
 * iFFTW function.
 */
IplImage * ifft(fftw_complex * complex_in, int WIDTH, int HEIGHT, int DIM)
{
	printf("Starting the IFFTW\n");

	fftw_plan plan = 0;
	fftw_complex * ifft_complex_in = 0, * ifft_complex_out = 0;
	
	IplImage * ipl_out = cvCreateImage(cvSize(WIDTH, HEIGHT), 8, 3);

	create_fftw_complex(& ifft_complex_in,  WIDTH, HEIGHT);
	create_fftw_complex(& ifft_complex_out, WIDTH, HEIGHT);
	

	// Assign the values of image (BGR) to the real parts of the array (array[i][0])
	printf("\tCoping the datas of image\n");

	plan = fftw_plan_dft_2d(WIDTH, HEIGHT, ifft_complex_in, ifft_complex_out,
			  FFTW_BACKWARD, FFTW_PATIENT);
	
	copy_complex(complex_in, &ifft_complex_in, WIDTH, HEIGHT);

	swap_quadrants_gray_fftw(ifft_complex_in, WIDTH, HEIGHT);
	
	printf("\tExecuting the FFTW\n");
	// Execute the forward FFT
	fftw_execute(plan);

	complex_to_ipl(ifft_complex_out, ipl_out->imageData, WIDTH, HEIGHT, DIM);

	fftw_cleanup();
	
	fftw_destroy_plan(plan);
	fftw_free(ifft_complex_in);
			  
	fftw_free(ifft_complex_out);

	return ipl_out;
}
