/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   fftw_functions.h
 * Author: kiev
 *
 * Created on July 11, 2017, 2:09 PM
 */

#ifndef FFTW_FUNCTIONS_H
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <fftw3.h>
#define FFTW_FUNCTIONS_H

#endif /* FFTW_FUNCTIONS_H */


const unsigned char MAXBGR = 255;


void swap_fftw_pixels(fftw_complex * base_position, int upper_offset, int lower_offset);

void swap_quadrants_gray_fftw(fftw_complex * c, int WIDTH, int HEIGHT);

void copy_complex(fftw_complex * from, fftw_complex ** to, int WIDTH, int HEIGHT) ;

void create_fftw_complex(fftw_complex ** f, int WIDTH, int HEIGHT);

void ipl_to_complex(fftw_complex * complex_fft, char * data, int WIDTH, int HEIGHT);

void complex_to_ipl(fftw_complex fft_complex_outside[], char * data, int WIDTH, int HEIGHT, int DIM);

fftw_complex * fft(IplImage * ipl_image_in, int WIDTH, int HEIGHT, int DIM);

IplImage * ifft(fftw_complex * complex_in, int WIDTH, int HEIGHT, int DIM);