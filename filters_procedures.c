/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   filters_procedures.cpp
 * Author: kiev
 * 
 * Created on July 11, 2017, 2:51 PM
 */

#include "filters_procedures.h"
#include "fftw_procedures.h"

/*
 * 
*
%meshFourier
u = 0:lin-1;
v = 0:col-1;

u = u - floor(lin/2);
v = v - floor(col/2);
[U, V] = meshgrid(u, v);

H = -(U.^2 + V.^2);

% Dominio de frequencia
novaFourier = H .* fourier;
 */
fftw_complex * laplace(fftw_complex * fft, int WIDTH, int HEIGHT) {
	fftw_complex * fftw_out = 0;
	int row = 0, col = 0, row_mesh = 0, col_mesh = 0, current = 0;
	double power_result = 0;
	
	printf("Doing the laplace detector of edges\n");
	
	printf("\tCreating the news fftw_complex for processing\n");
	create_fftw_complex(& fftw_out, WIDTH, HEIGHT);
	
	row_mesh = HEIGHT / 2;
	col_mesh = WIDTH  / 2;
	
	printf("\tProcessing\n");
	
	for (row = 0; row < HEIGHT; row++) {
		
		for (col = 0; col < WIDTH; col++) {
			
			current = row * WIDTH + col;
			
			power_result = - ( 
					  (col - col_mesh) * (col - col_mesh)  +  
					  (row - row_mesh) * (row - row_mesh) 
					  );
			
			fftw_out[current + 0] = fft[current + 0] * power_result; 
			fftw_out[current + 1] = fft[current + 1] * power_result;
		}
	}
	
	printf("\tDone\n");
	
	return fftw_out;
}
