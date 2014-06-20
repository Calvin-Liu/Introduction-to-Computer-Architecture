#include "header.h"
#include <math.h>
#include <stdlib.h>

float **
array_create(int w, int h)
{
	float **ptr = (float **) malloc(h * sizeof(float *));
	int i;
	for (i = 0; i < h; i++)
		ptr[i] = (float *) malloc(w * sizeof(float));
	return ptr;
}

void
array_free(float **ptr, int h)
{
	if (ptr) 
	{
		int i;
		for (i = 0; i < h; i++)
			free(ptr[i]);
		free(ptr);
	}
}

void
convert_grayscale(Image *img, float **gray)
{
	int x, y;
	int w = img->w;
	int h = img->h;

	for (y = 0; y < h; y++)
	for (x = 0; x < w; x++)
	{
		float r = img->r[y*w+x];
		float g = img->g[y*w+x];
		float b = img->b[y*w+x];
		gray[y][x] = (r + g + b) / 3.f;
	}
}

float *
create_gaussian_kernel(float sigma, int *size)
{
	int ksize = (int)(sigma * 2.f * 4.f + 1) | 1;
	int i, halfk = ksize / 2;
	float *kernel;
	float scale, sum;

	if (ksize <= 1)
	{
		if (size) *size = 0;
		return NULL;
	}

	kernel = (float *) malloc(ksize * sizeof(float));
	if (!kernel)
	{
		if (size) *size = -1;
		return NULL;
	}

	scale = -0.5f / (sigma*sigma);
	sum = 0.f;

	for (i = 0; i < ksize; i++)
	{
		float x = (float)(i - halfk);
		float t = expf(scale * x * x);
		kernel[i] = t;
		sum += t;
	}

	scale = 1.f / sum;
	for (i = 0; i < ksize; i++)
		kernel[i] *= scale;

	if (size) *size = ksize;
	return kernel;
}

void
filter_rows(float **src, float **dst, int w, int h, float *kernel, int ksize)
{
	int y, x;

	for (y = 0; y < h; y++)
	for (x = 0; x < w; x++)
	{
		int k = 0;
		int c = x - ksize/2;
		float accum = 0.f;

		for ( ; c < 0; c++, k++)
			accum += kernel[k] * src[y][0];

		for ( ; k < ksize && c < w; c++, k++)
			accum += kernel[k] * src[y][c];

		for ( ; k < ksize; k++)
			accum += kernel[k] * src[y][w-1];

		dst[y][x] = accum;
	}
}

void
filter_cols(float **src, float **dst, int w, int h, float *kernel, int ksize)
{
	int y, x;

	for (x = 0; x < w; x++)
	for (y = 0; y < h; y++)
	{
		int k = 0;
		int r = y - ksize/2;
		float accum = 0.f;

		for ( ; r < 0; r++, k++)
			accum += kernel[k] * src[0][x];

		for ( ; k < ksize && r < h; r++, k++)
			accum += kernel[k] * src[r][x];

		for ( ; k < ksize; k++)
			accum += kernel[k] * src[h-1][x];

		dst[y][x] = accum;
	}
}

void
gaussian_blur(float **img, int w, int h, float sigma)
{
	int ksize;
	float *kernel = create_gaussian_kernel(sigma, &ksize);
	if (!kernel) return;

	float **tmp = array_create(w, h);
	if (!tmp) return;

	filter_rows(img, tmp, w, h, kernel, ksize);
	filter_cols(tmp, img, w, h, kernel, ksize);

	array_free(tmp, h);
	free(kernel);
}

// Sobel mask values
const float SobelX[3][3] = {
    	{-0.25f, 0.f, 0.25f},
	{-0.5f , 0.f, 0.5f },
	{-0.25f, 0.f, 0.25f}
};
const float SobelY[3][3] = {
	{-0.25f,-0.5f,-0.25f},
	{ 0.f  , 0.f , 0.f  },
	{ 0.25f, 0.5f, 0.25f}
};

void
compute_gradient(float **src, float **g_mag, float **g_ang, int w, int h)
{
	int y, x, i, j, r, c;

	for (y = 0; y < h; y++)
	for (x = 0; x < w; x++)
	{
		float gx = 0.f, gy = 0.f;

		for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
		{
			r = y+i-1; if(r<0)r=0; else if(r>=h)r=h-1;
			c = x+j-1; if(c<0)c=0; else if(c>=w)c=w-1;
			
			gx += src[r][c] * SobelX[i][j];
			gy += src[r][c] * SobelY[i][j];
		}

		g_mag[y][x] = hypotf(gy, gx);
		g_ang[y][x] = atan2f(gy, gx);
	}
}

int
is_edge(float **g_mag, float **g_ang, int w, int h, \
        float threshold, int x, int y)
{
	if (g_mag[y][x] >= threshold)
	{
		int dir = ((int) roundf(g_ang[y][x]/M_PI_4) + 4) % 4;

		// horizontal gradient : vertical edge
		if (dir == 0)
		{
			float left  = g_mag[y][x - (x>0  )];
			float right = g_mag[y][x + (x<w-1)];
			if (g_mag[y][x] >= left && g_mag[y][x] >= right)
				return 1;
		}
		// vertical gradient : horizontal edge
		else if (dir == 2)
		{
			float above = g_mag[y - (y>0  )][x];
			float below = g_mag[y + (y<h-1)][x];
			if (g_mag[y][x] >= above && g_mag[y][x] >= below)
				return 1;
		}
		// diagonal gradient : diagonal edge
		else if (dir == 1)
		{
			float above_l = g_mag[y - (y>0  )][x - (x>0  )];
			float below_r = g_mag[y + (y<h-1)][x + (x<w-1)];
			if (g_mag[y][x] >= above_l && g_mag[y][x] >= below_r)
				return 1;
		}
		// diagonal gradient : diagonal edge
		else if (dir == 3)
		{
			float above_r = g_mag[y - (y>0  )][x + (x<w-1)];
			float below_l = g_mag[y + (y<h-1)][x - (x>0  )];
			if (g_mag[y][x] >= above_r && g_mag[y][x] >= below_l)
				return 1;
		}
	}
	return 0;
}

#define PIX(y,x) edge_pix[(y)*w+(x)]

void
mark_edges(float **g_mag, float **g_ang, unsigned char *edge_pix, \
           int w, int h, float threshold)
{
	int y, x;

	for (y = 0; y < h; y++)
	for (x = 0; x < w; x++)
	{
		PIX(y,x) = is_edge(g_mag,g_ang,w,h,threshold,x,y) ? 255 : 0;
	}
}

int
connect_edges(unsigned char *edge_pix, int w, int h)
{
	int changed = 0;
	int y, x;

	// connect horizontal edges
	for (y = 0; y < h  ; y++)
	for (x = 1; x < w-1; x++)
	{
		if (!PIX(y,x) && PIX(y,x+1) && PIX(y,x-1))
			{ PIX(y,x) = 255; changed = 1; }
	}

	// connect vertical edges
	for (x = 0; x < w  ; x++)
	for (y = 1; y < h-1; y++)
	{
		if (!PIX(y,x) && PIX(y+1,x) && PIX(y-1,x))
			{ PIX(y,x) = 255; changed = 1; }
	}

	// connect diagonal edges
	for (y = 1; y < h-1; y++)
	for (x = 1; x < w-1; x++)
	{
		if (!PIX(y,x) && PIX(y-1,x-1) && PIX(y+1,x+1))
			{ PIX(y,x) = 255; changed = 1; }
		if (!PIX(y,x) && PIX(y-1,x+1) && PIX(y+1,x-1))
			{ PIX(y,x) = 255; changed = 1; }
	}

	return changed;
}

void
detect_edges(Image *img, float sigma, float threshold, \
             unsigned char *edge_pix, PList *edge_pts)
{
	int x, y;
	int w = img->w;
	int h = img->h;
	int changed;

#ifdef __DEVICE_EMULATION__
	// This code will be compiled when emulation mode is enabled.
	// Set small block size...
	// ...
#else
	// Otherwise, this code will be compiled.
	// Set arbitrary block size...
	// ...
#endif
	// After calling a CUDA function or kernel,
	// use this macro to check for errors:
	//
	//     CUDA_CHECK("label");
	//
	// When computing grid and block dimensions, beware of rounding errors.
	// This macro may be helpful:
	//
	//      DIVCEIL(x,y)    // ceil(x/y) where x,y are positive integers
	//

	// convert image to grayscale
	float **gray = array_create(w, h);
	timer_start();
	convert_grayscale(img, gray);
	timer_end("convert_grayscale");

	// blur grayscale image
	timer_start();
	gaussian_blur(gray, w, h, sigma);
	timer_end("gaussian_blur");

	// compute gradient of blurred image
	float **g_mag = array_create(w, h);
	float **g_ang = array_create(w, h);
	timer_start();
	compute_gradient(gray, g_mag, g_ang, w, h);
	timer_end("compute_gradient");

	// mark edge pixels
	timer_start();
	mark_edges(g_mag, g_ang, edge_pix, w, h, threshold);
	timer_end("mark_edges");

	// connect edge pixels
	timer_start();
	do changed = connect_edges(edge_pix, w, h);
	while (changed);
	timer_end("connect_edges");

	// add edge points to list
	timer_start();
	if (edge_pts)
	{
		for (y = 0; y < h; y++)
		for (x = 0; x < w; x++)
		{
			if (PIX(y,x)) PList_push(edge_pts, x, y);
		}
	}
	timer_end("add to list");

	// cleanup
	array_free(g_mag, h);
	array_free(g_ang, h);
	array_free(gray, h);
}
