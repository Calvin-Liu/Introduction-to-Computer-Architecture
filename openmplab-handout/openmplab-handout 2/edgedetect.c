#include "header.h"
#include <math.h>
#include <stdlib.h>

float **
array_create(int w, int h)
{
	float **ptr = malloc(h * sizeof(float *));
	int i;
	for (i = 0; i < h; i++)
		ptr[i] = malloc(w * sizeof(float));
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

float
convolve(float *kernel, float *ringbuf, int ksize, int i0)
{
	int i;
	float sum = 0.f;
	for (i = 0; i < ksize; i++)
	{
		sum += kernel[i] * ringbuf[i0++];
		if (i0 == ksize) i0 = 0;
	}
	return sum;
}

void
gaussian_blur(float **src, float **dst, int w, int h, float sigma)
{
	int x, y, i;
	int ksize = (int)(sigma * 2.f * 4.f + 1) | 1;
	int halfk = ksize / 2;
	float scale = -0.5f/(sigma*sigma);
	float sum = 0.f;
	float *kernel, *ringbuf;
	int xmax = w - halfk;
	int ymax = h - halfk;

	// if sigma too small, just copy src to dst
	if (ksize <= 1)
	{
		for (y = 0; y < h; y++)
		for (x = 0; x < w; x++)
			dst[y][x] = src[y][x];
		return;
	}

	// create Gaussian kernel
	kernel = malloc(ksize * sizeof(float));
	ringbuf = malloc(ksize * sizeof(float));

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

	// blur each row
	for (y = 0; y < h; y++)
	{
		int bufi0 = ksize-1;
		float tmp = src[y][0];
		for (x = 0; x < halfk  ; x++) ringbuf[x] = tmp;
		for (     ; x < ksize-1; x++) ringbuf[x] = src[y][x-halfk];

		for (x = 0; x < xmax; x++)
		{
			ringbuf[bufi0++] = src[y][x+halfk];
			if (bufi0 == ksize) bufi0 = 0;
			dst[y][x] = convolve(kernel, ringbuf, ksize, bufi0);
		}

		tmp = src[y][w-1];
		for ( ; x < w; x++)
		{
			ringbuf[bufi0++] = tmp;
			if (bufi0 == ksize) bufi0 = 0;
			dst[y][x] = convolve(kernel, ringbuf, ksize, bufi0);
		}
	}

	// blur each column
	for (x = 0; x < w; x++)
	{
		int bufi0 = ksize-1;
		float tmp = dst[0][x];
		for (y = 0; y < halfk  ; y++) ringbuf[y] = tmp;
		for (     ; y < ksize-1; y++) ringbuf[y] = dst[y-halfk][x];

		for (y = 0; y < ymax; y++)
		{
			ringbuf[bufi0++] = dst[y+halfk][x];
			if (bufi0 == ksize) bufi0 = 0;

			dst[y][x] = convolve(kernel, ringbuf, ksize, bufi0);
		}

		tmp = dst[h-1][x];
		for ( ; y < h; y++)
		{
			ringbuf[bufi0++] = tmp;
			if (bufi0 == ksize) bufi0 = 0;

			dst[y][x] = convolve(kernel, ringbuf, ksize, bufi0);
		}
	}

	// clean up
	free(kernel);
	free(ringbuf);
}

void
compute_gradient(float **src, int w, int h, float **g_mag, float **g_ang)
{
	// Sobel mask values
	const float mx0[] = {-0.25f, 0.f, 0.25f};
	const float mx1[] = {-0.5f , 0.f, 0.5f };
	const float mx2[] = {-0.25f, 0.f, 0.25f};
	const float *mx[] = {mx0, mx1, mx2};
	const float my0[] = {-0.25f,-0.5f,-0.25f};
	const float my1[] = { 0.f  , 0.f , 0.f  };
	const float my2[] = { 0.25f, 0.5f, 0.25f};
	const float *my[] = {my0, my1, my2};

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
			
			gx += src[r][c] * mx[i][j];
			gy += src[r][c] * my[i][j];
		}
		g_mag[y][x] = hypotf(gy, gx);
		g_ang[y][x] = atan2f(gy, gx);
	}
}

int
is_edge(float **g_mag, float **g_ang, float threshold, int x, int y, int w, int h)
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

void
detect_edges(Image *img, float sigma, float threshold, \
             unsigned char *edge_pix, PList *edge_pts)
{
	int x, y;
	int w = img->w;
	int h = img->h;

	// convert image to grayscale
	float **gray = array_create(w, h);
	convert_grayscale(img, gray);

	// blur grayscale image
	float **gray2 = array_create(w, h);
	gaussian_blur(gray, gray2, w, h, sigma);

	// compute gradient of blurred image
	float **g_mag = array_create(w, h);
	float **g_ang = array_create(w, h);
	compute_gradient(gray2, w, h, g_mag, g_ang);

	// mark edge pixels
	#define PIX(y,x) edge_pix[(y)*w+(x)]
	for (y = 0; y < h; y++)
	for (x = 0; x < w; x++)
	{
		PIX(y,x) = is_edge(g_mag,g_ang,threshold,x,y,w,h) ? 255 : 0;
	}

	// connect horizontal edges
	for (y = 0; y < h  ; y++)
	for (x = 1; x < w-1; x++)
	{
		if (!PIX(y,x) && PIX(y,x+1) && PIX(y,x-1))
			PIX(y,x) = 255;
	}

	// connect vertical edges
	for (x = 0; x < w  ; x++)
	for (y = 1; y < h-1; y++)
	{
		if (!PIX(y,x) && PIX(y+1,x) && PIX(y-1,x))
			PIX(y,x) = 255;
	}

	// connect diagonal edges
	for (y = 1; y < h-1; y++)
	for (x = 1; x < w-1; x++)
	{
		if (!PIX(y,x) && PIX(y-1,x-1) && PIX(y+1,x+1))
			PIX(y,x) = 255;
		if (!PIX(y,x) && PIX(y-1,x+1) && PIX(y+1,x-1))
			PIX(y,x) = 255;
	}

	// add edge points to list
	if (edge_pts)
	{
		for (y = 0; y < h; y++)
		for (x = 0; x < w; x++)
		{
			if (PIX(y,x)) PList_push(edge_pts, x, y);
		}
	}

	// cleanup
	array_free(g_mag, h);
	array_free(g_ang, h);
	array_free(gray2, h);
	array_free(gray, h);
}
