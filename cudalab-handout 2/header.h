#ifndef HEADER_H_
#define HEADER_H_

#include <stdio.h>

// Start the timer.
// If already started, the previous call is "forgotten".
void
timer_start();

// Pause the timer and record the elapsed time.
// Timer can be started again later if desired.
void
timer_stop();

// Reset the elapsed time to 0.
void
timer_reset();

// Print elapsed time to stdout.
// Timer should be stopped first.
void
timer_print();

// Helper to stop/reset the timer, and print elapsed time with a label.
void
timer_end(const char *label);

////////////////////////////////////////////////////////////////////////////////

typedef struct P {
	int x, y;
} P;

typedef struct PList {
	int len, cap;
	P *pts;
} PList;

// Create an empty PList.
PList *
PList_create(void);

// Free all memory associated with a PList.
void
PList_free(PList *pl);

// Add the point (x,y) to a PList.
void
PList_push(PList *pl, int x, int y);

////////////////////////////////////////////////////////////////////////////////

typedef struct Image {
	int w, h;
	float *r, *g, *b;
} Image;

// Create an Image with the given dimensions.
Image *
Image_create(int width, int height);

// Create an Image from a BMP file.
Image *
Image_createFromBMP(FILE *ifile);

// Free all memory associated with an Image.
void
Image_free(Image *img);

// Write an image as a color BMP file.
int
Image_writeBMP(Image *img, FILE *ofile);

// Write a grayscale image as a BMP file.
// `pix` is an array of pixels in row-major order.
// Each byte represents a pixel (0=black,255=white).
int
grayscale_writeBMP(unsigned char *pix, int w, int h, FILE *ofile);

// Find edge pixels in the given image.
//   sigma     : blur radius
//   threshold : minimum gradient magnitude
//   edge_pix  : output image (B&W)
//   edge_pts  : list of edge points (optional)
void
detect_edges(Image *img, float sigma, float threshold, \
             unsigned char *edge_pix, PList *edge_pts);

// ceil(x/y) for integers
// Useful for setting CUDA grid/block sizes.
#define DIVCEIL(x,y) (((x)+(y)-1) / (y))

// Check for errors after launching a CUDA kernel.
#define CUDA_CHECK(msg) \
	do { \
		cudaError_t err = cudaGetLastError(); \
		if (err != cudaSuccess) { \
			fprintf(stderr, "CUDA Error: %s: %s.\n", \
			        (msg), cudaGetErrorString(err)); \
			exit(-1); \
		} \
	} while(0)

#endif // HEADER_H_
