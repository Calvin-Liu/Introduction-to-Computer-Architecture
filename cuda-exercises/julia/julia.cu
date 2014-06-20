#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ceil(x/y) for integers
#define DIVCEIL(x,y)  (((x)+(y)-1) / (y))

// function prototypes
void
checkCUDAerror(const char* msg);

__global__ void
juliaKernel(unsigned char* img, int W, int H, float scale);

int
writeBMP(unsigned char* img, int w, int h, FILE* ofile);

////////////////////////////////////////////////////////////////////////////////

int
main(int argc, char *argv[])
{
	// Parse command-line arguments
	if (argc < 3)
	{
		fprintf(stderr, "Usage: %s WIDTH HEIGHT [ZOOM [OUT.bmp]]\n", argv[0]);
		return 1;
	}

	// Image dimensions
	int W = atoi(argv[1]);
	int H = atoi(argv[2]);
	if (W <= 0 || H <= 0)
	{
		fprintf(stderr, "Invalid dimensions.\n");
		return 1;
	}

	// >1 for zoom-in, <1 for zoom-out
	float zoom = (argc > 3) ? atof(argv[3]) : 0.f;
	float scale = (isfinite(zoom) && zoom > 0.f) ? (1.5f/zoom) : 1.5f;

	// Open output file
	const char *ofname = (argc > 4) ? argv[4] : "out.bmp";
	FILE *ofile = fopen(ofname, "wb");
	if (!ofile)
	{
		fprintf(stderr, "Could not open output file '%s'.\n", ofname);
		return 1;
	}

	// Allocate image pixels on device and host
	int nbytes = W * H * sizeof(unsigned char);
	unsigned char *img, *img_d;
	img  = (unsigned char*) malloc(nbytes);
	cudaMalloc(&img_d, nbytes);
	checkCUDAerror("cudaMalloc");

	// Step 1 of 4: Compute grid and block dimensions
	const int BLKDIM = 4;
	int nblkx =     ;
	int nblky =     ;
	dim3 gridDim(nblkx,nblky);
	dim3 blockDim(BLKDIM,BLKDIM);
	fprintf(stderr, "Image W=%d H=%d\n", W, H);
	fprintf(stderr, "%dx%d blocks\n", nblkx, nblky);
	fprintf(stderr, "%dx%d threads per block\n", BLKDIM, BLKDIM);

	// Step 2 of 4: Run kernel to compute pixel values
	juliaKernel<<<    ,    >>>(    );
	checkCUDAerror("juliaKernel");

	// Step 4 of 4: Copy image from GPU
	cudaMemcpy(    );
	checkCUDAerror("cudaMemcpy");

	// Write image as a BMP file
	writeBMP(img, W, H, ofile);

	// Cleanup
	free(img);
	cudaFree(img_d);
	return 0;
}

void
checkCUDAerror(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
	{
		const char *s = cudaGetErrorString(err);
		fprintf(stderr, "CUDA error: %s: %s.\n", msg, s);
		exit(1);
	}                         
}

////////////////////////////////////////////////////////////////////////////////

// Complex floating-point type
typedef struct { float real, imag; } cplx_t;

// Complex magnitude
inline float
cmag(cplx_t c)
{
	return hypotf(c.real, c.imag);
}

// Complex addition
inline cplx_t
cadd(cplx_t a, cplx_t b)
{
	cplx_t c;
	c.real = a.real + b.real;
	c.imag = a.imag + b.imag;
	return c;
}

// Complex square
inline cplx_t
csqr(cplx_t c)
{
	cplx_t z;
	z.real = c.real*c.real - c.imag*c.imag;
	z.imag = c.real*c.imag*2.f;
	return z;
}

// Compute the intensity of the pixel (x,y) in a WxH image of the Julia set.
__device__ unsigned char
julia(int x, int y, int W, int H, float scale)
{
	const float halfW = 0.5f*(float)W;
	const float halfH = 0.5f*(float)H;

	// Map pixel (x,y) to the complex plane.
	// The origin is at the center of the image.
	cplx_t a;
	a.real = scale * ( (halfW-(float)x) / halfW );
	a.imag = scale * ( (halfH-(float)y) / halfW );

	// Magic constant c
	const cplx_t c = {-0.8f, 0.156f};

	// The intensity of the pixel depends on how fast `a` moves away from
	// the origin when repeatedly performing the step `a = a^2 + c`.
	int i;
	for (i = 0; i < 255; i++)
	{
		a = cadd(c, csqr(a));
		// stop when `a` has moved far enough away
		if (cmag(a) > 1000.f) break;
	}

	// Lighter pixels have higher value (lower `i`)
	// and represent points that grew quickly (broke out of the loop early).
	return (unsigned char)(255 - i);
	// For white-on-black, just remove `255 - `
}

// Step 3 of 4: implement the kernel
__global__ void
juliaKernel(unsigned char* img, int W, int H, float scale)
{
	// Each thread should compute a single pixel in `img`.
	// ...
}

////////////////////////////////////////////////////////////////////////////////

#define BMPHDRSZ 54
#define PALETTE_COUNT 256

typedef struct BMPHeader {
	int16_t space;		// for alignment (not part of header)
	int16_t magic;		// BMP magic number: 0x42 0x4D
	int32_t file_size;	// total file size
	int32_t unused;		//
	int32_t hdr_size;	// header size (54)
	int32_t hdr_rem;	// remaining header bytes (40)
	int32_t img_w;		// image width
	int32_t img_h;		// image height
	int16_t color_planes;	// 1
	int16_t bits_per_pix;	// RGB:24, grayscale:8
	int32_t compression;	// 0 (none)
	int32_t pix_data_size;	// (W+pad)*H*(3 or 1)
	int32_t phys_res_horz;	// 2835 pixel/meter (72 DPI)
	int32_t phys_res_vert;	// 2835 pixel/meter (72 DPI)
	int32_t palette_count;	// RGB:0, grayscale:256
	int32_t important_clr;	// 0 (all)
} BMPHeader;

static const BMPHeader bmp_header = {
	0, 0x4d42,
	0, 0,
	54, 40,
	0, 0,
	1, 24,
	0, 0,
	2835,2835,
	0, 0
};

int
writeBMP(unsigned char *pix, int w, int h, FILE *ofile)
{
	BMPHeader hdr = bmp_header;
	int y, i, n, rowpad, zero = 0;
	size_t m;
	assert(sizeof(hdr) == BMPHDRSZ+2);

	if (!pix) { fprintf(stderr, "NULL pix"); return -1; }
	if (!ofile) { fprintf(stderr, "NULL output file"); return -1; }
	if (w <= 0 || h <= 0)
	{
		fprintf(stderr, "Invalid dimensions: %d x %d", w, h);
		return -1;
	}

	// fill in BMP header
	hdr.img_w = w;
	hdr.img_h = h;
	rowpad = (4 - w % 4) % 4;
	n = (w + rowpad) * h;
	hdr.pix_data_size = n;
	n += BMPHDRSZ + (PALETTE_COUNT*4);
	hdr.file_size = n;
	hdr.bits_per_pix = 8;
	hdr.palette_count = 256;

	// write header to file
	m = fwrite(&hdr.magic, BMPHDRSZ, 1, ofile);
	if (m != 1) { fprintf(stderr, "Error writing BMP header\n"); return 1; }

	// write grayscale palette
	for (i = 0; i < PALETTE_COUNT; i++)
	{
		int pix = i | (i<<8) | (i<<16);
		fwrite(&pix, 4, 1, ofile);
	}

	// write pixel data
	for (y = h-1; y >= 0; y--)
	{
		i = y * w;
		m = fwrite(pix+i, 1, w, ofile);
		if (m != w) { fprintf(stderr, "Error writing pixels\n"); return 1; }
		if (rowpad) fwrite(&zero, 1, rowpad, ofile);
	}

	return 0;
}
