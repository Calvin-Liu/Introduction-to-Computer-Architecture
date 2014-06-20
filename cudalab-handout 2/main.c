#include "header.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#ifdef MTRACE
#include <mcheck.h>
#endif

////////////////////////////////////////////////////////////////////////////////
// TIMER FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

static double elapsed = 0.0;
static struct timeval start_time;

void
timer_reset()
{
	elapsed = 0.0;
}

void
timer_start()
{
	gettimeofday(&start_time, NULL);
}

void
timer_stop()
{
	struct timeval end_time;
	gettimeofday(&end_time, NULL);
	elapsed += (double)(end_time.tv_sec - start_time.tv_sec) +
	           (double)(end_time.tv_usec - start_time.tv_usec)*1e-6;
}

void
timer_print()
{
	fprintf(stderr, "%g seconds\n", elapsed);
}

void
timer_end(const char *label)
{
	timer_stop();
	fprintf(stderr, "%20s: ", label ? label : "");
	timer_print();
	timer_reset();
}

////////////////////////////////////////////////////////////////////////////////
// POINT LIST
////////////////////////////////////////////////////////////////////////////////

PList *
PList_create(void)
{
	PList *pl = (PList *) malloc(sizeof(PList));
	if (pl)
	{
		pl->len = 0;
		pl->cap = 128;
		pl->pts = (P *) malloc(pl->cap * sizeof(P));
		if (!pl->pts)
		{
			free(pl);
			pl = NULL;
		}
	}
	return pl;
}

void
PList_free(PList *pl)
{
	if (pl)
	{
		free(pl->pts);
		free(pl);
	}
}

void
PList_push(PList *pl, int x, int y)
{
	P p = {x, y};

	if (pl->len == pl->cap)
	{
		pl->cap *= 2;
		pl->pts = (P *) realloc(pl->pts, pl->cap * sizeof(P));
	}

	pl->pts[(pl->len)++] = p;
}

////////////////////////////////////////////////////////////////////////////////
// TYPES AND GLOBAL DATA
////////////////////////////////////////////////////////////////////////////////

#define BMPHDRSZ 54
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

typedef struct RGBA {
	unsigned char b, g, r, a;
} RGBA;

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

#define GRAY_PALETTE_COUNT 256
static const RGBA gray_palette[GRAY_PALETTE_COUNT] = {
	{0,0,0,0},{1,1,1,0},{2,2,2,0},{3,3,3,0},
	{4,4,4,0},{5,5,5,0},{6,6,6,0},{7,7,7,0},
	{8,8,8,0},{9,9,9,0},{10,10,10,0},{11,11,11,0},
	{12,12,12,0},{13,13,13,0},{14,14,14,0},{15,15,15,0},
	{16,16,16,0},{17,17,17,0},{18,18,18,0},{19,19,19,0},
	{20,20,20,0},{21,21,21,0},{22,22,22,0},{23,23,23,0},
	{24,24,24,0},{25,25,25,0},{26,26,26,0},{27,27,27,0},
	{28,28,28,0},{29,29,29,0},{30,30,30,0},{31,31,31,0},
	{32,32,32,0},{33,33,33,0},{34,34,34,0},{35,35,35,0},
	{36,36,36,0},{37,37,37,0},{38,38,38,0},{39,39,39,0},
	{40,40,40,0},{41,41,41,0},{42,42,42,0},{43,43,43,0},
	{44,44,44,0},{45,45,45,0},{46,46,46,0},{47,47,47,0},
	{48,48,48,0},{49,49,49,0},{50,50,50,0},{51,51,51,0},
	{52,52,52,0},{53,53,53,0},{54,54,54,0},{55,55,55,0},
	{56,56,56,0},{57,57,57,0},{58,58,58,0},{59,59,59,0},
	{60,60,60,0},{61,61,61,0},{62,62,62,0},{63,63,63,0},
	{64,64,64,0},{65,65,65,0},{66,66,66,0},{67,67,67,0},
	{68,68,68,0},{69,69,69,0},{70,70,70,0},{71,71,71,0},
	{72,72,72,0},{73,73,73,0},{74,74,74,0},{75,75,75,0},
	{76,76,76,0},{77,77,77,0},{78,78,78,0},{79,79,79,0},
	{80,80,80,0},{81,81,81,0},{82,82,82,0},{83,83,83,0},
	{84,84,84,0},{85,85,85,0},{86,86,86,0},{87,87,87,0},
	{88,88,88,0},{89,89,89,0},{90,90,90,0},{91,91,91,0},
	{92,92,92,0},{93,93,93,0},{94,94,94,0},{95,95,95,0},
	{96,96,96,0},{97,97,97,0},{98,98,98,0},{99,99,99,0},
	{100,100,100,0},{101,101,101,0},{102,102,102,0},{103,103,103,0},
	{104,104,104,0},{105,105,105,0},{106,106,106,0},{107,107,107,0},
	{108,108,108,0},{109,109,109,0},{110,110,110,0},{111,111,111,0},
	{112,112,112,0},{113,113,113,0},{114,114,114,0},{115,115,115,0},
	{116,116,116,0},{117,117,117,0},{118,118,118,0},{119,119,119,0},
	{120,120,120,0},{121,121,121,0},{122,122,122,0},{123,123,123,0},
	{124,124,124,0},{125,125,125,0},{126,126,126,0},{127,127,127,0},
	{128,128,128,0},{129,129,129,0},{130,130,130,0},{131,131,131,0},
	{132,132,132,0},{133,133,133,0},{134,134,134,0},{135,135,135,0},
	{136,136,136,0},{137,137,137,0},{138,138,138,0},{139,139,139,0},
	{140,140,140,0},{141,141,141,0},{142,142,142,0},{143,143,143,0},
	{144,144,144,0},{145,145,145,0},{146,146,146,0},{147,147,147,0},
	{148,148,148,0},{149,149,149,0},{150,150,150,0},{151,151,151,0},
	{152,152,152,0},{153,153,153,0},{154,154,154,0},{155,155,155,0},
	{156,156,156,0},{157,157,157,0},{158,158,158,0},{159,159,159,0},
	{160,160,160,0},{161,161,161,0},{162,162,162,0},{163,163,163,0},
	{164,164,164,0},{165,165,165,0},{166,166,166,0},{167,167,167,0},
	{168,168,168,0},{169,169,169,0},{170,170,170,0},{171,171,171,0},
	{172,172,172,0},{173,173,173,0},{174,174,174,0},{175,175,175,0},
	{176,176,176,0},{177,177,177,0},{178,178,178,0},{179,179,179,0},
	{180,180,180,0},{181,181,181,0},{182,182,182,0},{183,183,183,0},
	{184,184,184,0},{185,185,185,0},{186,186,186,0},{187,187,187,0},
	{188,188,188,0},{189,189,189,0},{190,190,190,0},{191,191,191,0},
	{192,192,192,0},{193,193,193,0},{194,194,194,0},{195,195,195,0},
	{196,196,196,0},{197,197,197,0},{198,198,198,0},{199,199,199,0},
	{200,200,200,0},{201,201,201,0},{202,202,202,0},{203,203,203,0},
	{204,204,204,0},{205,205,205,0},{206,206,206,0},{207,207,207,0},
	{208,208,208,0},{209,209,209,0},{210,210,210,0},{211,211,211,0},
	{212,212,212,0},{213,213,213,0},{214,214,214,0},{215,215,215,0},
	{216,216,216,0},{217,217,217,0},{218,218,218,0},{219,219,219,0},
	{220,220,220,0},{221,221,221,0},{222,222,222,0},{223,223,223,0},
	{224,224,224,0},{225,225,225,0},{226,226,226,0},{227,227,227,0},
	{228,228,228,0},{229,229,229,0},{230,230,230,0},{231,231,231,0},
	{232,232,232,0},{233,233,233,0},{234,234,234,0},{235,235,235,0},
	{236,236,236,0},{237,237,237,0},{238,238,238,0},{239,239,239,0},
	{240,240,240,0},{241,241,241,0},{242,242,242,0},{243,243,243,0},
	{244,244,244,0},{245,245,245,0},{246,246,246,0},{247,247,247,0},
	{248,248,248,0},{249,249,249,0},{250,250,250,0},{251,251,251,0},
	{252,252,252,0},{253,253,253,0},{254,254,254,0},{255,255,255,0},
};

////////////////////////////////////////////////////////////////////////////////
// IMAGE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

#define LOG(...) \
	do { \
	fprintf(stderr, "[%s] ", __func__); \
	fprintf(stderr, __VA_ARGS__); \
	fputc('\n', stderr); \
	fflush(stderr); \
	} while(0)

#define LOGF(f,s) \
	LOG("%s: %s", (s), strerror(ferror(f)))

Image *
Image_create(int width, int height)
{
	Image *img;
	int n;

	if (width <= 0 || height <= 0)
	{
		LOG("Invalid image dimensions: %d x %d", width, height);
		return NULL;
	}

	img = (Image *) malloc(sizeof(Image));
	if (img)
	{
		img->w = width;
		img->h = height;

		n = width * height * sizeof(float);
		img->r = (float *) malloc(n);
		img->g = (float *) malloc(n);
		img->b = (float *) malloc(n);

		if (!img->r || !img->g || !img->b)
		{
			free(img->r);
			free(img->g);
			free(img->b);
			free(img);
			img = NULL;
		}
	}

	return img;
}

void
Image_free(Image *img)
{
	if (img)
	{
		free(img->r);
		free(img->g);
		free(img->b);
		free(img);
	}
}

Image *
Image_createFromBMP(FILE *ifile)
{
	struct Image *img = NULL;
	struct BMPHeader hdr;
	struct RGBA *palette = NULL;
	struct RGBA pix;
	int x, y, i, rowpad, isRGB = 0, Bpp, rowrev = 0;
	size_t n;
	assert(sizeof(hdr) == BMPHDRSZ+2);
	assert(sizeof(pix) == 4);

	n = fread(&hdr.magic, BMPHDRSZ, 1, ifile);
	if (n != 1)
	{
		LOGF(ifile, "Error reading BMP header");
		goto done;
	}

	if (hdr.magic != 0x4d42)
	{
		LOG("BMP header has invalid magic number 0x%04x.", hdr.magic);
		goto done;
	}

	if (hdr.bits_per_pix == 8)
	{
		// read palette for 8bpp
		if (hdr.palette_count < 1 || hdr.palette_count > 256)
		{
			LOG("Invalid palette count: %d", hdr.palette_count);
			goto done;
		}

		palette = (RGBA *) malloc(sizeof(RGBA) * hdr.palette_count);
		if (!palette)
		{
			LOG("alloc palette failed");
			goto done;
		}
		n = fread(palette, sizeof(RGBA), hdr.palette_count, ifile);
		if (n != hdr.palette_count)
		{
			LOGF(ifile, "Error reading BMP palette");
			goto done;
		}
	}
	else if (hdr.bits_per_pix == 24 || hdr.bits_per_pix == 32)
	{
		isRGB = 1;
	}
	else
	{
		LOG("Unsupported bits-per-pixel: %d", hdr.bits_per_pix);
		goto done;
	}

	// create image
	if (hdr.img_h < 0) { rowrev = 1; hdr.img_h = -hdr.img_h; }
	img = Image_create(hdr.img_w, hdr.img_h);
	if (!img) goto done;

	// read pixels from file
	rowpad = (4 - (hdr.img_w % 4)) % 4;
	Bpp = hdr.bits_per_pix / 8; // 1 or 3 or 4

	for (y = 0; y < hdr.img_h; y++)
	{
		i = hdr.img_w * (rowrev ? y : (hdr.img_h-1-y));
		for (x = 0; x < hdr.img_w; x++, i++)
		{
			if (isRGB)
			{
				fread(&pix, 1, Bpp, ifile);
			}
			else
			{
				unsigned char index;
				fread(&index, 1, 1, ifile);
				pix = palette[index];
			}
			// save pixel as floats in image
			img->b[i] = (float)pix.b;
			img->g[i] = (float)pix.g;
			img->r[i] = (float)pix.r;
		}
		// each row is padded to a multiple of 4 bytes
		if (rowpad) fseek(ifile, rowpad, SEEK_CUR);
	}
done:
	free(palette);
	return img;
}

int
Image_writeBMP(Image *img, FILE *ofile)
{
	BMPHeader hdr = bmp_header;
	int x, y, i;
	int n, rowpad, zero = 0;
	size_t m;

	if (!img) { LOG("NULL image"); return -1; }
	if (!ofile) { LOG("NULL output file"); return -1; }

	// fill in BMP header
	hdr.img_w = img->w;
	hdr.img_h = img->h;
	rowpad = (4 - (3*hdr.img_w) % 4) % 4;
	n = (3*hdr.img_w + rowpad) * hdr.img_h;
	hdr.pix_data_size = n;
	n += BMPHDRSZ;
	hdr.file_size = n;

	// write header to file
	m = fwrite(&hdr.magic, BMPHDRSZ, 1, ofile);
	if (m != 1) { LOGF(ofile, "Error writing BMP header"); return 1; }

	// write pixel data
	for (y = hdr.img_h-1; y >= 0; y--)
	{
		i = y * hdr.img_w;
		for (x = 0; x < hdr.img_w; x++, i++)
		{
			RGBA pix;
			pix.b = roundf(img->b[i]);
			pix.g = roundf(img->g[i]);
			pix.r = roundf(img->r[i]);
			m = fwrite(&pix, 1, 3, ofile);
			if (m != 3) { LOGF(ofile, "Error writing pixels"); return 1; }
		}
		if (rowpad) fwrite(&zero, 1, rowpad, ofile);
	}

	return 0;
}

int
grayscale_writeBMP(unsigned char *pix, int w, int h, FILE *ofile)
{
	BMPHeader hdr = bmp_header;
	int y, i, n, rowpad, zero = 0;
	size_t m;
	assert(sizeof(hdr) == BMPHDRSZ+2);
	assert(sizeof(RGBA) == 4);

	if (!pix) { LOG("NULL pix"); return -1; }
	if (!ofile) { LOG("NULL output file"); return -1; }
	if (w <= 0 || h <= 0)
	{
		LOG("Invalid dimensions: %d x %d", w, h);
		return -1;
	}

	// fill in BMP header
	hdr.img_w = w;
	hdr.img_h = h;
	rowpad = (4 - w % 4) % 4;
	n = (w + rowpad) * h;
	hdr.pix_data_size = n;
	n += BMPHDRSZ + (GRAY_PALETTE_COUNT*sizeof(RGBA));
	hdr.file_size = n;
	hdr.bits_per_pix = 8;
	hdr.palette_count = 256;

	// write header to file
	m = fwrite(&hdr.magic, BMPHDRSZ, 1, ofile);
	if (m != 1) { LOGF(ofile, "Error writing BMP header"); return 1; }

	// write grayscale palette
	fwrite(gray_palette, sizeof(RGBA), GRAY_PALETTE_COUNT, ofile);

	// write pixel data
	for (y = h-1; y >= 0; y--)
	{
		i = y * w;
		m = fwrite(pix+i, 1, w, ofile);
		if (m != w) { LOGF(ofile, "Error writing pixels"); return 1; }
		if (rowpad) fwrite(&zero, 1, rowpad, ofile);
	}

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// MAIN()
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	FILE *ifile, *ofile, *ofile2;
	const char *ofname, *ofname2;
	Image *img;
	unsigned char *edge_pix;
	PList *edge_pts;
	float threshold, radius;
#ifdef MTRACE
	setenv("MALLOC_TRACE", "mtrace.out", 0);
	mtrace();
#endif
	// check args
	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s IN.bmp [BLUR_RADIUS [THRESHOLD [OUT.bmp [POINTS.txt]]]]\n", argv[0]);
		return 1;
	}

	// open input image
	ifile = fopen(argv[1], "r");
	if (!ifile)
	{
		fprintf(stderr, "Error opening '%s'.\n", argv[1]);
		return 1;
	}

	// read input image
	img = Image_createFromBMP(ifile);
	fclose(ifile);
	if (!img) { LOG("Image_createFromBMP failed"); return -1; }

	// read blur radius
	radius = (argc >= 3) ? atof(argv[2]) : -1.f;
	if (!isfinite(radius) || radius < 0.f)
	{
		radius = 5.f;
		fprintf(stderr, "Using default blur radius: %g\n", radius);
	}

	// read threshold
	threshold = (argc >= 4) ? atof(argv[3]) : -1.f;
	if (!isfinite(threshold) || threshold < 0.f)
	{
		threshold = 10.f;
		fprintf(stderr, "Using default threshold: %g\n", threshold);
	}

	// open output image
	ofname = (argc >= 5) ? argv[4] : "out.bmp";
	ofile = fopen(ofname, "wb");
	if (!ofile) fprintf(stderr, "Error opening '%s'.\n", ofname);

	// open output points file
	ofname2 = (argc >= 6) ? argv[5] : "pts.txt";
	ofile2 = fopen(ofname2, "w");
	if (!ofile2) fprintf(stderr, "Error opening '%s'.\n", ofname2);

	// create output structures
	edge_pix = (unsigned char *) malloc(img->w * img->h);
	if (!edge_pix) { LOG("alloc edge_pix failed"); return -1; }
	edge_pts = PList_create();
	if (!edge_pts) LOG("alloc edge_pts failed");

	// find edge pixels
timer_start();
struct timeval tmp = start_time; // hack so other calls don't affect main timer

	detect_edges(img, radius, threshold, edge_pix, edge_pts);

start_time = tmp; elapsed = 0.0;
timer_end("detect_edges");

	// write output image (edge pixels)
	if (ofile && edge_pix)
	{
		grayscale_writeBMP(edge_pix, img->w, img->h, ofile);
		fclose(ofile);
	}

	// write list of edge points
	if (ofile2 && edge_pts)
	{
		int i, n = edge_pts->len;
		for (i = 0; i < n; i++)
		{
			P p = edge_pts->pts[i];
			fprintf(ofile2, "%04d,%04d\n", p.y, p.x);
		}
		fclose(ofile2);
	}

	// cleanup
	Image_free(img);
	free(edge_pix);
	PList_free(edge_pts);
#ifdef MTRACE
	muntrace();
#endif
	return 0;
}
