This directory contains six CUDA exercises.
(the first five are from Nvidia)
In order of increasing complexity:

cudaMallocAndMemcpy/
	cudaMallocAndMemcpy.cu
		Basics of GPU memory management.
myFirstKernel/
	myFirstKernel.cu
		Simple kernel using threadIdx, blockIdx, blockDim.
reverseArray/
	reverseArray_singleblock.cu
		Reverse an array using a single thread block.
	reverseArray_multiblock.cu
		Reverse an array using multiple thread blocks.
	reverseArray_multiblock_fast.cu
		Reverse an array using multiple thread blocks
		and shared memory.
julia/
	julia.cu
		Render the Julia set using one thread per pixel.
		Don't forget to view the output image!

In each subdirectory, there is also a `solution` directory containing solution
files.

Before you can compile on lnxsrv, run the following commands to setup
environment variables:

If you use csh/tcsh:

	setenv PATH ${PATH}:/usr/local/cuda/bin
	setenv MANPATH ${MANPATH}:/usr/local/cuda/man
	setenv LD_LIBRARY_PATH /usr/local/cuda/lib64

	To have these automatically run when you login,
	append them to the file `.cshrc` in your home directory.

If you use Bash:

	export PATH=$PATH:/usr/local/cuda/bin
	export MANPATH=$MANPATH:/usr/local/cuda/man
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64

	To have these automatically run when you login,
	append them to the file `.profile` in your home directory.

To compile and run on lnxsrv:

	nvcc -deviceemu FILE.cu
	./a.out

gcc options such as `-o` may also be used.
Note that device emulation will be slow when simulating many GPU threads.
