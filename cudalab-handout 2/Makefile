.PHONY: seq cuda run clean check checkmem

SRC ?= edgedetect
PROG ?= edgedetect
IN ?= img.bmp
OUT ?= out.bmp

CCPRE=
ifdef DEBUG
CCPRE+=-g
endif
ifndef NOMTRACE
CCPRE+=-DMTRACE
endif

seq:
	gcc -Wall -O3 $(CCPRE) $(SRC).c main.c -o $(PROG) -lm
cuda:
	ln -fs main.c main.cu
	nvcc -ccbin /usr/bin -deviceemu -O $(CCPRE) $(SRC).cu main.cu -o $(PROG)
run:
	./$(PROG) $(IN) 3 10 $(OUT) pts.txt
clean:
	@rm -f $(PROG) main.cu core.* mtrace.out out.bmp pts.txt
check:
	@sort pts.txt | diff --brief correct.txt - || true
checkmem:
	@mtrace $(PROG) mtrace.out | grep -v -e '0x3[0-9a-f]\{9\}$$' -e 'Free [0-9]\+ was' || true

