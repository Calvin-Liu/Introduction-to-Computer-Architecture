.PHONY: seq omp run clean check

SRC ?= edgedetect.c
PROG ?= edgedetect
IN ?= img.bmp
OUT ?= out.bmp

CCPRE=-Wall -I. -O3
CCPOST=main.c -o $(PROG) -lm

ifdef GPROF
CCPRE+=-O2 -pg
endif

ifdef MTRACE
CCPRE+=-DMTRACE
endif

seq:
	gcc $(CCPRE) $(SRC) $(CCPOST)
omp:
	gcc $(CCPRE) -fopenmp $(SRC) $(CCPOST)
run:
	./$(PROG) $(IN) 5 10 $(OUT) pts.txt
clean:
	rm -f $(PROG) core.* gmon.out mtrace.out out.bmp pts.txt
check:
	@sort pts.txt | diff --brief correct.txt - || true
checkmem:
	@mtrace $(PROG) mtrace.out || true
