/* 
 * CS:APP Data Lab 
 * 
 * <Calvin Liu 804182525>
 * 
 * bits.c - Source file with your solutions to the Lab.
 *          This is the file you will hand in to your instructor.
 *
 * WARNING: Do not include the <stdio.h> header; it confuses the dlc
 * compiler. You can still use printf for debugging without including
 * <stdio.h>, although you might get a compiler warning. In general,
 * it's not good practice to ignore compiler warnings, but in this
 * case it's OK.  
 */

#if 0
/*
 * Instructions to Students:
 *
 * STEP 1: Read the following instructions carefully.
 */

You will provide your solution to the Data Lab by
editing the collection of functions in this source file.

INTEGER CODING RULES:
 
  Replace the "return" statement in each function with one
  or more lines of C code that implements the function. Your code 
  must conform to the following style:
 
  int Funct(arg1, arg2, ...) {
      /* brief description of how your implementation works */
      int var1 = Expr1;
      ...
      int varM = ExprM;

      varJ = ExprJ;
      ...
      varN = ExprN;
      return ExprR;
  }

  Each "Expr" is an expression using ONLY the following:
  1. Integer constants 0 through 255 (0xFF), inclusive. You are
      not allowed to use big constants such as 0xffffffff.
  2. Function arguments and local variables (no global variables).
  3. Unary integer operations ! ~
  4. Binary integer operations & ^ | + << >>
    
  Some of the problems restrict the set of allowed operators even further.
  Each "Expr" may consist of multiple operators. You are not restricted to
  one operator per line.

  You are expressly forbidden to:
  1. Use any control constructs such as if, do, while, for, switch, etc.
  2. Define or use any macros.
  3. Define any additional functions in this file.
  4. Call any functions.
  5. Use any other operations, such as &&, ||, -, or ?:
  6. Use any form of casting.
  7. Use any data type other than int.  This implies that you
     cannot use arrays, structs, or unions.

 
  You may assume that your machine:
  1. Uses 2s complement, 32-bit representations of integers.
  2. Performs right shifts arithmetically.
  3. Has unpredictable behavior when shifting an integer by more
     than the word size.

EXAMPLES OF ACCEPTABLE CODING STYLE:
  /*
   * pow2plus1 - returns 2^x + 1, where 0 <= x <= 31
   */
  int pow2plus1(int x) {
     /* exploit ability of shifts to compute powers of 2 */
     return (1 << x) + 1;
  }

  /*
   * pow2plus4 - returns 2^x + 4, where 0 <= x <= 31
   */
  int pow2plus4(int x) {
     /* exploit ability of shifts to compute powers of 2 */
     int result = (1 << x);
     result += 4;
     return result;
  }

FLOATING POINT CODING RULES

For the problems that require you to implent floating-point operations,
the coding rules are less strict.  You are allowed to use looping and
conditional control.  You are allowed to use both ints and unsigneds.
You can use arbitrary integer and unsigned constants.

You are expressly forbidden to:
  1. Define or use any macros.
  2. Define any additional functions in this file.
  3. Call any functions.
  4. Use any form of casting.
  5. Use any data type other than int or unsigned.  This means that you
     cannot use arrays, structs, or unions.
  6. Use any floating point data types, operations, or constants.


NOTES:
  1. Use the dlc (data lab checker) compiler (described in the handout) to 
     check the legality of your solutions.
  2. Each function has a maximum number of operators (! ~ & ^ | + << >>)
     that you are allowed to use for your implementation of the function. 
     The max operator count is checked by dlc. Note that '=' is not 
     counted; you may use as many of these as you want without penalty.
  3. Use the btest test harness to check your functions for correctness.
  4. Use the BDD checker to formally verify your functions
  5. The maximum number of ops for each function is given in the
     header comment for each function. If there are any inconsistencies 
     between the maximum ops in the writeup and in this file, consider
     this file the authoritative source.

/*
 * STEP 2: Modify the following functions according the coding rules.
 * 
 *   IMPORTANT. TO AVOID GRADING SURPRISES:
 *   1. Use the dlc compiler to check that your solutions conform
 *      to the coding rules.
 *   2. Use the BDD checker to formally verify that your solutions produce 
 *      the correct answers.
 */


#endif
/*
 * leftBitCount - returns count of number of consective 1's in
 *     left-hand (most significant) end of word.
 *   Examples: leftBitCount(-1) = 32, leftBitCount(0xFFF0F0F0) = 12
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 50
 *   Rating: 4
 */
int leftBitCount(int x) {
    int s16, c16, x16, s8, c8, x8, s4, c4, x4, s2, c2, x2, s1, c1, x1;
    c16 = !((x >> 16) + 1);
    s16 = c16 << 4;
    x16 = x << s16;
    
    c8 = !((x16 >> 24) + 1);
    s8 = c8 << 3;
    x8 = x16 << s8;
    
    c4 = !((x8 >> 28) + 1);
    s4 = c4 << 2;
    x4 = x8 << s4;
    
    c2 = !((x4 >> 30) + 1);
    s2 = c2 << 1;
    x2 = x4 << s2;
    
    c1 = !((x2 >> 31) + 1);
    s1 = c1;
    x1 = x2 << s1;      //did all of this to get something like 0...01..1
    
    return ((s16 + s8 + s4 + s2 + s1) + (x1 >> 31 & 1)); //get the sign on the right side
}
/* howManyBits - return the minimum number of bits required to represent x in
 *             two's complement
 *  Examples: howManyBits(12) = 5
 *            howManyBits(298) = 10
 *            howManyBits(-5) = 4
 *            howManyBits(0)  = 1
 *            howManyBits(-1) = 1
 *            howManyBits(0x80000000) = 32
 *  Legal ops: ! ~ & ^ | + << >>
 *  Max ops: 90
 *  Rating: 4
 */
int howManyBits(int x) {
    int concatenate;
    int bias;
    int sign = x >> 31;  //get the sign
    x = (sign & (~x)) | (~sign & x); // either 1 or 0, union it with, 1..10 or 0..01 x off by 1 or 1/0
    concatenate = (!!(x >> 16)) << 4;
    concatenate |= (!!(x >> (concatenate + 8))) << 3; //same form as the shifting pattern in leftBitCount
    concatenate |= (!!(x >> (concatenate + 4))) << 2;
    concatenate |= (!!(x >> (concatenate + 2))) << 1;
    concatenate |= x >> (concatenate + 1);
    bias = !(x ^ 0);
    return concatenate + 2 + (~bias + 1);
}
/*
 * satAdd - adds two numbers but when positive overflow occurs, returns
 *          maximum possible value, and when negative overflow occurs,
 *          it returns minimum positive value.
 *   Examples: satAdd(0x40000000,0x40000000) = 0x7fffffff
 *             satAdd(0x80000000,0xffffffff) = 0x80000000
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 30
 *   Rating: 4
 */
int satAdd(int x, int y) {
    int overflow = ((x^y)|(x^~(x+y))) >> 31; // left side 10011^10001 = 00010 | 10011^11011 = 01000 , 01010 shift to see sign
    return (overflow & (x+y))|(~overflow&(((x+y) >> 31)^(1 << 31)));
}
/* 
 * sm2tc - Convert from sign-magnitude to two's complement
 *   where the MSB is the sign bit
 *   Example: sm2tc(0x80000005) = -5.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 15
 *   Rating: 4
 */
int sm2tc(int x) {
    int sign = x >> 31;   //get the bit sign either 0 or 1
    int firstAdd = sign & 1;  // if it is positive then 0 else 1
    int Xor = sign^x; //every spot in the original with a 1 stays, the right spot should be 0 or 1
    int temp = Xor + firstAdd; //Xor is the original or off by 1. if it is the original add 0 if it's off by 1 add 1
    int shiftback = firstAdd << 31; //put the sign bit back
    return (shiftback + temp); //now combine the sign bit for TC and append the temp or the actual number

}
/*
 * ezThreeFourths - multiplies by 3/4 rounding toward 0,
 *   Should exactly duplicate effect of C expression (x*3/4),
 *   including overflow behavior.
 *   Examples: ezThreeFourths(11) = 8
 *             ezThreeFourths(-9) = -6
 *             ezThreeFourths(1073741824) = -268435456 (overflow)
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 12
 *   Rating: 3
 */
int ezThreeFourths(int x) {
    int mask;
    int whatSign;
    int bias;
    int divByFour;
    x = ((x << 1) +x); //multiply by 3 and then use pwrof2
    mask = (1 << 2) + ~0;
    whatSign = x >> 31;
    bias = mask&whatSign;
    divByFour = ((x + bias) >> 2);
    return divByFour;
}
/* 
 * isNonNegative - return 1 if x >= 0, return 0 otherwise 
 *   Example: isNonNegative(-1) = 0.  isNonNegative(0) = 1.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 6
 *   Rating: 3
 */
int isNonNegative(int x) {
  return !(x >> 31);
}
/* 
 * replaceByte(x,n,c) - Replace byte n in x with c
 *   Bytes numbered from 0 (LSB) to 3 (MSB)
 *   Examples: replaceByte(0x12345678,1,0xab) = 0x1234ab78
 *   You can assume 0 <= n <= 3 and 0 <= c <= 255
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 10
 *   Rating: 3
 */
int replaceByte(int x, int n, int c) {
    int nbits = n << 3; //multiply by 8
    int mask = 0xFF << nbits;  //1...10..0
    int cshift = c << nbits;  // shifting c over with 0s on the right
    return (x & ~mask) | cshift;  // original & 0..01..1 union with c0..0
}
/* 
 * rotateRight - Rotate x to the right by n
 *   Can assume that 0 <= n <= 31
 *   Examples: rotateRight(0x87654321,4) = 0x76543218
 *   Legal ops: ~ & ^ | + << >>
 *   Max ops: 25
 *   Rating: 3 
 */
int rotateRight(int x, int n) {
    int mask = (~!n)+1;  //all 0's
    int leftshiftingvalue = 33+ ~n; //0...01...1
    int left = x << leftshiftingvalue; // x << 1-31 0's to the right
    int right = x >> n; // x >> 1-31, shifting to the right
    int mask1 = ~0 << leftshiftingvalue; // 11...0's leftshiftvalue
    right &= ~mask1;
    return (mask & x) | (~mask & (left|right)); //left side 0's right side, the combination of the sum
}
/* 
 * divpwr2 - Compute x/(2^n), for 0 <= n <= 30
 *  Round toward zero
 *   Examples: divpwr2(15,1) = 7, divpwr2(-33,4) = -2
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 15
 *   Rating: 2
 */
int divpwr2(int x, int n) {
    int shifted = x >> 31;
    int mask =(1 << n) + ~0;
    int bias = shifted&mask;
    return (x + bias) >> n;
}
/* 
 * allOddBits - return 1 if all odd-numbered bits in word set to 1
 *   Examples allOddBits(0xFFFFFFFD) = 0, allOddBits(0xAAAAAAAA) = 1
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 12
 *   Rating: 2
 */
int allOddBits(int x) {
    int sixteenBit = (0x55 << 8)|0x55;
    int thirdtyTwoBit = (sixteenBit << 16) | sixteenBit;
    int result = thirdtyTwoBit | x;
    return !(~result);
}
/* 
 * bitXor - x^y using only ~ and & 
 *   Example: bitXor(4, 5) = 1
 *   Legal ops: ~ &
 *   Max ops: 14
 *   Rating: 1
 */
int bitXor(int x, int y) {
    int first = ~(x&y);
    int second = ~(x&first);
    int third = ~(y&first);
    int answer = ~(second&third);
    return answer;
}
/*
 * isTmin - returns 1 if x is the minimum, two's complement number,
 *     and 0 otherwise 
 *   Legal ops: ! ~ & ^ | +
 *   Max ops: 10
 *   Rating: 1
 */
int isTmin(int x) {
   return !(x+x)&!!(x);
}
