#ifndef _PRINTFLOAT_H_
#define _PRINTFLOAT_H_

#define PFLOAT(A) *(int*)&A

int power(int base, int exp);

void reverse(char *str, int len);

int intToStr(int x, char str[], int d);

void ftoa(float n, char *res, int afterpoint);

#endif
