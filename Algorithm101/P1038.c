#include <stdio.h>

int f(int input)
{
    return input / 10 + input % 10;
}

void main()
{
    int NumberA = 27;
    int NumberB = 33;
    int Sum = 0;
    for (int i = NumberA; i++; i <= NumberB)
    {
        Sum = Sum + f(i);
    }
}