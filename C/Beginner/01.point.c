#include <stdio.h>

void update(int *i)
{
    for (int j = 0; j < 5; j++)
    {
        i[j] = scanf("%d", i);
        printf("%d\n", i[j]);
    }
}

int main()
{
    int i[5] = {1, 2, 3, 4, 5};
    update(i);
    return 0;
}