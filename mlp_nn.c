#define MLP_NN_IMPLEMENTATION
#include "mlp_nn.h"
#include <time.h>
#include <stdio.h>

int main(void)
{
    srand(time(0));

    float raw_data[4] = {1, 0, 0, 1};

    Mat a = mat_alloc(1, 2);
    mat_rand(a, 0, 1);
    Mat b = {.rows = 2, .cols = 2, .es = raw_data};
    

    Mat d = mat_alloc(1, 2);
    mat_dot(d, a, b);

    mat_print(d);

    return 0;

}