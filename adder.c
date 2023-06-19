#define MLP_NN_IMPLEMENTATION
#include "mlp_nn.h"

#define BITS 2

int main(void)
{

    size_t n = (1<<BITS);

    printf("n: %zu\n", n);

    size_t rows = n * n;
    Mat ti = mat_alloc(rows, 2 * BITS);
    Mat to = mat_alloc(rows, BITS + 1);

    for (size_t i = 0; i < ti.rows; i++)
    {
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x + y;
        size_t overflow = z >= n;


        for (size_t j = 0; j < BITS; j++)
        {
            MAT_AT(ti, i, j)        = (x>>j)&1;
            MAT_AT(ti, i, j + BITS) = (y>>j)&1;    

            if (overflow)
            {
                MAT_AT(to, i, j)        = 0;
            
            }
            else
            {
                
                MAT_AT(to, i, j)        = (z>>j)&1;
            }
        }    
        MAT_AT(to, i, BITS) = overflow;

    }
    MAT_PRINT(ti);
    MAT_PRINT(to);
    
    size_t arch[] = {2* BITS, BITS + 1};
    printf("arch: %zu, %zu\n", arch[0], arch[1]);
    NN_model nn = nn_model_alloc(ARRAY_LEN(arch), arch);
    NN_model g = nn_model_alloc(ARRAY_LEN(arch), arch);

    nn_rand(nn, 0, 1);
    NN_PRINT(nn);

    printf("LOSS: %f", nn_loss(nn, ti, to));

    return 0;   
}