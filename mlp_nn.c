#define MLP_NN_IMPLEMENTATION
#define ACTIVATION_FUNCTION sigmoidf // you set the activaiton function like this!
                                    // P.S. default is sigmoidf
#include "mlp_nn.h"
#include <time.h>
#include <stdio.h>

float td_xor[] = 
{
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
};

float td_sum[] = 
{
    0, 0,    0, 0,   0, 0,
    0, 0,    0, 1,   0, 1,
    0, 0,    1, 0,   1, 0,
    0, 0,    1, 1,   1, 1,
            
    0, 1,    0, 0,   0, 1,
    0, 1,    0, 1,   1, 0,
    0, 1,    1, 0,   1, 1,
    0, 1,    1, 1,   0, 0,
        
    1, 0,    0, 0,   1, 0,
    1, 0,    0, 1,   1, 1,
    1, 0,    1, 0,   0, 0,
    1, 0,    1, 1,   0, 1,
            
    1, 1,    0, 0,   1, 1,
    1, 1,    0, 1,   0, 0,
    1, 1,    1, 0,   0, 1,
    1, 1,    1, 1,   1, 0

};

int main(void)
{
    // check for activation funciton.
    printf("%s\n", STRINGIZE(ACTIVATION_FUNC_NAME));
    srand(time(0));
    // srand(29); // seed

    float *td = td_sum;
    
    size_t stride = 6;
    float eps = 1e-1;
    float lr = 1e-1;
    size_t epochs = 100000;


    size_t n = 16; //sizeof(td) / sizeof(td[0])/3;
    Mat ti = 
    {
        .rows = n,
        .cols = 4,
        .stride = stride,
        .es = td
    };

    Mat to = 
    {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td + 4
    };

    size_t arch[] = {4, 4, 6, 4, 2};
    NN_model nn = nn_model_alloc(ARRAY_LEN(arch), arch);
    NN_model g = nn_model_alloc(ARRAY_LEN(arch), arch);

    nn_rand(nn, -1 ,1);
    printf("no errors on initialization\n");

/*     Mat row = mat_row(ti, 1);
    MAT_PRINT(row);
    mat_copy(NN_INPUT(nn), row);
    nn_forward(nn);
    MAT_PRINT(NN_OUTPUT(nn)); */

    printf("loss: %f\n", nn_loss(nn, ti, to));


    nn_train(nn, g, ti, to, epochs, lr);

    printf("loss: %f\n", nn_loss(nn, ti, to));

    
    
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);

            printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }
    }

    //  NN_PRINT(g);

    nn_save_model(nn);

    printf("loss: %f\n", nn_loss(nn, ti, to));
    return 0;

}