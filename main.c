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

int main(void)
{
    
    size_t arch[] = {2, 3, 2, 1};

    NN_model nn = nn_model_alloc(ARRAY_LEN(arch), arch);

    nn_load_model("model.nn", nn);


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

    return 0;
}
