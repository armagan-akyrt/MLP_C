#ifndef MLP_NN_H_
#define MLP_NN_H_

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef MLP_NN_ASSERT
#include <assert.h>
#define MLP_NN_ASSERT assert
#endif //MLP_NN_ASSERT

#ifndef MLP_NN_MALLOC
#define MLP_NN_MALLOC malloc
#endif //MLP_NN_MALLOC

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])
#define MAT_PRINT(m) mat_print(m, #m, 0)
#define NN_PRINT(m) nn_print(m, #m)
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

typedef struct 
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
}Mat;

typedef struct 
{
        size_t count;
        Mat *ws;
        Mat *bs;
        Mat *as; // # of activations: count + 1;
}NN_model;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

typedef float (*activation_function) (float);

// set activation funciton, default is sifmoidf
#ifndef ACTIVATION_FUNCTION
#define ACTIVATION_FUNCTION sigmoidf
#endif // ACTIVATION_FUNCTION

// this abomination of a code will set the name for activation funciton derivative, I apologize if anyone had to see this.
#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define D_ACTIVATION_FUNCTION_HELPER(x) CONCATENATE(x, _der)
#define D_ACTIVATION_FUNCTION D_ACTIVATION_FUNCTION_HELPER(ACTIVATION_FUNCTION)

#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)
#define ACTIVATION_FUNC_NAME STRINGIZE(D_ACTIVATION_FUNCTION)

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, float val);
void mat_rand(Mat m, float l, float h);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat b);
void mat_print(Mat m, char *name, size_t padding);
void mat_activation(Mat m, activation_function func);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);

NN_model nn_model_alloc(size_t arch_count, size_t *arch);
void nn_print(NN_model m, const char *name);
void nn_rand(NN_model nn, float l, float h);
void nn_forward(NN_model nn);
float nn_loss(NN_model nn, Mat ti, Mat to);
void nn_finite_diff(NN_model nn, NN_model g, float eps, Mat ti, Mat to);
void nn_learn(NN_model nn, NN_model g, float lr);

float sigmoidf(float inp);
float ReLUf(float inp);
float rand_float(void);




#endif //MLP_NN_H_

// #define MLP_NN_IMPLEMENTATION

#ifdef MLP_NN_IMPLEMENTATION

/**
 * @brief activation funciton: sigmoid 
 * 
 * @param inp input value to be activated
 * @return result
 */
float sigmoidf(float inp)
{
        return 1.f/(1.f + expf(-inp));

}

float ReLUf(float inp)
{
    return inp > 0 ? inp : 0;
}

void mat_activation(Mat m, activation_function func)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = func(MAT_AT(m, i ,j));
        }   
    }
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat)
                {
                    .rows = 1,
                    .cols = m.cols,
                    .stride = m.stride,
                    .es = &MAT_AT(m, row, 0)
                };
                
}

void mat_copy(Mat dst, Mat src)
{
    MLP_NN_ASSERT(dst.rows == src.rows);
    MLP_NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
        
    }
    
}

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX; // return a random number between 0-1
}

/**
 * @brief allocate memory for the matrix
 * 
 * @param rows # of rows for matrix to be created
 * @param cols # of columns for matrix to be created
 * @return Mat created matrix.
 */
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = MLP_NN_MALLOC(sizeof(*m.es)*rows*cols);
    MLP_NN_ASSERT(m.es != NULL);

    return m;
}

/**
 * @brief dot product of two matricies. Result is written to the dst matrix
 * 
 * @param dst destination matrix
 * @param a first matrix
 * @param b second matrix
 */
void mat_dot(Mat dst, Mat a, Mat b)
{   
    // dot product pre-requisites.
    MLP_NN_ASSERT(a.cols == b.rows); 
    size_t n = a.cols;
    MLP_NN_ASSERT(dst.rows == a.rows);
    MLP_NN_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < n; k++)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i ,k) * MAT_AT(b, k ,j); 
            }
            
        }
        
    }
    
}

/**
 * @brief fills the matrix with the desired number
 * 
 * @param m matrix to be filled
 * @param val value to be filled in matrix
 */
void mat_fill(Mat m, float val) 
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = val;
        }
        
    }
}

/**
 * @brief sum of two matricies, result is written to dst matrix
 * 
 * @param dst destination
 * @param b second matrix
 */
void mat_sum(Mat dst, Mat b)
{
    MLP_NN_ASSERT(dst.rows == b.rows);
    MLP_NN_ASSERT(dst.cols == b.cols);
    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) += MAT_AT(b, i, j);
        }
        
    }
    
    (void) dst;
    (void) b;
}

/**
 * @brief prints out the provided matrix.
 * 
 * @param m matrix
 */
void mat_print(Mat m, char *name, size_t padding)
{
    printf("%*s%s = [\n", (int)padding, " ", name);
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("%*s    ", (int)padding, " ");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f ",MAT_AT(m, i, j));
        }
        printf("\n");
        
    }
    printf("%*s]\n",(int)padding, " ");

}

/**
 * @brief this function randomly assigns values to a provided matrix.
 * 
 * @param m matrix
 * @param l lower bound of random number
 * @param h upper bound of random number
 */
void mat_rand(Mat m, float l, float h)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = rand_float() * (h - l) + l;
        }
        
    }
}

NN_model nn_model_alloc(size_t arch_count, size_t *arch)
{
    MLP_NN_ASSERT(arch_count > 0);

    NN_model m;
    m.count = arch_count - 1;
    m.ws = MLP_NN_MALLOC(sizeof(*m.ws) * m.count); 
    MLP_NN_ASSERT(m.ws != NULL);

    m.bs = MLP_NN_MALLOC(sizeof(*m.bs) * m.count);
    MLP_NN_ASSERT(m.bs != NULL);

    m.as = MLP_NN_MALLOC(sizeof(*m.as) * m.count + 1);
    MLP_NN_ASSERT(m.as != NULL);

    m.as[0] = mat_alloc(1, arch[0]);

    for (size_t i = 1; i < arch_count; i++)
    {
        m.ws[i-1] = mat_alloc(m.as[i-1].cols, arch[i]);
        m.bs[i-1] = mat_alloc(1, arch[i]);
        m.as[i] = mat_alloc(1, arch[i]);
    }

    return m;
}

void nn_print(NN_model m, const char *name)
{
    char buf[256];
    printf("%s = [\n", name);

    for (size_t i = 0; i < m.count; i++)
    {
        snprintf(buf, sizeof(buf), "ws%zu", i); 
        mat_print(m.ws[i], buf, 4);
        
        snprintf(buf, sizeof(buf), "bs%zu", i); 
        mat_print(m.bs[i], buf, 4);

    }
    
    printf("]\n");

}

void nn_rand(NN_model nn, float l, float h)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_rand(nn.ws[i], l, h);
        mat_rand(nn.bs[i], l, h);
    }
}

float nn_loss(NN_model nn, Mat ti, Mat to)
{
    MLP_NN_ASSERT(ti.rows == to.rows);
    MLP_NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;
    float l = 0;

    for (size_t i = 0; i < n; i++)
    {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++)
        {
            float diff = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            l += diff * diff;            
        }
    }
    return l / n;
}

void nn_forward(NN_model nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i+1], nn.bs[i]);
        mat_activation(nn.as[i+1], ACTIVATION_FUNCTION);
    }
}

void nn_finite_diff(NN_model nn, NN_model g, float eps, Mat ti, Mat to)
{
    float saved;
    float l = nn_loss(nn, ti, to);

    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[j].cols; k++)
            {
                saved = MAT_AT(nn.ws[i], j ,k);
                MAT_AT(nn.ws[i], j ,k) += eps;

                MAT_AT(g.ws[i], j ,k) = (nn_loss(nn, ti, to) - l) / eps;
                MAT_AT(nn.ws[i], j ,k) = saved;
            }
        }
    }

    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[j].cols; k++)
            {
                saved = MAT_AT(nn.bs[i], j ,k);
                MAT_AT(nn.bs[i], j ,k) += eps;

                MAT_AT(g.bs[i], j ,k) = (nn_loss(nn, ti, to) - l) / eps;
                MAT_AT(nn.bs[i], j ,k) = saved;
            }
        }
    }
}

void nn_learn(NN_model nn, NN_model g, float lr)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[j].cols; k++)
            {
                MAT_AT(nn.ws[i], j ,k) -= lr * MAT_AT(g.ws[i], j ,k);
            }
        }
    }

    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[j].cols; k++)
            {
                MAT_AT(nn.bs[i], j ,k) -= lr * MAT_AT(g.bs[i], j ,k);
            }
        }
    }
}

#endif //MLP_NN_IMPLEMENTATION
 