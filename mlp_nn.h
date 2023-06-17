#ifndef MLP_NN_H_
#define MLP_NN_H_

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef MLP_NN_ASSERT
#include <assert.h>
#define MLP_NN_ASSERT assert
#endif //MLP_NN_ASSERT


#ifndef MLP_NN_MALLOC
#define MLP_NN_MALLOC malloc
#endif //MLP_NN_MALLOC

typedef struct 
{
    size_t rows;
    size_t cols;
    float *es;
}Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).cols + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, float val);
void mat_rand(Mat m, float l, float h);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat b);
void mat_print(Mat m);

float rand_float(void);


#endif //MLP_NN_H_

// #define MLP_NN_IMPLEMENTATION

#ifdef MLP_NN_IMPLEMENTATION

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
void mat_print(Mat m)
{

    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f ",MAT_AT(m, i, j));
        }
        printf("\n");
        
    }
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

#endif //MLP_NN_IMPLEMENTATION
 