#include "nb_convert.h"
#include <iostream>

NBMatrixXd ConvertEigenMatrixXdToNBArray(const Eigen::MatrixXd &mat)
{
    size_t n_cols = mat.cols();
    size_t n_rows = mat.rows();
    size_t shape[2] = {n_rows, n_cols};

    double *data = new double[n_cols * n_rows]{0};
    nb::capsule deleter(data, [](void *data) noexcept
                        { delete[] (double *)data; });
    NBMatrixXd arr(data, 2, shape, deleter);
    for (size_t y = 0; y < n_rows; ++y)
    {
        for (size_t x = 0; x < n_cols; ++x)
        {
            arr(y, x) = mat(y, x);
        }
    }
    return arr;
}

NBVectorXd ConvertEigenVectorXdToNBArray(const Eigen::VectorXd &vec)
{
    size_t n_elems = vec.size();
    size_t shape[1] = {n_elems};
    double *data = new double[n_elems]{0};
    nb::capsule deleter(data, [](void *data) noexcept
                        { delete[] (double *)data; });
    NBVectorXd arr(data, 1, shape, deleter);
    for (size_t i = 0; i < n_elems; ++i)
    {
        arr(i) = vec(i);
    }
    return arr;
}

Eigen::MatrixXd ConvertNBArrayToEigenMatrixXd(NBMatrixXd &arr_mat)
{
    size_t n_cols = arr_mat.shape(1);
    size_t n_rows = arr_mat.shape(0);
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(n_rows, n_cols);
    for (size_t y = 0; y < n_rows; ++y)
    {
        for (size_t x = 0; x < n_cols; ++x)
        {
            mat(y, x) = (double)arr_mat(y, x);
        }
    }
    return mat;
}

Eigen::VectorXd ConvertNBArrayToEigenVectorXd(NBVectorXd &arr_vec)
{
    Eigen::VectorXd vec(arr_vec.shape(0));
    for (size_t i = 0; i < arr_vec.shape(0); i++)
    {
        vec(i) = arr_vec(i);
    }
    return vec;
}