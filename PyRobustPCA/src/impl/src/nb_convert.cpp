#include "nb_convert.h"
#include <iostream>

NBTensorMatrixXd ConvertEigenMatrixXdToNBTensor(const Eigen::MatrixXd &mat)
{
    size_t n_cols = mat.cols();
    size_t n_rows = mat.rows();
    size_t shape[2] = {n_rows, n_cols};

    double *data = new double[n_cols * n_rows]{0};
    nb::capsule deleter(data, [](void *data) noexcept
                        { delete[](double *) data; });
    NBTensorMatrixXd tensor(data, 2, shape, deleter);
    for (size_t y = 0; y < n_rows; ++y)
    {
        for (size_t x = 0; x < n_cols; ++x)
        {
            tensor(y, x) = mat(y, x);
        }
    }
    return tensor;
}

NBTensorVectorXd ConvertEigenVectorXdToNBTensor(const Eigen::VectorXd &vec)
{
    size_t n_elems = vec.size();
    size_t shape[1] = {n_elems};
    double *data = new double[n_elems]{0};
    nb::capsule deleter(data, [](void *data) noexcept
                        { delete[](double *) data; });
    NBTensorVectorXd tensor(data, 1, shape, deleter);
    for (size_t i = 0; i < n_elems; ++i)
    {
        tensor(i) = vec(i);
    }
    return tensor;
}

Eigen::MatrixXd ConvertNBTensorToEigenMatrixXd(NBTensorMatrixXd &tensor_mat)
{
    /* @TODO: validate NXd
     */
    size_t n_cols = tensor_mat.shape(1);
    size_t n_rows = tensor_mat.shape(0);
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(n_rows, n_cols);
    for (size_t y = 0; y < n_rows; ++y)
    {
        for (size_t x = 0; x < n_cols; ++x)
        {
            mat(y, x) = (double)tensor_mat(y, x);
        }
    }
    return mat;
}


Eigen::VectorXd ConvertNBTensorToEigenVectorXd(NBTensorVectorXd &tensor_vec)
{
    /* @TODO: use std::copy & tensor.stride
     */
    Eigen::VectorXd vec(tensor_vec.shape(0));
    for (size_t i = 0; i < tensor_vec.shape(0); i++)
    {
        vec(i) = tensor_vec(i);
    }
    return vec;
}