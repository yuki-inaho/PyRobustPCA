#include "nb_convert.h"

NBTensorMatrixXd ConvertEigenXdToNBTensor(const Eigen::MatrixXd &mat)
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

Eigen::MatrixXd ConvertNBTensorToEigenXd(NBTensorMatrixXd &tensor)
{
    /* @TODO: validate NxD
     */
    size_t n_cols = tensor.shape(1);
    size_t n_rows = tensor.shape(0);
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(n_rows, n_cols);
    for (size_t y = 0; y < n_rows; ++y)
    {
        for (size_t x = 0; x < n_cols; ++x)
        {
            mat(y, x) = (double)tensor(y, x);
        }
    }
    return mat;
}
