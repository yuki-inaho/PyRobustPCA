#ifndef PYROBUSTPCA_VANILLA_PCA_H
#define PYROBUSTPCA_VANILLA_PCA_H

#include <stdint.h>
#include "nb_convert.h"

using NBTensorMatrixXd = nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>, nb::c_contig, nb::device::cpu>;

/*
    Temporally
*/
class VanillaPCA
{
public:
    VanillaPCA() : num_features(0), num_data(0){};
    void SetData(NBTensorMatrixXd &data);
    int32_t num_features; // num_features = D
    int32_t num_data;     // num_data = N
private:
    Eigen::MatrixXd X_; // N x D data matrix
};

#endif // PYROBUSTPCA_VANILLA_PCA_H