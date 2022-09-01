#ifndef PYROBUSTPCA_VANILLA_PCA_H
#define PYROBUSTPCA_VANILLA_PCA_H

#include <stdint.h>
#include <numeric>
#include <algorithm>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "nb_convert.h"

using NBTensorMatrixXd = nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>, nb::c_contig, nb::device::cpu>;
using NBTensorVectorXd = nb::tensor<nb::numpy, double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>;

/*
    @TODO: handling the result getter calling when Fit() is not done yet
*/

class VanillaPCA
{
public:
    VanillaPCA() : num_features(0), num_data(0){};
    void Fit(NBTensorMatrixXd &data);
    NBTensorVectorXd GetMean() { return mean_nb_; };
    NBTensorVectorXd GetScores() { return scores_nb_; };
    NBTensorMatrixXd GetPrincipalComponents() { return principal_components_nb_; };
    int32_t num_features; // num_features = D
    int32_t num_data;     // num_data = N
private:
    Eigen::MatrixXd X_; // N x D data matrix
    Eigen::VectorXd mean_;
    NBTensorVectorXd mean_nb_;
    NBTensorVectorXd scores_nb_;
    NBTensorMatrixXd principal_components_nb_;
};

#endif // PYROBUSTPCA_VANILLA_PCA_H