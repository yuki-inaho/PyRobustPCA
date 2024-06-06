#ifndef PYROBUSTPCA_VANILLA_PCA_H
#define PYROBUSTPCA_VANILLA_PCA_H

#include <stdint.h>
#include <numeric>
#include <algorithm>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "nb_convert.h"

/*
    @TODO: handling the result getter calling when Fit() is not done yet
*/

class VanillaPCA
{
public:
    VanillaPCA() : num_features(0), num_data(0){};
    void Fit(NBMatrixXd &data);
    NBVectorXd GetMean() { return mean_nb_; };
    NBVectorXd GetScores() { return scores_nb_; };
    NBMatrixXd GetPrincipalComponents() { return principal_components_nb_; };
    int32_t num_features; // num_features = D
    int32_t num_data;     // num_data = N
private:
    Eigen::MatrixXd X_; // N x D data matrix
    Eigen::VectorXd mean_;
    NBVectorXd mean_nb_;
    NBVectorXd scores_nb_;
    NBMatrixXd principal_components_nb_;
};

#endif // PYROBUSTPCA_VANILLA_PCA_H