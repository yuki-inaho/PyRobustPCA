#ifndef PYROBUSTPCA_ROBUST_PCA_OGK_H
#define PYROBUSTPCA_ROBUST_PCA_OGK_H

#include <stdint.h>
#include <numeric>
#include <algorithm>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "stat_utils.h"
#include "nb_convert.h"

/*
    @TODO: handling the result getter calling when Fit() is not done yet
*/

class RobustPCAOGK
{
public:
    RobustPCAOGK() : num_features(0), num_data(0){};
    bool Fit(NBMatrixXd &data, double const_weighted_mean, double const_winsored_mean);
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

#endif // PYROBUSTPCA_ROBUST_PCA_OGK_H