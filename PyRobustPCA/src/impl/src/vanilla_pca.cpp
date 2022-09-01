#include "vanilla_pca.h"
#include <iostream>

void VanillaPCA::Fit(NBTensorMatrixXd &data)
{
    X_ = ConvertNBTensorToEigenMatrixXd(data);
    num_features = X_.cols();
    num_data = X_.rows();

    /* @Add assert line when input data is not given
     */
    Eigen::MatrixXd X_centrized = X_;
    mean_ = X_.colwise().mean();
    X_centrized.rowwise() -= mean_.transpose();

    // Calculate variance-covariance matrix
    Eigen::MatrixXd S = X_centrized.transpose() * X_centrized / (num_data - 1);
    std::cout << "S" << S << std::endl;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver;
    eigen_solver.compute(S);

    Eigen::MatrixXd V = eigen_solver.eigenvectors();
    Eigen::VectorXd D = eigen_solver.eigenvalues();

    mean_nb_ = ConvertEigenVectorXdToNBTensor(mean_);
    scores_nb_ = ConvertEigenVectorXdToNBTensor(D);
    principal_components_nb_ = ConvertEigenMatrixXdToNBTensor(V);
}