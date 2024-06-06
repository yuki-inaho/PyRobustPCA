#include "robust_pca_ogk.h"

bool RobustPCAOGK::Fit(NBMatrixXd &data, double const_weighted_mean, double const_winsored_mean)
{
    /* @Add assert line for shape validation
     */
    X_ = ConvertNBArrayToEigenMatrixXd(data);
    num_features = X_.cols();
    num_data = X_.rows();

    // @TODO: implement covariance matrix getter?
    Eigen::MatrixXd covariance;
    if (!CovarianceOGK(X_, mean_, covariance, const_weighted_mean, const_winsored_mean))
    {
        return false;
    }

    // Calculate variance-covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver;
    eigen_solver.compute(covariance);

    Eigen::MatrixXd eig_vectors = eigen_solver.eigenvectors();
    Eigen::VectorXd eig_values = eigen_solver.eigenvalues();

    // Sort eigen vector
    std::vector<size_t> indices_sorted(num_features);
    std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
    std::stable_sort(indices_sorted.begin(), indices_sorted.end(),
                     [&eig_values](size_t i1, size_t i2)
                     { return eig_values(i1) > eig_values(i2); });

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(num_features, num_features);
    Eigen::VectorXd D(num_features);
    for (size_t i = 0; i < num_features; i++)
    {
        D(i) = eig_values(indices_sorted[i]);
        V.col(i) = eig_vectors.col(indices_sorted[i]);
    }

    mean_nb_ = ConvertEigenVectorXdToNBArray(mean_);
    scores_nb_ = ConvertEigenVectorXdToNBArray(D);
    principal_components_nb_ = ConvertEigenMatrixXdToNBArray(V);
    return true;
}