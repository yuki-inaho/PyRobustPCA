#include "robust_pca_detmcd.h"

bool RobustPCADetMCD::Fit(NBMatrixXd &data, int n_iter, double const_weighted_mean, double const_winsored_mean)
{
    /* @Add assert line for shape validation
     */
    X_ = ConvertNBArrayToEigenMatrixXd(data);
    num_features = X_.cols();
    num_data = X_.rows();

    // @TODO: implement covariance matrix getter?
    Eigen::MatrixXd covariance_ogk;
    if (!CovarianceOGK(X_, mean_, covariance_ogk, const_weighted_mean, const_winsored_mean))
    {
        return false;
    }

    // Run DetMCD process
    // @TODO: separate below lines as function
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver;
    Eigen::MatrixXd covariance_k = covariance_ogk;
    Eigen::VectorXd location_k = mean_;

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(num_features, num_features);
    for (size_t i = 0; i < num_features; i++)
    {
        Eigen::VectorXd vec_i = X_.col(i);
        double m_i = CalculateRobustMean(vec_i, const_weighted_mean);
        double s_i = CalculateRobustScale(vec_i, m_i, const_winsored_mean);
        D(i, i) = s_i;
    }
    Eigen::MatrixXd D_inv = D.inverse();
    Eigen::MatrixXd Z = X_ * D_inv;
    for(int k=0; k< n_iter; k++){
        eigen_solver.compute(covariance_k);
        Eigen::MatrixXd E = eigen_solver.eigenvectors();
        Eigen::MatrixXd V = Z * E;
        Eigen::MatrixXd L = Eigen::MatrixXd::Zero(num_features, num_features);
        Eigen::MatrixXd L_half = Eigen::MatrixXd::Zero(num_features, num_features);
        Eigen::MatrixXd L_half_inv = Eigen::MatrixXd::Zero(num_features, num_features);
        for (size_t i = 0; i < num_features; i++)
        {
            Eigen::VectorXd vec_i = V.col(i);
            double m_i = CalculateRobustMean(vec_i, const_weighted_mean);
            double s_i = CalculateRobustScale(vec_i, m_i, const_winsored_mean);
            L(i, i) = s_i * s_i;
            L_half(i, i) = s_i;
            L_half_inv(i, i) = 1.0 / s_i;
        }
        Eigen::MatrixXd sigma_mat_z_k = E * L * E.transpose();
        Eigen::MatrixXd sigma_mat_z_half_k = E * L_half * E.transpose();
        Eigen::MatrixXd sigma_mat_z_half_inv_k = E * L_half_inv * E.transpose();

        Eigen::MatrixXd z_transformed = Z * sigma_mat_z_half_inv_k;
        Eigen::VectorXd median_z_transformed(num_features);
        for(int d=0; d<num_features; d++){
            median_z_transformed(d) = CalculateMedian(z_transformed.col(d));
        }
        Eigen::VectorXd mu_k = median_z_transformed.transpose() * sigma_mat_z_half_k;
        Eigen::VectorXd mahalanobis_distances_zk = CalculateMahalanobisDistance(Z, mu_k, sigma_mat_z_k);
        double mahalanobis_distances_zk_median = CalculateMedian(mahalanobis_distances_zk);

        std::vector<size_t> indices_inlier;
        for (size_t i = 0; i < num_data; i++)
        {
            if (mahalanobis_distances_zk(i) <= mahalanobis_distances_zk_median)
                indices_inlier.push_back(i);
        }
        Eigen::MatrixXd data_mat_inlier_k = Eigen::MatrixXd::Zero(indices_inlier.size(), num_features);
        int inlier_index = 0;
        for (auto it = indices_inlier.begin(); it != indices_inlier.end(); ++it, ++inlier_index)
        {
            data_mat_inlier_k.row(inlier_index) = X_.row(*it);
        }
        location_k = data_mat_inlier_k.colwise().mean();
        Eigen::MatrixXd data_mat_inlier_centrized_k = data_mat_inlier_k;
        data_mat_inlier_centrized_k.rowwise() -= location_k.transpose();
        covariance_k = (data_mat_inlier_centrized_k.transpose() * data_mat_inlier_centrized_k) / (data_mat_inlier_centrized_k.rows() - 1);
    }

    // Calculate variance-covariance matrix
    eigen_solver.compute(covariance_k);
    Eigen::MatrixXd eig_vectors = eigen_solver.eigenvectors();
    Eigen::VectorXd eig_values = eigen_solver.eigenvalues();

    // Sort eigen vector
    std::vector<size_t> indices_sorted(num_features);
    std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
    std::stable_sort(indices_sorted.begin(), indices_sorted.end(),
                     [&eig_values](size_t i1, size_t i2)
                     { return eig_values(i1) > eig_values(i2); });

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(num_features, num_features);
    Eigen::VectorXd scores(num_features);
    for (size_t i = 0; i < num_features; i++)
    {
        scores(i) = eig_values(indices_sorted[i]);
        V.col(i) = eig_vectors.col(indices_sorted[i]);
    }

    mean_nb_ = ConvertEigenVectorXdToNBArray(location_k);
    scores_nb_ = ConvertEigenVectorXdToNBArray(scores);
    principal_components_nb_ = ConvertEigenMatrixXdToNBArray(V);
    return true;
}