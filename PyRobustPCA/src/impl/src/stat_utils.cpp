#include "stat_utils.h"
#include <iostream>

double CalculateMedian(const Eigen::VectorXd &data)
{
    Eigen::VectorXd vec = data;
    std::sort(vec.data(), vec.data() + vec.size());
    int data_size = vec.size();
    if (data_size % 2 == 0)
    {
        // vec.size() is even
        return (vec(data_size / 2 - 1) + vec(data_size / 2)) / 2;
    }
    else
    {
        // vec.size() is odd
        return vec(data_size / 2);
    }
};

double CalculateMedianAbsoluteDeviation(const Eigen::VectorXd &data, double scale_const)
{
    // scale_const = 1 / scipy.stats.Gaussian.ppf(3 / 4.0)
    double data_median = CalculateMedian(data);
    return scale_const * CalculateMedian((data.array() - data_median).abs());
};

Eigen::VectorXd CalculateBisquareWeights(const Eigen::VectorXd &vec, double const_weighted_mean)
{
    Eigen::VectorXd weights(vec.size());
    std::transform(vec.data(), vec.data() + vec.size(), weights.data(), [=](double x)
                   { return std::pow(1 - std::pow(x / const_weighted_mean, 2), 2) * int(std::abs(x) <= const_weighted_mean); });
    return weights;
};

double WeightedMean(const Eigen::VectorXd &vec, const Eigen::VectorXd &weights)
{
    return (double)(weights.transpose() * vec) / weights.sum();
};

double WinsoredSquaredMean(const Eigen::VectorXd &vec, double const_winsored_mean)
{
    Eigen::VectorXd vec_transformed(vec.size());
    std::transform(vec.data(), vec.data() + vec.size(), vec_transformed.data(), [=](double x)
                   { return std::min(std::pow(x, 2), std::pow(const_winsored_mean, 2)); });
    return vec_transformed.mean();
};

double CalculateRobustMean(const Eigen::VectorXd &vec, double const_weighted_mean)
{
    double med = CalculateMedian(vec);
    double mad = CalculateMedianAbsoluteDeviation(vec);
    Eigen::VectorXd weights = CalculateBisquareWeights((vec.array() - med) / mad, const_weighted_mean);
    return WeightedMean(vec, weights);
}

double CalculateRobustScale(const Eigen::VectorXd &vec, const double &robust_mean, double const_winsored_mean)
{
    /* using tau-estimator
     */
    double mad = CalculateMedianAbsoluteDeviation(vec);
    Eigen::VectorXd vec_scaled = (vec.array() - robust_mean) / mad;
    return std::sqrt(mad * mad * WinsoredSquaredMean(vec_scaled, const_winsored_mean));
};

Eigen::MatrixXd GenerateCorrelationMatrix(const Eigen::MatrixXd &data_mat, double const_weighted_mean, double const_winsored_mean)
{
    size_t n_features = data_mat.cols();
    Eigen::MatrixXd correlation_mat = Eigen::MatrixXd::Zero(n_features, n_features);

    for (size_t j = 0; j < n_features; j++)
    {
        for (size_t i = 0; i < n_features; i++)
        {
            if (i == j)
            {
                correlation_mat(i, i) = 1.0;
            }
            else if (i > j)
            {
                Eigen::VectorXd vec_ipj = data_mat.col(i).array() + data_mat.col(j).array();
                Eigen::VectorXd vec_imj = data_mat.col(i).array() - data_mat.col(j).array();
                double m_ipj = CalculateRobustMean(vec_ipj, const_weighted_mean);
                double m_imj = CalculateRobustMean(vec_imj, const_weighted_mean);
                double s_ipj = CalculateRobustScale(vec_ipj, m_ipj, const_winsored_mean);
                double s_imj = CalculateRobustScale(vec_imj, m_imj, const_winsored_mean);
                double cij = (s_ipj * s_ipj - s_imj * s_imj) / 4;
                correlation_mat(i, j) = cij;
            }
            else
            {
                correlation_mat(i, j) = correlation_mat(j, i);
            }
        }
    }
    return correlation_mat;
};

Eigen::VectorXd CalculateMahalanobisDistance(const Eigen::MatrixXd &data, const Eigen::VectorXd &mu, const Eigen::MatrixXd &sigma_mat)
{
    // @TODO: add shape validation (assert x.ndim == 2)
    size_t data_size = data.rows();
    Eigen::MatrixXd sigma_mat_ = sigma_mat;
    Eigen::MatrixXd data_centrized = data.rowwise() - mu.transpose();
    Eigen::MatrixXd sigma_mat_inv = sigma_mat_.inverse();
    Eigen::VectorXd distance_vec(data_size);
    for (size_t i = 0; i < data_size; i++)
    {
        distance_vec(i) = std::sqrt(data_centrized.row(i) * sigma_mat_inv * data_centrized.row(i).transpose());
    }
    return distance_vec;
};

double ChiSquaredQuantile(double v, double p)
{
    return quantile(boost::math::chi_squared(v), p);
}

bool CovarianceOGK(const Eigen::MatrixXd &data_mat, Eigen::VectorXd &location, Eigen::MatrixXd &covariance, double const_weighted_mean, double const_winsored_mean)
{
    size_t n_observation = data_mat.rows();
    size_t n_features = data_mat.cols();
    Eigen::MatrixXd covariance_mat = Eigen::MatrixXd::Zero(n_features, n_features);

    // Generate scale matrix
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_features, n_features);
    for (size_t i = 0; i < n_features; i++)
    {
        Eigen::VectorXd vec_i = data_mat.col(i);
        double m_i = CalculateRobustMean(vec_i, const_weighted_mean);
        double s_i = CalculateRobustScale(vec_i, m_i, const_winsored_mean);
        D(i, i) = s_i;
    }
    Eigen::MatrixXd D_inv = D.inverse();

    // Generate scaled data matrix
    Eigen::MatrixXd Z = data_mat * D_inv;

    // Generate correlation matrix
    Eigen::MatrixXd U = GenerateCorrelationMatrix(Z, const_weighted_mean, const_winsored_mean);

    // Compute eigen vector of the correlation matrix
    // @TODO: return status?
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver;
    eigen_solver.compute(U);
    Eigen::VectorXd eig_values = eigen_solver.eigenvalues();
    Eigen::MatrixXd eig_vectors = eigen_solver.eigenvectors();
    if (eigen_solver.info() != Eigen::Success)
    {
        return false;
    }

    // @TODO: separate below lines as a function?
    // Sort eigen vector
    std::vector<size_t> indices_sorted(n_features);
    std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
    std::stable_sort(indices_sorted.begin(), indices_sorted.end(),
                     [&eig_values](size_t i1, size_t i2)
                     { return eig_values(i1) > eig_values(i2); });
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(n_features, n_features);
    for (size_t i = 0; i < n_features; i++)
    {
        E.col(i) = eig_vectors.col(indices_sorted[i]);
    }

    // Generate the projected data matrix and the squared scale matrix
    Eigen::MatrixXd V = Z * E;
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n_features, n_features);
    for (size_t i = 0; i < n_features; i++)
    {
        Eigen::VectorXd vec_i = V.col(i);
        double m_i = CalculateRobustMean(vec_i, const_weighted_mean);
        double s_i = CalculateRobustScale(vec_i, m_i, const_winsored_mean);
        L(i, i) = s_i * s_i;
    }

    // Calculate
    Eigen::VectorXd m_vec(n_features);
    for (size_t i = 0; i < n_features; i++)
    {
        m_vec(i) = CalculateRobustMean(V.col(i).array(), const_weighted_mean); // @TODO: need array?
    }

    Eigen::VectorXd mu_z_hat = E * m_vec;
    Eigen::MatrixXd sigma_z_hat = E * L * E.transpose();

    Eigen::VectorXd mu_rawogk = D * mu_z_hat;
    Eigen::MatrixXd sigma_rawogk = D * sigma_z_hat * D.transpose();

    // Calculate mahalanobis distances and extract inlier data
    Eigen::VectorXd mahalanobis_distances = CalculateMahalanobisDistance(data_mat, mu_rawogk, sigma_rawogk);
    double mahalanobis_distance_squared_median = CalculateMedian(mahalanobis_distances.array().pow(2));
    double chi2_ppf_90 = ChiSquaredQuantile((double)n_features, 0.90);
    double chi2_ppf_50 = ChiSquaredQuantile((double)n_features, 0.5);
    double cutoff_threshold = mahalanobis_distance_squared_median * chi2_ppf_90 / chi2_ppf_50;

    std::vector<size_t> indices_obs(n_observation);
    std::vector<size_t> indices_inlier;
    for (size_t i = 0; i < n_observation; i++)
    {
        if (std::pow(mahalanobis_distances(i), 2.0) <= cutoff_threshold)
            indices_inlier.push_back(i);
    }

    Eigen::MatrixXd data_mat_inlier = Eigen::MatrixXd::Zero(indices_inlier.size(), n_features);
    int inlier_index = 0;
    for (auto it = indices_inlier.begin(); it != indices_inlier.end(); ++it, ++inlier_index)
    {
        data_mat_inlier.row(inlier_index) = data_mat.row(*it);
    }

    // Compute location and covariance matrix
    location = data_mat_inlier.colwise().mean();
    Eigen::MatrixXd data_mat_inlier_centrized = data_mat_inlier;
    data_mat_inlier_centrized.rowwise() -= location.transpose();
    covariance = (data_mat_inlier_centrized.transpose() * data_mat_inlier_centrized).array() / (data_mat_inlier_centrized.rows() - 1);

    return true;
};
