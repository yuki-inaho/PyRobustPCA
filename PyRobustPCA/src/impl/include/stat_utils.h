#ifndef PYROBUSTPCA_NB_STAT_UTILS_H
#define PYROBUSTPCA_NB_STAT_UTILS_H

#include <cmath>
#include <numeric>
#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>
#include "nb_convert.h"

double CalculateMedian(const Eigen::VectorXd &data);
double CalculateMedianAbsoluteDeviation(const Eigen::VectorXd &data, double scale_const = 1.482602218505602);
Eigen::VectorXd CalculateBisquareWeights(const Eigen::VectorXd &vec, double const_weighted_mean = 4.5);
double WeightedMean(const Eigen::VectorXd &vec, const Eigen::VectorXd &weights);
double WinsoredSquaredMean(const Eigen::VectorXd &vec, double const_winsored);
double CalculateRobustMean(const Eigen::VectorXd &vec, double const_weighted_mean = 4.5);
double CalculateRobustScale(const Eigen::VectorXd &vec, const double &robust_mean, double const_winsored_mean = 3.0);
Eigen::MatrixXd GenerateCorrelationMatrix(const Eigen::MatrixXd &data_mat, double const_weighted_mean = 4.5, double const_winsored_mean = 3.0);
Eigen::VectorXd CalculateMahalanobisDistance(const Eigen::MatrixXd &data, const Eigen::VectorXd &mu, const Eigen::MatrixXd &sigma_mat);
double ChiSquaredQuantile(double v, double p);
bool CovarianceOGK(const Eigen::MatrixXd &data_mat, Eigen::VectorXd &location, Eigen::MatrixXd &covariance, double const_weighted_mean = 4.5, double const_winsored_mean = 3.0);

#endif // PYROBUSTPCA_NB_STAT_UTILS_H