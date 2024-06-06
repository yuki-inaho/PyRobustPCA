#include "vanilla_pca.h"
#include "robust_pca_ogk.h"
#include "robust_pca_detmcd.h"
#include "stat_utils.h"

#include <boost/math/distributions/chi_squared.hpp>

namespace nb = nanobind;
using namespace nb::literals;

double CalculateMedianNB(NBVectorXd &data)
{
    return CalculateMedian(ConvertNBArrayToEigenVectorXd(data));
};

double CalculateMedianAbsoluteDeviationNB(NBVectorXd &data)
{
    return CalculateMedianAbsoluteDeviation(ConvertNBArrayToEigenVectorXd(data));
};

NBVectorXd CalculateMahalanobisDistanceNB(NBMatrixXd &data, NBVectorXd &mu, NBMatrixXd &sigma_mat)
{
    Eigen::MatrixXd data_eigenxd = ConvertNBArrayToEigenMatrixXd(data);
    Eigen::VectorXd mu_eigenxd = ConvertNBArrayToEigenVectorXd(mu);
    Eigen::MatrixXd sigma_mat_eigenxd = ConvertNBArrayToEigenMatrixXd(sigma_mat);
    return ConvertEigenVectorXdToNBArray(CalculateMahalanobisDistance(data_eigenxd, mu_eigenxd, sigma_mat_eigenxd));
};

NBMatrixXd GenerateCorrelationMatrixNB(NBMatrixXd &data, double const_weighted_mean, double const_winsored_mean)
{
    return ConvertEigenMatrixXdToNBArray(GenerateCorrelationMatrix(ConvertNBArrayToEigenMatrixXd(data), const_weighted_mean, const_winsored_mean));
};

nb::list CovarianceOGKNB(NBMatrixXd &data, double const_weighted_mean, double const_winsored_mean)
{
    nb::list results;
    Eigen::VectorXd location;
    Eigen::MatrixXd covariance;
    bool status = CovarianceOGK(ConvertNBArrayToEigenMatrixXd(data), location, covariance, const_weighted_mean, const_winsored_mean);
    std::cout << "test:" << int(status) << std::endl;
    if (!status)
    {
        return results;
    }
    results.append(ConvertEigenVectorXdToNBArray(location));
    results.append(ConvertEigenMatrixXdToNBArray(covariance));
    return results;
}

/* @TODO: simplify NBMatrixXd?
 */
NB_MODULE(pywrapper_robust_pca_impl, m)
{
    nb::class_<VanillaPCA>(m, "VanillaPCA")
        .def(nb::init<>())
        .def("fit", &VanillaPCA::Fit, "data"_a)
        .def("get_mean", &VanillaPCA::GetMean)
        .def("get_scores", &VanillaPCA::GetScores)
        .def("get_principal_components", &VanillaPCA::GetPrincipalComponents)
        .def_ro("num_data", &VanillaPCA::num_data)
        .def_ro("num_features", &VanillaPCA::num_features);
    nb::class_<RobustPCAOGK>(m, "RobustPCAOGK")
        .def(nb::init<>())
        .def("fit", &RobustPCAOGK::Fit, "data"_a, "const_weighted_mean"_a = 4.5, "const_winsored_mean"_a = 3.0)
        .def("get_mean", &RobustPCAOGK::GetMean)
        .def("get_scores", &RobustPCAOGK::GetScores)
        .def("get_principal_components", &RobustPCAOGK::GetPrincipalComponents)
        .def_ro("num_data", &RobustPCAOGK::num_data)
        .def_ro("num_features", &RobustPCAOGK::num_features);
    nb::class_<RobustPCADetMCD>(m, "RobustPCADetMCD")
        .def(nb::init<>())
        .def("fit", &RobustPCADetMCD::Fit, "data"_a, "n_iter"_a = 3, "const_weighted_mean"_a = 4.5, "const_winsored_mean"_a = 3.0)
        .def("get_mean", &RobustPCADetMCD::GetMean)
        .def("get_scores", &RobustPCADetMCD::GetScores)
        .def("get_principal_components", &RobustPCADetMCD::GetPrincipalComponents)
        .def_ro("num_data", &RobustPCADetMCD::num_data)
        .def_ro("num_features", &RobustPCADetMCD::num_features);
    m.def("median", &CalculateMedianNB, "data"_a);
    m.def("mad", &CalculateMedianAbsoluteDeviationNB, "data"_a);
    m.def("mahalanobis_distance", &CalculateMahalanobisDistanceNB, "data"_a, "mu"_a, "sigma_mat"_a);
    m.def("generate_correlation_matrix", &GenerateCorrelationMatrixNB, "data"_a, "const_weighted_mean"_a = 4.5, "const_winsored_mean"_a = 3.0);
    m.def("covariance_ogk", &CovarianceOGKNB, "data"_a, "const_weighted_mean"_a = 4.5, "const_winsored_mean"_a = 3.0);
}
