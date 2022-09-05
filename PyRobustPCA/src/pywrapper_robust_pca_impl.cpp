#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include "vanilla_pca.h"
#include "stat_utils.h"

#include <boost/math/distributions/chi_squared.hpp>

namespace nb = nanobind;
using namespace nb::literals;

using NBTensorMatrixXd = nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>, nb::c_contig, nb::device::cpu>;
using NBTensorVectorXd = nb::tensor<nb::numpy, double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>;

double CalculateMedianNB(NBTensorVectorXd &data)
{
    return CalculateMedian(ConvertNBTensorToEigenVectorXd(data));
};

double CalculateMedianAbsoluteDeviationNB(NBTensorVectorXd &data)
{
    return CalculateMedianAbsoluteDeviation(ConvertNBTensorToEigenVectorXd(data));
};

NBTensorVectorXd CalculateMahalanobisDistanceNB(NBTensorMatrixXd &data, NBTensorVectorXd &mu, NBTensorMatrixXd &sigma_mat)
{
    Eigen::MatrixXd data_eigenxd = ConvertNBTensorToEigenMatrixXd(data);
    Eigen::VectorXd mu_eigenxd = ConvertNBTensorToEigenVectorXd(mu);
    Eigen::MatrixXd sigma_mat_eigenxd = ConvertNBTensorToEigenMatrixXd(sigma_mat);
    return ConvertEigenVectorXdToNBTensor(CalculateMahalanobisDistance(data_eigenxd, mu_eigenxd, sigma_mat_eigenxd));
};

NBTensorMatrixXd GenerateCorrelationMatrixNB(NBTensorMatrixXd &data, double const_weighted_mean, double const_winsored_mean)
{
    return ConvertEigenMatrixXdToNBTensor(GenerateCorrelationMatrix(ConvertNBTensorToEigenMatrixXd(data), const_weighted_mean, const_winsored_mean));
};

/* @TODO: simplify NBTensorMatrixXd?
 */
NB_MODULE(pywrapper_robust_pca_impl, m)
{
    nb::class_<VanillaPCA>(m, "VanillaPCA")
        .def(nb::init<>())
        .def("fit", &VanillaPCA::Fit, "data"_a)
        .def("get_mean", &VanillaPCA::GetMean)
        .def("get_scores", &VanillaPCA::GetScores)
        .def("get_principal_components", &VanillaPCA::GetPrincipalComponents)
        .def_readonly("num_data", &VanillaPCA::num_data)
        .def_readonly("num_features", &VanillaPCA::num_features);
    m.def("median", &CalculateMedianNB, "data"_a);
    m.def("mad", &CalculateMedianAbsoluteDeviationNB, "data"_a);
    m.def("mahalanobis_distance", &CalculateMahalanobisDistanceNB, "data"_a, "mu"_a, "sigma_mat"_a);
    m.def("generate_correlation_matrix", &GenerateCorrelationMatrixNB, "data"_a, "const_weighted_mean"_a = 4.5, "const_winsored_mean"_a = 3.0);
}
