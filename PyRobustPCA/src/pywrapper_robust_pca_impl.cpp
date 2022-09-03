#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include "vanilla_pca.h"
#include "stat_utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using NBTensorMatrixXd = nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>, nb::c_contig, nb::device::cpu>;
using NBTensorVectorXd = nb::tensor<nb::numpy, double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>;


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
    m.def("calculate_median", &CalculateMedianNB, "data"_a);
    m.def("calculate_mad", &CalculateMedianAbsoluteDeviationNB, "data"_a);
}
