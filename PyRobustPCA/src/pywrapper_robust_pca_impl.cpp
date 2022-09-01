#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include "vanilla_pca.h"

namespace nb = nanobind;
using namespace nb::literals;
using NBTensorMatrixXd = nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>, nb::c_contig, nb::device::cpu>;

/* @TODO: simplify NBTensorMatrixXd?
*/

NB_MODULE(pywrapper_robust_pca_impl, m)
{
    nb::class_<VanillaPCA>(m, "VanillaPCA")
        .def(nb::init<>())
        .def("set_data", &VanillaPCA::SetData, "data"_a)
        .def_readonly("num_data", &VanillaPCA::num_data)
        .def_readonly("num_features", &VanillaPCA::num_features);
}
