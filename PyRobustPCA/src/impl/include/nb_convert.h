#ifndef PYROBUSTPCA_NB_CONVERT_HPP
#define PYROBUSTPCA_NB_CONVERT_HPP

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include <Eigen/Dense>
#include <algorithm>

namespace nb = nanobind;
using namespace nb::literals;
using NBTensorMatrixXd = nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>, nb::c_contig, nb::device::cpu>;

/* @TODO: add type and shape validation
   @TODO: use std::copy?
*/
Eigen::MatrixXd ConvertNBTensorToEigenXd(NBTensorMatrixXd &tensor);
NBTensorMatrixXd ConvertEigenXdToNBTensor(const Eigen::MatrixXd &mat);

#endif // PYROBUSTPCA_NB_CONVERT_HPP
