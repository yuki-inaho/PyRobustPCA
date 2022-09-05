#ifndef PYROBUSTPCA_NB_CONVERT_HPP
#define PYROBUSTPCA_NB_CONVERT_HPP

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include <Eigen/Dense>
#include <algorithm>

namespace nb = nanobind;
using namespace nb::literals;

using NBTensorMatrixXd = nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>, nb::c_contig, nb::device::cpu>;
using NBTensorVectorXd = nb::tensor<nb::numpy, double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>;


/* @TODO: add type and shape validation
   @TODO: use std::copy?
*/
NBTensorMatrixXd ConvertEigenMatrixXdToNBTensor(const Eigen::MatrixXd &mat);
NBTensorVectorXd ConvertEigenVectorXdToNBTensor(const Eigen::VectorXd &vec);
Eigen::MatrixXd ConvertNBTensorToEigenMatrixXd(NBTensorMatrixXd &tensor_mat);
Eigen::VectorXd ConvertNBTensorToEigenVectorXd(NBTensorVectorXd &tensor_vec);

#endif // PYROBUSTPCA_NB_CONVERT_HPP
