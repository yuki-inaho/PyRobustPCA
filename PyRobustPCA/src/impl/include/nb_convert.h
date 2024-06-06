#ifndef PYROBUSTPCA_NB_CONVERT_HPP
#define PYROBUSTPCA_NB_CONVERT_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <Eigen/Dense>
#include <algorithm>

namespace nb = nanobind;
using namespace nb::literals;

using NBMatrixXd = nb::ndarray<nb::numpy, double, nb::shape<nb::any, nb::any>, nb::c_contig, nb::device::cpu>;
using NBVectorXd = nb::ndarray<nb::numpy, double, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>;

/* @TODO: add type and shape validation
   @TODO: use std::copy?
*/
NBMatrixXd ConvertEigenMatrixXdToNBArray(const Eigen::MatrixXd &mat);
NBVectorXd ConvertEigenVectorXdToNBArray(const Eigen::VectorXd &vec);
Eigen::MatrixXd ConvertNBArrayToEigenMatrixXd(NBMatrixXd &tensor_mat);
Eigen::VectorXd ConvertNBArrayToEigenVectorXd(NBVectorXd &tensor_vec);

#endif // PYROBUSTPCA_NB_CONVERT_HPP
