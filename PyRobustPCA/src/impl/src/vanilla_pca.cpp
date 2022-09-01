#include "vanilla_pca.h"

using NBTensorMatrixXd = nb::tensor<nb::numpy, double, nb::shape<nb::any, nb::any>, nb::c_contig, nb::device::cpu>;

void VanillaPCA::SetData(NBTensorMatrixXd &data)
{
    X_ = ConvertNBTensorToEigenXd(data);
    num_features = X_.cols();
    num_data = X_.rows();
}
