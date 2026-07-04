

inputPlaceholder

conv1/weightsConst
%
conv1Conv2Dinputconv1/weights


conv1/reluReluconv1

pool1MaxPool
conv1/relu


fc/weightsConst
&
	fc/matmulMatMulpool1
fc/weights

fc/biasConst
%
logitsBiasAdd	fc/matmulfc/bias

outputSoftmaxlogits