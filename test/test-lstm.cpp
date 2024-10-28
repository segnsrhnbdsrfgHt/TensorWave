/*
* test case for LeastSquareLoss
*/

#include <iostream>
#include <stdlib.h>

#include "Tensor.h"
#include "TensorOps.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Activation.h"
#include "Layer.h"
#include "ShapeOps.h"
#include "data.h"

using namespace std;
using namespace Eigen;
using namespace redtea::core;

int main(int argc, char* argv[]) {

    if(argc < 2) return 1;
    int epoch = atoi(argv[1]);
    cout<<"epoch="<<epoch<<endl;

    MatrixX sample;
    MatrixX target;
    loadCsv("./test/lstm.csv", sample, target);

    Constant x(sample);
    Constant y(target);

    RefVector<Tensor> lstm_in = SubTensor::split(x);
	
    LstmLayer lstmLayer(lstm_in, 10);
	
	ConcatTensor concat(lstmLayer.getOutputTensor());
	
    DenseLayer dense(concat, 1);  
   
    LeastSquareLoss loss(dense, y);

    AdamOptimizer opti;
    //SGDOptimizer opti(1e-3);
    //AdadeltaOptimizer opti;
    //MomentumOptimizer opti(0.8, 1e-3);

    opti.minimize(loss);

    for(int i=0;i<epoch;i++) {
        opti.run();
        if(i % 10 == 0) {
            cout<<"epoch: "<<i<<", loss: "<<loss.getOutput().mean()<<endl;
        }
    }
    cout<<"output: "<<dense.getOutput()<<endl; 
    
    return 0;
}
