/*
* test case for LeastSquareLoss
*/

#include <iostream>
#include <stdlib.h>

#include "Tensor.h"
#include "TensorOps.h"
#include "Loss.h"
#include "Optimizer.h"


using namespace std;
using namespace Eigen;
using namespace redtea::core;

int main(int argc, char* argv[]) {

    if(argc < 2) return 1;
    int epoch = atoi(argv[1]);
    cout<<"epoch="<<epoch<<endl;

    Matrix<type, 5, 2> sample;
    sample << 1, 1,
              2, 1,
              3, 2,
              5, 3,
              6, 0;

    Matrix<type, 5, 2> target;
    target << 6, 7.95, 
              8, 9.4,
              13, 15.95,
              20, 23.95,
              13, 10.1;
/*
    auto x = Constant::create(sample);
    auto y = Constant::create(target);

    auto w = Variable::create(MatrixX::Random(2, 2));
    auto b = Variable::create(MatrixX::Random(1, 2));
    
    auto mul = Mul::create(x, w);
    auto add = Add::create(mul, b);

    LeastSquareLoss loss(add, y);


    cout<<"w: "<<w->getOutput()<<endl;
    cout<<"b: "<<b->getOutput()<<endl;

    Optimizer opti(1e-2);
    for(int i=0;i<epoch;i++) {
        loss.forward();
        loss.backward(opti);

        cout<<"o: "<<add->getOutput();
        cout<<", w: "<<w->getOutput();
        cout<<", b: "<<b->getOutput();
        cout<<", l: "<<loss.getTotalLoss()<<endl;
    }
*/

    Constant x(sample);
    Constant y(target);
  
    Variable w(MatrixX::Random(2, 2));
    Variable b(MatrixX::Random(1, 2));

//    Mul mul = x * w;
    Add add = x * w + b;

    LeastSquareLoss loss(add, y);


    cout<<"w: "<<w.getOutput()<<endl;
    cout<<"b: "<<b.getOutput()<<endl;

    //AdamOptimizer opti;
    //MomentumOptimizer opti(0.8, 1e-3);
    //AdadeltaOptimizer opti;
    SGDOptimizer opti(1e-3);
    opti.minimize(loss);

    for(int i=0;i<epoch;i++) {
        opti.run();
 
        cout<<"o: "<<add.getOutput();
        cout<<", w: "<<w.getOutput();
        cout<<", b: "<<b.getOutput();
        cout<<", l: "<<loss.getOutput().sum()<<endl;
    }

    return 0;
}
