#ifndef __OPTIMIZER_H
#define __OPTIMIZER_H

#include <iostream>
#include <list>
#include <set>
#include "def.h"
#include "Tensor.h"

using namespace std;
using namespace Eigen;

namespace redtea{
    namespace core{
        class Optimizer {
            protected :
                shared_ptr<Tensor> loss;
            public :
                void minimize(shared_ptr<Tensor> l) {
                    l->setOptimizer(*this);
                    loss = l;
                }
                void minimize(Tensor& l) {
                    l.setOptimizer(*this);
                    loss = l.copy();
                }
				
				void backward() {
					set<long> tensorSets; 
					RefVector<Tensor> tensors;
					RefVector<Tensor> tempTensors;
					
					tensors.push_back(loss);
					
					while(tensors.size() > 0) {
						tensorSets.clear();
						tempTensors.clear();
						for(int i=0;i<tensors.size();i++) {
							tensors[i]->backward();
							
							RefVector<Tensor>& inputs = tensors[i]->getInputs();
							for(int j=0;j<inputs.size();j++) {
								long tensorPointer = (long) inputs[j]->getParam().get();
								set<long>::iterator iter;
								if((iter = tensorSets.find(tensorPointer)) == tensorSets.end()) {
									tensorSets.insert(tensorPointer);
									tempTensors.push_back(inputs[j]);
								}
							}
						}
						tensors = tempTensors;
					}
				}
				
                void run() {
                    if(!loss) {
                        cerr<<"Error: no loss to minimize!"<<endl;
						return;
                    }
					
					//ready for tensor network 
					loss->reset();
					
					//forward procedure
                    loss->forward();
                    
					//initialize the loss of last layer
					MatrixX& o = loss->getOutput();
                    const MatrixX& ones = MatrixX::Ones(o.rows(), o.cols());
					loss->addLoss(ones);
					
					
					backward();
					
                    loss->update();
                }

                virtual shared_ptr<Optimizer> copy() const = 0;
                virtual void update(MatrixX& param, MatrixX& loss) = 0;
        };

        class SGDOptimizer : public Optimizer{
            protected :
                double learningRate;
                shared_ptr<Tensor> loss;
            public :
                SGDOptimizer() {
                    learningRate = 1e-3;
                }
                SGDOptimizer(double learningRate) {
                    this->learningRate = learningRate;
                }

                double setLearningRate(double l) {
                    learningRate = l;
                }
                double getLearningRate() const{
                    return learningRate;
                }

                typedef SGDOptimizer Type;
                shared_ptr<Optimizer> copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setLearningRate(this->getLearningRate());
                    return c;
                }

                void update(MatrixX& param, MatrixX& loss) {
                    param -= loss * learningRate;
                }
        };

        class MomentumOptimizer : public Optimizer {
            protected :
                double rho;
                double learningRate;
                MatrixX delta;

            public :
                MomentumOptimizer() {
                    rho = 0.95;
                    learningRate = 1e-3;
                }
                MomentumOptimizer(double r, double l) {
                    rho = r;
                    learningRate = l;
                }
 
                double getRho() const{
                    return rho;
                }

                double getLearningRate() const{
                    return learningRate;
                }

                typedef MomentumOptimizer Type;
                shared_ptr<Optimizer> copy() const{
                    shared_ptr<Type> c(
                        new Type(
                            this->getRho(), this->getLearningRate() ));
                    return c;
                }
 
                void update(MatrixX& param, MatrixX& loss) {
                    if(delta.rows() <= 0) 
                        delta = MatrixX::Zero(param.rows(), param.cols());
                    
                    delta = rho * delta -learningRate * loss;
                    param += delta;
                }
        };

        class AdadeltaOptimizer : public Optimizer {
            protected :
                double rho;
                double epsilon;
                MatrixX egs;
                MatrixX exs;

            public :
                AdadeltaOptimizer() {
                    rho = 0.95;
                    epsilon = 1e-8;
                }
                AdadeltaOptimizer(double r, double e) {
                    rho = r;
                    epsilon = e;
                }

                double getRho() const{
                    return rho;
                }

                double getEpsilon()  const {
                    return epsilon;
                }

                typedef AdadeltaOptimizer Type;
                shared_ptr<Optimizer> copy() const{
                    shared_ptr<Type> c(
                        new Type(
                            this->getRho(), this->getEpsilon() ));
                    return c;
                }

                void update(MatrixX& param, MatrixX& loss) {
                    if(egs.rows() <= 0) {
                        egs = MatrixX::Zero(param.rows(), param.cols());
                    }
                    if(exs.rows() <= 0) {
                        exs = MatrixX::Zero(param.rows(), param.cols());
                    }

                    MatrixX temp = loss.array().square();
                    egs = rho * egs + (1.0-rho)*temp;

                    double learningRate = 0.0;
                    MatrixX delta(param.rows(), param.cols());
                    for(int i=0;i<param.rows();i++) {
                        for(int j=0;j<param.cols();j++) {
                            learningRate = sqrt(exs(i, j)+epsilon) 
                                               / sqrt(egs(i, j)+epsilon);
                            //cout<<"learningRate: "<<learningRate<<endl;
                            delta(i, j) = learningRate * loss(i, j);
                        }
                    }

                    temp = delta.array().square();
                    exs = rho * exs + (1.0-rho) * temp;
                    
                    param -= delta;
                }
        };

        class AdamOptimizer : public Optimizer {
            protected :
                int iteration;
                double alpha;
                double beta1;
                double beta2;
                double epsilon;
                MatrixX m;
                MatrixX n;

            public :
                AdamOptimizer() {
                    iteration = 0;
                    alpha = 1e-3;
                    beta1 = 0.9;
                    beta2 = 0.999;
                    epsilon = 1e-8;
                }
                AdamOptimizer(double a, double b1, double b2, double e) {
                    alpha = a;
                    beta1 = b1;
                    beta2 = b2;
                    epsilon = e;
                }

                double getAlpha() const {
                    return alpha;
                }
 
                double getBeta1() const {
                    return beta1;
                }

                double getBeta2() const {
                    return beta2;
                }

                double getEpsilon()  const {
                    return epsilon;
                }

                typedef AdamOptimizer Type;
                shared_ptr<Optimizer> copy() const{
                    shared_ptr<Type> c(
                        new Type(
                            this->getAlpha(), this->getBeta1(), 
                            this->getBeta2(), this->getEpsilon() ));
                    return c;
                }

                void update(MatrixX& param, MatrixX& loss) {
                    if(m.rows() <= 0) {
                        m = MatrixX::Zero(param.rows(), param.cols());
                    }
                    if(n.rows() <= 0) {
                        n = MatrixX::Zero(param.rows(), param.cols());
                    }

                    iteration ++;

                    MatrixX temp=loss.array().square();;
                    m = beta1 * m + (1.0 - beta1) * loss;
                    n = beta2 * n + (1.0 - beta2) * temp;
           
                    MatrixX tempM = m / (1.0 - pow(beta1, iteration));
                    MatrixX tempN = n / (1.0 - pow(beta2, iteration));

                    temp = tempN.array().sqrt()+epsilon;
                    temp = tempM.array() / temp.array();
                    
 
                    param -= alpha * temp;
                }
        };
    };
};


#endif
