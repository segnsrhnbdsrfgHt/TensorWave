#ifndef __ACTIVATION_H
#define __ACTIVATION_H

#include "Tensor.h"
#include "Optimizer.h"
#include <iostream>

using namespace std;

namespace redtea {
    namespace core {
        class ReLU : public Tensor {
        public :
            ReLU() : Tensor() {}
            ReLU(PTensor in) : Tensor() {
                inputs.push_back(in);
                setRows(in->rows());
                setCols(in->cols());
            }
            ReLU(Tensor& in) : Tensor() {
                inputs.push_back(in.copy());
                setRows(in.rows());
                setCols(in.cols());
            }
        public :
                ReLU(const ReLU& other) {
                    set(other);
                }

                typedef ReLU Type;
                shared_ptr<Tensor> copy() const{
                    shared_ptr<Tensor> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }
				
				Type& operator=(const Type& other) {
                    this->set(other);
                    return *this;
                }

        public :
            void forward() {
                Tensor::forward();

                MatrixX& in = inputs[0]->getOutput();
                MatrixX& o = this->getOutput();
                o.resize(in.rows(), in.cols());

                for(int i=0;i<in.rows();i++) {
                    for(int j=0;j<in.cols();j++) {
                        o(i, j) = in(i, j) > 0 ? in(i, j): 0; 
                    }
                }
            }

            void backward() {
				const MatrixX& deltaLoss = getLoss();
				
                MatrixX& o = this->getOutput();
                const MatrixX& l = deltaLoss;

                MatrixX deltaLossIn = MatrixX::Zero(o.rows(), o.cols());
                for(int i=0;i<o.rows();i++) {
                    for(int j=0;j<o.cols();j++) {
                        if(o(i, j) > 0) {
                            deltaLossIn(i, j) = l(i, j);
                        } else {
                            deltaLossIn(i, j) = 0;
                        }
                    }
                }
                inputs[0]->addLoss(deltaLossIn);
				clearLoss();
            }
        public :
            static shared_ptr<Type> create(PTensor in) {
                return shared_ptr<Type>(new Type(in));
            }

        };
 
        class Sigmoid : public Tensor {
        public :
            Sigmoid() : Tensor() {}
            Sigmoid(PTensor in) : Tensor() {
                inputs.push_back(in);
                setRows(in->rows());
                setCols(in->cols());
            }
            Sigmoid(const Tensor& in) : Tensor() {
                inputs.push_back(in.copy());
                setRows(in.rows());
                setCols(in.cols());
            }
        public :
                Sigmoid(const Sigmoid& other) {
                    set(other);
                }
                 
                typedef Sigmoid Type;
                shared_ptr<Tensor> copy() const{
                    shared_ptr<Tensor> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }
				Type& operator=(const Type& other) {
                    this->set(other);
                    return *this;
                }
        public :
            void forward() {
                Tensor::forward();

                MatrixX& in = inputs[0]->getOutput();
                MatrixX& o = this->getOutput();
                o.resize(in.rows(), in.cols());
     
                for(int i=0;i<in.rows();i++) {
                    for(int j=0;j<in.cols();j++) {
                        o(i, j) = 1/(1+exp(-in(i, j)));
                    }
                }
            }

            void backward() {
				const MatrixX& deltaLoss = getLoss();
				
                MatrixX& o = this->getOutput();
                const MatrixX& l = deltaLoss;

                MatrixX deltaLossIn = MatrixX::Zero(o.rows(), o.cols()); 
                for(int i=0;i<o.rows();i++) {
                    for(int j=0;j<o.cols();j++) {
                        deltaLossIn(i, j) = o(i, j) * (1.0- o(i, j));
                        deltaLossIn(i, j) *= l(i, j);
                    }
                }
                inputs[0]->addLoss(deltaLossIn);
				clearLoss();
            }

        public :
            static shared_ptr<Sigmoid> create(PTensor in) {
                return shared_ptr<Sigmoid>(new Sigmoid(in));
            }
        };

        class Softmax : public Tensor {
        private :
            MatrixX expVals;
            MatrixX expSums;
            vector<int> maxIndexes;
        public :
            Softmax() : Tensor() {}
            Softmax(PTensor in) : Tensor(){
                inputs.push_back(in);
                setRows(in->rows());
                setCols(in->cols());
            }
            Softmax(const Tensor& in) : Tensor() {
                inputs.push_back(in.copy());
                setRows(in.rows());
                setCols(in.cols());
            }
        public :
                Softmax(const Softmax& other) {
                    set(other);
                }
                
                typedef Softmax Type;
                shared_ptr<Tensor> copy() const{
                    shared_ptr<Tensor> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }
				Type& operator=(const Type& other) {
                    this->set(other);
                    return *this;
                }
        public :
            void forward() {
                Tensor::forward();

                MatrixX& in = inputs[0]->getOutput();
                MatrixX& o = this->getOutput();

                o.resize(in.rows(), in.cols());
                expVals.resize(in.rows(), in.cols());
                expSums.resize(in.rows(), 1);
                maxIndexes.clear();

                for(int i=0;i<in.rows();i++) {
                    int maxIndex = 0;
                    double max = in.row(i).maxCoeff(&maxIndex);
                    double sum= 0;
                    //set expVal and expSum for backward
                    for(int j=0;j<in.cols();j++) {
                        expVals(i, j) = exp(in(i, j) - max);
                        sum += expVals(i, j);
                    }
                    expSums(i, 0) = sum;
                    maxIndexes.push_back(maxIndex);
                    
                    //set ouput
                    for(int j=0;j<in.cols();j++) {
                        o(i, j) = expVals(i, j) / sum;
                    }
                }
            }

            void backward() {
				const MatrixX& deltaLoss = getLoss();
				
                MatrixX& o = this->getOutput();
                const MatrixX& l = deltaLoss;
                MatrixX deltaLossIn = MatrixX::Zero(o.rows(), o.cols());

                for(int i=0;i<o.rows();i++) {
                    for(int j=0;j<o.cols();j++) {
                        if(j != maxIndexes[i]) {
                            double sum = expSums(i);
                            deltaLossIn(i, j) = 1.0/sum-expVals(i, j)/(sum*sum);
                            deltaLossIn(i, j) *= expVals(i, j)*l(i, j);
                        } else {
                            deltaLossIn(i, j) = 0;
                        }
                    }
                }

                inputs[0]->addLoss(deltaLossIn);
				clearLoss();
            }
        public :
            static shared_ptr<Softmax> create(PTensor in) {
                return shared_ptr<Softmax>(new Softmax(in));
            }
        };
		
        class Tanh : public Tensor {
        public :
            Tanh() : Tensor() {}
            Tanh(PTensor in) : Tensor() {
                inputs.push_back(in);
                setRows(in->rows());
                setCols(in->cols());
            }
            Tanh(const Tensor& in) : Tensor() {
                inputs.push_back(in.copy());
                setRows(in.rows());
                setCols(in.cols());
            }
        public :
            Tanh(const Tanh& other) {
                set(other);
            }
                 
            typedef Tanh Type;
            shared_ptr<Tensor> copy() const{
                shared_ptr<Tensor> c(new Type());
                c->setParam(this->getParam());
                c->setInputs(this->getInputs());
                c->setOptimizer(this->getOptimizer());
                return c;
			}
			Type& operator=(const Type& other) {
                this->set(other);
                return *this;
            }
        public :
            void forward() {
                Tensor::forward();

                MatrixX& in = inputs[0]->getOutput();
                MatrixX& o = this->getOutput();
                o.resize(in.rows(), in.cols());
     
                for(int i=0;i<in.rows();i++) {
                    for(int j=0;j<in.cols();j++) {
                        double tmp = exp(-in(i, j)*2);
                        o(i, j) = (1.0 - tmp)/(1.0 + tmp);
                    }
                }
            }

            void backward() {
				const MatrixX& deltaLoss = getLoss();
				
                MatrixX& o = this->getOutput();
                const MatrixX& l = deltaLoss;

                MatrixX deltaLossIn = MatrixX::Zero(o.rows(), o.cols()); 
                for(int i=0;i<o.rows();i++) {
                    for(int j=0;j<o.cols();j++) {                       
						deltaLossIn(i, j) = 1.0 - o(i, j) * o(i, j);
                        deltaLossIn(i, j) *= l(i, j);
                    }
                }
                inputs[0]->addLoss(deltaLossIn);
				clearLoss();
            }

        public :
            static shared_ptr<Tanh> create(PTensor in) {
                return shared_ptr<Tanh>(new Tanh(in));
            }
        };

		
    };
};
#endif
