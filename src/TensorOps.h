#ifndef __TENSOR_OPS_H
#define __TENSOR_OPS_H

#include "Tensor.h"
#include "Optimizer.h"
#include <iostream>

using namespace std;

namespace redtea {
    namespace core {
        class Variable : public Tensor {
            public :
                Variable() : Tensor() { }
                Variable(const MatrixX& mat) : Tensor() {
                    this->getOutput() = mat;
                    setRows(mat.rows());
                    setCols(mat.cols());
                }
                Variable(int row, int col) : Tensor(){
                    this->getOutput().resize(row, col);
                    setRows(row);
                    setCols(col);
                }
            public :
                Variable(const Variable& other) {
                    set(other);
                }

                typedef Variable Type;
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
                void update() {
					if(param->updated) return;
					
                    optimizer->update(this->getOutput(), this->getLoss());
					this->clearLoss();
					
					param->updated = true;
                }
            public :
                static shared_ptr<Variable> create(const MatrixX& mat) {
                    return shared_ptr<Variable>(new Variable(mat));
                }
                static shared_ptr<Variable> create(int row, int col) {
                    return shared_ptr<Variable>(new Variable(row, col));
                }
				static Variable random(int row, int col) {
					return Variable(MatrixX::Random(row,col));
				}
        };

        class Constant : public Tensor {
            public :
                Constant() : Tensor() {}
                Constant(const MatrixX& mat) : Tensor(){
                    this->getOutput() = mat;
                    setRows(mat.rows());
                    setCols(mat.cols());
                }
                Constant(int row, int col) : Tensor(){
                    this->getOutput().resize(row, col);
                    setRows(row);
                    setCols(col);
                }
            public :
                Constant(const Constant& other) {
                    set(other);
                }
 
                typedef Constant Type;
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
                static shared_ptr<Constant> create(const MatrixX& mat) {
                    return shared_ptr<Constant>(new Constant(mat));
                }
                static shared_ptr<Constant> create(int row, int col) {
                    return shared_ptr<Constant>(new Constant(row, col));
                }
				static Constant zeros(int row, int col) {
					return Constant(MatrixX::Zero(row,col));
				}
				static Constant ones(int row, int col) {
					return Constant(MatrixX::Ones(row,col));
				}
        };
        
        class Add : public Tensor {
        public :
            Add() : Tensor() {}
            Add(PTensor a, PTensor b) : Tensor() {
                assert(a->cols() == b->cols());
                inputs.push_back(a);
                inputs.push_back(b);
                setRows(a->rows());
                setCols(a->cols());
            }

            Add(const Tensor& a, const Tensor& b) {
                assert(a.cols() == b.cols());
                inputs.push_back(a.copy());
                inputs.push_back(b.copy());
                setRows(a.rows());
                setCols(a.cols());
            }
        public :
            Add(const Add& other) {
                set(other);
            }

            typedef Add Type;
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
                MatrixX& a = inputs[0]->getOutput();
                MatrixX& b = inputs[1]->getOutput();

                MatrixX& o = this->getOutput();
                if(a.rows() != b.rows()) {
                    o = a + b.replicate(a.rows(), 1);
                } else {
                    o = a + b;
                }
            }

            void backward() {
				const MatrixX& deltaLoss = getLoss();
				
                MatrixX& a = inputs[0]->getOutput();
                MatrixX& b = inputs[1]->getOutput();

                MatrixX deltaLoss0 = deltaLoss;
                MatrixX deltaLoss1;
                if(a.rows() != b.rows()) {
                    deltaLoss1 = MatrixX::Zero(b.rows(), b.cols());
                    for(int i=0;i<a.rows();i++) {
                        deltaLoss1 += deltaLoss.row(i);             
                    }
                } else {
                    deltaLoss1 = deltaLoss;
                }

                inputs[0]->addLoss(deltaLoss0);
                inputs[1]->addLoss(deltaLoss1);
				
				clearLoss();
            }

        public :
            static shared_ptr<Add> create(PTensor a, PTensor b) {
                return shared_ptr<Add>(new Add(a, b));
            }
        };

        class Mul : public Tensor {
        public :
            Mul() : Tensor() {}
            Mul(PTensor a, PTensor b) : Tensor(){
                assert(a->cols() == b->rows());
                inputs.push_back(a);
                inputs.push_back(b);
                setRows(a->rows());
                setCols(b->cols());                
            }
            Mul(const Tensor& a, const Tensor& b) : Tensor(){
                assert(a.cols() == b.rows());
                inputs.push_back(a.copy());
                inputs.push_back(b.copy());
                setRows(a.rows());
                setCols(b.cols());
            }
        public :
                Mul(const Mul& other) {
                    set(other);
                }
                
                typedef Mul Type;
                shared_ptr<Tensor> copy() const {
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
                this->getOutput() = inputs[0]->getOutput() 
                                        * inputs[1]->getOutput();
            }

            void backward() {
				const MatrixX& deltaLoss = getLoss();
				
                MatrixX deltaLoss0 = 
                           deltaLoss * inputs[1]->getOutput().transpose();
                MatrixX deltaLoss1 =
                           inputs[0]->getOutput().transpose() * deltaLoss;
                
                inputs[0]->addLoss(deltaLoss0);
                inputs[1]->addLoss(deltaLoss1);
				
				clearLoss();
            }

        public :
            static shared_ptr<Mul> create(PTensor a, PTensor b) {
                return shared_ptr<Mul>(new Mul(a, b));
            }
        };
        
		//element wise multiplication
        class MulElt : public Tensor {
        public :
            MulElt() : Tensor() {}
            MulElt(PTensor a, PTensor b) : Tensor(){
                assert(a->cols() == b->cols() 
					&& a->rows() == b->rows());
                inputs.push_back(a);
                inputs.push_back(b);
                setRows(a->rows());
                setCols(a->cols());                
            }
            MulElt(const Tensor& a, const Tensor& b) : Tensor(){
                assert(a.cols() == b.cols()
					&& a.rows() == b.rows());
                inputs.push_back(a.copy());
                inputs.push_back(b.copy());
                setRows(a.rows());
                setCols(a.cols());
            }
        public :
                MulElt(const MulElt& other) {
                    set(other);
                }
                
                typedef MulElt Type;
                shared_ptr<Tensor> copy() const {
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
                this->getOutput() = inputs[0]->getOutput().array() 
                                        * inputs[1]->getOutput().array();
            }

            void backward() {
				const MatrixX& deltaLoss = getLoss();
				
                MatrixX deltaLoss0 = 
                           deltaLoss.array() * inputs[1]->getOutput().array();
                MatrixX deltaLoss1 =
                           inputs[0]->getOutput().array() * deltaLoss.array();
                
                inputs[0]->addLoss(deltaLoss0);
                inputs[1]->addLoss(deltaLoss1);
				
				clearLoss();
            }

        public :
            static shared_ptr<MulElt> create(PTensor a, PTensor b) {
                return shared_ptr<MulElt>(new MulElt(a, b));
            }
        };
    };
};
#endif
