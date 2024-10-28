#ifndef __LAYER_H
#define __LAYER_H

#include "Tensor.h"
#include "TensorOps.h"
#include "Activation.h"

namespace redtea {
    namespace core {

        class DenseLayer : public Tensor {
            public :
                DenseLayer() : Tensor() {}
                DenseLayer(PTensor in, int outputSize) : Tensor() {
                    _init(*in, outputSize);
                }

                DenseLayer(const Tensor& in, int outputSize) : Tensor() {
                    _init(in, outputSize);
                }

                void _init(const Tensor& x, int outputSize) {
                    Variable w(MatrixX::Random(x.cols(), outputSize));
                    Variable b(MatrixX::Random(1, outputSize));
                    Add o = x * w + b;

                    //directly output to the DenseLayer class
					this->setParam(o.getParam());
                    inputs.push_back(o.copy());
                }
            public :
                DenseLayer(const DenseLayer& other) {
                    set(other);
                }

                typedef DenseLayer Type;
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
        };
		
		//output class of LstmLayer
		class LstmCell : public Tensor {
			
			struct LstmCellParam : public Param {
				shared_ptr<Tensor> C;
				shared_ptr<Tensor> H;
			};
			
			public :
                LstmCell() : Tensor() {}
                LstmCell(PTensor in, PTensor c0, PTensor h0, int outputSize, 
					PTensor w_xf, PTensor w_hf, PTensor b_f, 
					PTensor w_xi, PTensor w_hi, PTensor b_i,
					PTensor w_xc, PTensor w_hc, PTensor b_c,
					PTensor w_xo, PTensor w_ho, PTensor b_o) : Tensor() {
                    _init(*in, *c0, *h0, outputSize,
						*w_xf, *w_hf, *b_f, *w_xi, *w_hi, *b_i, 
						*w_xc, *w_hc, *b_c, *w_xo, *w_ho, *b_o);
                }

                LstmCell(const Tensor& in, const Tensor& c0, 
					const Tensor& h0, int outputSize,
					const Tensor& w_xf, const Tensor& w_hf, const Tensor& b_f, 
					const Tensor& w_xi, const Tensor& w_hi, const Tensor& b_i,
					const Tensor& w_xc, const Tensor& w_hc, const Tensor& b_c,
					const Tensor& w_xo, const Tensor& w_ho, const Tensor& b_o) : Tensor() {
                    _init(in, c0, h0, outputSize,
						w_xf, w_hf, b_f, w_xi, w_hi, b_i, 
						w_xc, w_hc, b_c, w_xo, w_ho, b_o);
                }

				//reference http://www.jianshu.com/p/9dc9f41f0b29
                void _init(const Tensor& x, const Tensor& c0, 
					const Tensor& h0, int outputSize,
					const Tensor& w_xf, const Tensor& w_hf, const Tensor& b_f, 
					const Tensor& w_xi, const Tensor& w_hi, const Tensor& b_i,
					const Tensor& w_xc, const Tensor& w_hc, const Tensor& b_c,
					const Tensor& w_xo, const Tensor& w_ho, const Tensor& b_o) {
                    
					shared_ptr<LstmCellParam> lstmCellParam(new LstmCellParam());
					param = lstmCellParam;
					
					Sigmoid f = Sigmoid(x * w_xf + h0 * w_hf + b_f);
					Sigmoid i = Sigmoid(x * w_xi + h0 * w_hi + b_i);
					Sigmoid o = Sigmoid(x * w_xo + h0 * w_ho + b_o);
					Tanh c = Tanh(x * w_xc + h0 * w_hc + b_c);
					
					Add C = MulElt(f, c0) + MulElt(i, c);
					MulElt output = MulElt(o, C);
					
					//directly output to the layer
					this->setRows(output.rows());
					this->setCols(output.cols());
					output.setParam(this->getParam());
                    inputs.push_back(output.copy());
					
					lstmCellParam->C = C.copy();
					lstmCellParam->H = output.copy();
                }
				
				shared_ptr<Tensor>& getC() {
					LstmCellParam* lstmCellParam = (LstmCellParam*)param.get();
					return lstmCellParam->C;
				}
				
				shared_ptr<Tensor>& getH() {
					LstmCellParam* lstmCellParam = (LstmCellParam*)param.get();
					return lstmCellParam->H;
				}
            public :
				typedef LstmCell Type;
                LstmCell(const Type& other) {
                    set(other);
                }
           
                shared_ptr<Tensor> copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }
				
				Type& operator=(const Type& other) {
                    this->set(other);
                    return *this;
                }
		};
		
		class LstmLayer : public Tensor{
			
			struct LstmParam : public Param {
				shared_ptr<Tensor> w_xf, w_hf, b_f;
				shared_ptr<Tensor> w_xi, w_hi, b_i;
				shared_ptr<Tensor> w_xc, w_hc, b_c;
				shared_ptr<Tensor> w_xo, w_ho, b_o;
				shared_ptr<Tensor> h0, c0;
				RefVector<Tensor> outputs;
			};
				
			public :
                LstmLayer(){
					param = shared_ptr<LstmParam>(new LstmParam());
				}
				
                LstmLayer(const RefVector<Tensor> ins, int outputSize) {
					assert(ins.size() > 0);
					
					shared_ptr<LstmParam> lstmParam(new LstmParam());
					param = lstmParam;
					
					
					int inputRow = ins[0]->rows();
					int inputCol = ins[0]->cols();
					
					lstmParam->w_xf = Variable::random(inputCol, outputSize).copy();
					lstmParam->w_hf = Variable::random(outputSize, outputSize).copy();
					lstmParam->b_f = Variable::random(inputRow, outputSize).copy();
					lstmParam->w_xi = Variable::random(inputCol, outputSize).copy();
					lstmParam->w_hi = Variable::random(outputSize, outputSize).copy();
					lstmParam->b_i = Variable::random(inputRow, outputSize).copy();
					lstmParam->w_xc = Variable::random(inputCol, outputSize).copy();
					lstmParam->w_hc = Variable::random(outputSize, outputSize).copy();
					lstmParam->b_c = Variable::random(inputRow, outputSize).copy();
					lstmParam->w_xo = Variable::random(inputCol, outputSize).copy();
					lstmParam->w_ho = Variable::random(outputSize, outputSize).copy();
					lstmParam->b_o = Variable::random(inputRow, outputSize).copy();
					
					lstmParam->h0 = Constant::zeros(inputRow, outputSize).copy();
					lstmParam->c0 = Constant::zeros(inputRow, outputSize).copy();
					
					shared_ptr<LstmCell>  cell;
					for(int i=0;i<ins.size();i++) {
						if(i == 0) {
							cell = shared_ptr<LstmCell>(
								new LstmCell(ins[i], lstmParam->c0, lstmParam->h0, outputSize, 
											 lstmParam->w_xf, lstmParam->w_hf, lstmParam->b_f, 
											 lstmParam->w_xi, lstmParam->w_hi, lstmParam->b_i,
											 lstmParam->w_xc, lstmParam->w_hc, lstmParam->b_c,
											 lstmParam->w_xo, lstmParam->w_ho, lstmParam->b_o) );
							lstmParam->outputs.push_back(cell);
						}else {
							LstmCell* lastCell = (LstmCell*)cell.get();
							cell = shared_ptr<LstmCell>(
								new LstmCell(ins[i], lastCell->getC(), lastCell->getH(), outputSize, 
											 lstmParam->w_xf, lstmParam->w_hf, lstmParam->b_f, 
											 lstmParam->w_xi, lstmParam->w_hi, lstmParam->b_i,
											 lstmParam->w_xc, lstmParam->w_hc, lstmParam->b_c,
											 lstmParam->w_xo, lstmParam->w_ho, lstmParam->b_o) );
							/*
							//set the last cell as the layer output
							if(i == ins.size()-1) {
								this->setRows(cell->rows());
								this->setCols(cell->cols());
								cell->setParam(this->param);
							}				 
							*/
							lstmParam->outputs.push_back(cell);
						}
					}
					
                    inputs.push_back(cell);
                }
				
			public :
				
				RefVector<Tensor>& getOutputTensor() {
					LstmParam* lstmParam = (LstmParam*)param.get();
					return lstmParam->outputs;
				}
				
			public :
				typedef LstmLayer Type;
                LstmLayer(const Type& other) {
                    set(other);
                }
           
                shared_ptr<Tensor> copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }
				
				Type& operator=(const Type& other) {
                    this->set(other);
                    return *this;
                }
		};

    };
};

#endif
