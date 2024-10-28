#ifndef __SHAPE_OPS_H
#define __SHAPE_OPS_H


namespace redtea {
    namespace core {

		class SubTensor : public Tensor {
			
			struct SubTensorParam : public Param {
				int  index;
				bool row;
			};
			
            public :
                SubTensor() : Tensor() { }
                SubTensor(const Tensor& in, int index, bool row=true) {
					_init(in, index, row);
                }
				SubTensor(PTensor in, int index, bool row=true) {
					_init(*in, index, row);
				}
				void _init(const Tensor& in, int index, bool row=true) {
					shared_ptr<SubTensorParam> subTensorParam(new SubTensorParam());
					param = subTensorParam;
					
					subTensorParam->index = index;
					subTensorParam->row = row;
					
					if(row) {
						setRows(1);
						setCols(in.cols());
					}else {
						setRows(in.rows());
						setCols(1);
					}
					inputs.push_back(in.copy());
				}
            public :
                typedef SubTensor Type;
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
					
					SubTensorParam* subTensorParam = (SubTensorParam*)param.get();
					MatrixX& o = inputs[0]->getOutput();
					if(subTensorParam->row) {
						this->getOutput() = o.row(subTensorParam->index); 
					} else {
						this->getOutput() = o.col(subTensorParam->index); 
					}
				}
				
				void backward() {
					const MatrixX& deltaLoss = getLoss();
					MatrixX deltaLoss0 = MatrixX::Zero(inputs[0]->rows(), inputs[0]->cols());
					SubTensorParam* subTensorParam = (SubTensorParam*)param.get();
					if(subTensorParam->row) {
						deltaLoss0.row(subTensorParam->index) = deltaLoss;
					}else {
						deltaLoss0.col(subTensorParam->index) = deltaLoss;
					}
					inputs[0]->addLoss(deltaLoss0);
					clearLoss();
				}

			static RefVector<Tensor> split(const Tensor& tensor, bool row = true) {
					RefVector<Tensor> subTensors;
					int count = 0;
					if(row) {
						count = tensor.rows();
					}else {
						count = tensor.cols();
					}
					
					for(int i=0;i<count;i++) {
							subTensors.push_back(
								shared_ptr<Tensor>(
									new SubTensor(tensor, i, row)));
					}
					
					return subTensors;
				}
        };
		
		
		class ConcatTensor : public Tensor {
			
			struct ConcatTensorParam : public Param {
				bool row;
			};
			
            public :
                ConcatTensor() : Tensor() { }
                ConcatTensor(RefVector<Tensor>& subTensors, bool row=true) {
					_init(subTensors, row);
                }
				void _init(RefVector<Tensor>& subTensors, bool row=true) {					
					assert(subTensors.size() > 0);
					
					shared_ptr<ConcatTensorParam> concatTensorParam(new ConcatTensorParam());
					param = concatTensorParam;
					
					concatTensorParam->row = row;
					
					if(row) {
						setRows(subTensors.size());
						setCols(subTensors[0]->cols());
					}else {
						setRows(subTensors[0]->rows());
						setCols(subTensors.size());
					}
					
					inputs = subTensors;
				}
            public :
                typedef ConcatTensor Type;
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
					
					ConcatTensorParam* concatTensorParam = (ConcatTensorParam*)param.get();
					
					MatrixX& o = this->getOutput();
					o.resize(rows(), cols());
					for(int i=0;i<inputs.size();i++) {
						if(concatTensorParam->row) {
							o.row(i) = inputs[i]->getOutput();
						}else {
							o.col(i) = inputs[i]->getOutput();
						}
					}
				}
				
				void backward() {
					const MatrixX& deltaLoss = getLoss();
					ConcatTensorParam* concatTensorParam = (ConcatTensorParam*)param.get();
					
					for(int i=0;i<inputs.size();i++) {
						if(concatTensorParam->row) {
							inputs[i]->addLoss(deltaLoss.row(i));
						}else {
							inputs[i]->addLoss(deltaLoss.col(i));
						}
					}
					
					clearLoss();
				}
        };

	};
};
#endif
