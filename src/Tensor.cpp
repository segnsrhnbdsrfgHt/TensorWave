#include "Tensor.h"

#include <vector>
#include <memory>

#include "def.h"
#include "Optimizer.h"
#include "TensorOps.h"

using namespace Eigen;
using namespace std;

namespace redtea{
    namespace core{

                template<class T>
                RefVector<T>::RefVector() : vector<shared_ptr<T>>() {} 
             
                template<class T>
                RefVector<T>::RefVector(const RefVector<T>& other) 
                    : vector<shared_ptr<T>>() {
                    this->assign(other.begin(), other.end());
                }

                template<class T>
                RefVector<T>& RefVector<T>::operator=(
                                  const RefVector<T>& other) {
                    this->assign(other.begin(), other.end());
                    return *this;
                }

                Tensor::Tensor() {
                    param = shared_ptr<Param>(new Param);
                }
                Tensor::Tensor(int r, int c) {
                    param = shared_ptr<Param>(new Param);
                    param->rows = r;
                    param->cols = c;
                }
                shared_ptr<Param>& Tensor::getParam() {
                    return param;
                }
                shared_ptr<Param> Tensor::getParam() const {
                    return param;
                }
                void Tensor::setParam(const shared_ptr<Param>& param) {
                    this->param = param;
                }
                RefVector<Tensor>& Tensor::getInputs(){
                    return inputs;
                }
                RefVector<Tensor> Tensor::getInputs() const {
                    return inputs;
                }
                void Tensor::setInputs(const RefVector<Tensor>& inputs) {
                    this->inputs = inputs;
                }
                shared_ptr<Optimizer> Tensor::getOptimizer() const {
                    return optimizer;
                }
                void Tensor::setOptimizer(const shared_ptr<Optimizer>& opti) {
                    this->optimizer = opti;
                }
                void Tensor::setOptimizer(const Optimizer& opti) {
                    if(optimizer) return;

                    optimizer = opti.copy();
                    for(int i=0;i<inputs.size();i++) {
                        inputs[i]->setOptimizer(opti);
                    }
                }

                Tensor::Tensor(const Tensor& other) {
                    set(other);
                }
                
                typedef Tensor Type;
                shared_ptr<Tensor> Tensor::copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }

                Tensor& Tensor::set(const Tensor& other) {
                    this->param = other.getParam();
                    this->inputs = other.getInputs();
                    this->optimizer = other.getOptimizer();
                    return *this;
                }

                

                /*
                * It will be time efficient if you call this method
                * when there are more than one collections for                 *                * a Tensor Object in the Tensor graph.
                *
                */
                void Tensor::reset() {
					if(!param->forwarded) return;
					
					for(int i=0;i<inputs.size();i++) {
						inputs[i]->reset();
					}
					
					//must be set after reset of children, because layer tensor share 'param'.
					//if before, the children tensor will directly return, param will never be updated
                    param->forwarded = false;
                    param->updated = false;

					clearLoss();
                }
                void Tensor::forward() {
                    if(param->forwarded) return;
                    
                    for(int i=0;i<inputs.size();i++) {
                        inputs[i]->forward();
                    }
                    param->forwarded = true;
                }

                void Tensor::backward() {
                    for(int i=0;i<inputs.size();i++) {
                        inputs[i]->backward();
                    }
                }

                void Tensor::update() {
                    if(param->updated) return;
                    for(int i=0;i<inputs.size();i++) {
                        inputs[i]->update();
                    }
                    param->updated = true;
                }

                void Tensor::setOutput(const MatrixX& output) {
                    param->tensorOutput = output;
                }

                MatrixX& Tensor::getOutput() {
                    return param->tensorOutput;
                }

                void Tensor::addLoss(const MatrixX& deltaLoss) {
                    MatrixX& loss = param->tensorLoss;
                    if(loss.rows() <= 0) loss = MatrixX::Zero(
                                          deltaLoss.rows(), deltaLoss.cols()); 
                    loss += deltaLoss;
                }
				
                void Tensor::clearLoss() {
                    MatrixX& loss = param->tensorLoss;
                    if(loss.rows() > 0) loss = MatrixX::Zero(
                                          loss.rows(), loss.cols()); 
                }
				
				

                MatrixX& Tensor::getLoss() {
                    return param->tensorLoss;
                }

                int Tensor::rows() const {
                    return param->rows;
                }
                void Tensor::setRows(int r) {
                    param->rows = r;
                }
                int Tensor::cols() const {
                    return param->cols;
                }
                void Tensor::setCols(int c) {
                    param->cols = c;
                }

                Tensor& Tensor::operator=(const Tensor& other) {
                    set(other);
                    return *this;
                }
                
                Add Tensor::operator+(const Tensor& other) const {
                    Add add(*this, other);
                    return add;
                }
                Mul Tensor::operator*(const Tensor& other) const {
                    Mul mul(*this, other);
                    return mul;
                }
    };
};
