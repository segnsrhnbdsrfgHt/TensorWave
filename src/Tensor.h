#ifndef __TENSOR_H
#define __TENSOR_H

#include <vector>
#include <memory>

#include "def.h"

using namespace Eigen;
using namespace std;

namespace redtea{
    namespace core{
        class Add;
        class Mul;
        class Optimizer;
        struct Param {
                bool forwarded;
                bool updated;
                int rows;
                int cols;
                MatrixX tensorLoss;
                MatrixX tensorOutput;

                Param() {
                    forwarded = false;
                    updated = false;
                    rows = 0;
                    cols = 0;
                }
        };

        template<class T>
        class RefVector : public vector<shared_ptr<T>>
        {
            public :
                RefVector();
                RefVector(const RefVector<T>& other);
                RefVector& operator=(const RefVector<T>& other);
        };

        class Tensor {
            protected :
                /*
                * used in forward process to avoid duplicated forking of forward                * method
                */
                shared_ptr<Param> param;
                //input tensors, for back propergation
                RefVector<Tensor> inputs;
                
                //optimizer for updating parameters
                shared_ptr<Optimizer> optimizer;
            public :
                Tensor(); 
                Tensor(int r, int c);
                Tensor& set(const Tensor& other);
                shared_ptr<Param>& getParam();
                shared_ptr<Param> getParam() const;
                void setParam(const shared_ptr<Param>& param);
                RefVector<Tensor>& getInputs();
                RefVector<Tensor> getInputs() const;
                void setInputs(const RefVector<Tensor>& inputs);
                shared_ptr<Optimizer> getOptimizer() const;
                void setOptimizer(const shared_ptr<Optimizer>& opti);

            public :
                Tensor(const Tensor& other); 
                virtual shared_ptr<Tensor> copy() const; 
                /*
                  this method is different from setOptimizer(const shared_ptr<>
                  in the way that it will recursively set its input node's
                  optimizer, while the former one only sets its own property
                */
                void setOptimizer(const Optimizer& opti);

            public :
                /*
                * It will be time efficient if you call this method 
                * when there are more than one collections for                 
				* a Tensor Object in the Tensor graph. 
                * 
                */
                virtual void reset();
                virtual void forward(); 
                virtual void backward();
                virtual void update();

            public :
                void setOutput(const MatrixX& output);
                MatrixX& getOutput();
                void addLoss(const MatrixX& deltaLoss); 
                MatrixX& getLoss();
				void clearLoss();
            public :
                int rows() const;
                void setRows(int r);
                int cols() const; 
                void setCols(int c);
            public :
                Tensor& operator=(const Tensor& other);
                Add operator+(const Tensor& other) const ; 
                Mul operator*(const Tensor& other) const ;
        };

        typedef shared_ptr<Tensor> PTensor;
    };
};


#endif
