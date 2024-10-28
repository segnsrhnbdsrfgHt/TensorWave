cmake_minimum_required(VERSION 3.5)

#project redtea

PROJECT(readtea)


SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++11)

AUX_SOURCE_DIRECTORY(src DIR_SRCS)  
  
#head file path  
INCLUDE_DIRECTORIES(  
    ../eigen/
    src/
)  

#source directory  
AUX_SOURCE_DIRECTORY(src DIR_SRCS)  

#add executable file  
ADD_EXECUTABLE(test-lsl test/test-lsl.cpp ${DIR_SRCS})  
ADD_EXECUTABLE(test-log test/test-log.cpp ${DIR_SRCS})
ADD_EXECUTABLE(test-softmax test/test-softmax.cpp ${DIR_SRCS})
ADD_EXECUTABLE(test-layer test/test-layer.cpp ${DIR_SRCS})
ADD_EXECUTABLE(test-tanh test/test-tanh.cpp ${DIR_SRCS})
ADD_EXECUTABLE(test-lstm test/test-lstm.cpp ${DIR_SRCS})
