#ifndef __DATA_H
#define __DATA_H
#include<iostream>
#include<fstream>
#include <sstream>  
#include<string>

#include "def.h"

using namespace std;

bool loadCsv(const char* csvFile, MatrixX& data, MatrixX& label) {
    ifstream in(csvFile);
    if(!in) {
        cerr<<"Error: file '"<<csvFile<<"' not Found!"<<endl;
    } 

    string line;
    vector<vector<string>> strArray;  
    while (getline(in, line)) {
        stringstream ss(line);

        string field;
        vector<string> lineArray;  
        while (getline(ss, field, ',')) {
            lineArray.push_back(field);  
        }  
        strArray.push_back(lineArray); 
    }     

    if(strArray.size() == 0 || strArray[0].size() == 0) return false;

    data.resize(strArray.size(), strArray[0].size()-1);
    label.resize(strArray.size(), 1);

    for(int i=0;i<strArray.size();i++) {
        for(int j=0;j<data.cols();j++) {
            data(i, j) = atof(strArray[i][j].c_str()); 
        }
        label(i, 0) = atof(strArray[i][data.cols()].c_str());
    }
    return true;
}

#endif
