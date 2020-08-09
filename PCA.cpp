#include<iostream>
#include<cstdlib>
#include<armadillo>
#include<string>

#define CSV_IO_NO_THREAD
#include "csv.h"

double sampleMean(arma::colvec attributeArray){
    return arma::mean(attributeArray);
}



arma::mat readFile(const char* fileName){
    arma::mat rv(150, 3, arma::fill::zeros);
    io::CSVReader<5> reader(fileName);
    double c1, c2, c3, c4;
    std::string c5;
    for(int i = 0; reader.read_row(c1, c2, c3, c4, c5); i++)
        rv.row(i) = arma::rowvec({c1, c2, c3});
    return rv;
}

arma::mat PCA(arma::mat D, double alpha){
    double n = D.n_rows, d = D.n_cols;
    arma::mat sigma = arma::cov(D, 1);
    sigma.save("./armaMatrixDump/covariance.csv", arma::csv_ascii);
    arma::cx_vec eigval;
    arma::cx_mat eigvec;
    arma::eig_gen(eigval, eigvec, sigma, "balance");
    //eigval.save("./armaMatrixDump/eigenVals.csv", arma::csv_ascii);
    //eigvec.save("./armaMatrixDump/eigenVecs.csv", arma::csv_ascii);
    return D;
}

int main(){
    try{
        arma::mat dataMatrix = readFile("iris.txt");
        //dataMatrix.save("./armaMatrixDump/dataMatrix.csv", arma::csv_ascii);
        PCA(dataMatrix, 0.95);
    }catch(std::exception& e){
        std::cout<<e.what()<<"\n";
    }
}