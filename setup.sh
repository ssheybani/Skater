#!/bin/sh
# Requirements: R >= 3.4.0
conda install gxx_linux-64
if [ "$1" = "mac" ]
then
    brew install r
    brew install r-cran-rcpp
    # installing amp
    brew install libgmp3-dev
elif [ "$1" = "linux-ubuntu" ]
then
    sudo apt-get -y install r-base
    sudo apt-get install r-cran-rcpp
    # installing amp
    sudo apt-get install libgmp3-dev
elif [ "$1" = "linux-rpm" ]
then
    sudo yum -y install r-base
    sudo yum install r-cran-rcpp
    sudo yum install libgmp3-dev
fi
# downloads the required R packages locally in the same directory as setup.py
wget https://cran.r-project.org/src/contrib/Rcpp_0.12.15.tar.gz
R CMD INSTALL Rcpp_0.12.15.tar.gz
wget https://cran.r-project.org/src/contrib/arules_1.5-5.tar.gz
R CMD INSTALL arules_1.5-5.tar.gz