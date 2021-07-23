#include<string>
#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
using namespace std;
using namespace cv;
#define PI acos(-1)
/*
MultiVariate Gaussian Function
@brief 
    Estimate MultiVariate Gaussian distribution by given data, 
    use the distribution get the density probablity(pdf) of data points by matrix manipulateion.

Formula Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
*/
class MultiVariateGaussian{
public:
    cv::Mat mu;
    cv::Mat sig;
    cv::Mat sig_inv;
    double sig_det;
    double constant;
public:
    MultiVariateGaussian(){
    }
    /*
    @brief 
        Estimate MultiVariate Gaussian distribution by given data,
        Get the mean and covariance, covariance's determinant etc.
    @param
        -x: x[batch, dim, channel], channel is 1, dim is 3 default.
    @return
    */
    void estimate_gaussian(cv::Mat x){
        int b=x.rows, d=x.cols;
        cv::calcCovarMatrix(x, this->sig, this->mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
        if(b>1){
            this->sig = this->sig/(1.0*b-1);
        }
        cv::invert(this->sig, this->sig_inv);
        this->sig_det = cv::determinant(this->sig);
        // this->constant is a constant of MultiVariate Gaussian, which can be found in Formula Ref.
        this->constant = (1.0/(pow(2*PI,d*0.5)*sqrt(this->sig_det)));

        //Format Conversion
        this->mu.convertTo(this->mu, CV_64FC1);
        this->sig_inv.convertTo(this->sig_inv, CV_64FC1);
    }
    /*
    @brief 
        Use the distribution get the density probablity of data points by matrix manipulateion.
    @param
        -x: x[batch, dim, channel], channel is 1, dim is 3 default.
    @return
        ret: ret[batch, 1], density probablity of each data point
    */
    cv::Mat multivariate_gaussian(cv::Mat x){ 
        int b=x.rows, d=x.cols;
        auto u_type = this->mu.type();  
        x.convertTo(x, u_type);

        // calculate x-mu
        cv::Mat x_mu= x-(Mat::ones(b, 1, u_type)*this->mu);
        cv::Mat x_mu_T;
        cv::transpose(x_mu, x_mu_T);
        cv::Mat mat_mul = this->sig_inv*x_mu_T;
        cv::transpose(mat_mul, mat_mul); 

        // dot-wise multiplication, then get the sum of each point
        mat_mul = x_mu.mul(mat_mul);
        cv::Mat exp_res = mat_mul.col(0)+mat_mul.col(1)+mat_mul.col(2);
        cv::exp(-0.5*exp_res, exp_res);
        return this->constant * exp_res;
    }   
};
int main(){
    MultiVariateGaussian MVG;
    cv::Mat train_data(10, 3, CV_32FC1);
    cv::Mat val_data(5, 3, CV_32FC1);

    // random data
    cv::RNG rnger(cv::getTickCount());
    rnger.fill(train_data, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(255));
    rnger.fill(val_data, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(255));
    
    // estimate gaussian 
    MVG.estimate_gaussian(train_data);
    cv::Mat val_data_prob = MVG.multivariate_gaussian(val_data);
    
    // show result 
    int n = val_data_prob.rows;
    for(int i=0; i<n; i++){
        cout<<"data: "<<val_data.row(i)<< "\tdense probability: "<<val_data_prob.row(i) << endl;
    }
    return 0;
}