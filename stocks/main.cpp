#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>  
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <time.h>

using namespace std;

int cnt = 0;
vector<double> predicts;
mt19937 gen((unsigned int)time(NULL));

template<int N_X, int N_Y>

class KalmanFilter{
    public: 
        Eigen::Matrix <double, N_X, 1> X; //predict vector
        Eigen::Matrix <double, N_X, 1> U;
        Eigen::Matrix <double, N_Y, 1> Z; //watch vector
        Eigen::Matrix <double, N_X, N_X> P; //x covariance
        Eigen::Matrix <double, N_X, N_X> Q; //u variance
        Eigen::Matrix <double, N_X, N_X> F; //transformation matrix
        Eigen::Matrix <double, N_Y, N_Y> R; //z covariance
        Eigen::Matrix <double, N_Y, N_X> H;

        KalmanFilter(){
            X[0] = predicts[2];
            X[1] = (predicts[2] - predicts[0]) / 2.0;
            X[2] = (predicts[2] - predicts[1]) - (predicts[1] - predicts[0]);
            Z[0] = predicts[2] - predicts[0];
            P << 2., 0., 0.,
                0., 6., 0.,
                0., 0., 8.;
            Q << 0.0001, 0., 0.,
                0., 0.00005, 0.,
                0., 0., 0.000001;
            F << 1., 1., 0.,
                0., 1., 1.,
                0., 0., 1.;
            R << 0.01;
            H << 1., 0., 0.;
        }
        
        void update(double z){
            Eigen::Matrix <double, N_X, 1> X_predict;
            X_predict = F * X + U;
            Eigen::Matrix <double, N_Y, 1> Y;
            Z[0] = z;
            Y = Z - H * X_predict;
            Eigen::Matrix <double, N_Y, N_Y> S;
            S = H * (F * P * F.transpose() + Q) * H.transpose() + R;
            Eigen::Matrix <double, N_X, N_Y> K;
            K = (F * P * F.transpose() + Q) * H.transpose() * S.inverse();
            X = X_predict + K * Y;
            P = (Eigen::Matrix <double, N_X, N_X>::Identity() - K * H) * (F * P * F.transpose() + Q);
        }

        Eigen::MatrixXd multiNormalDistribution(Eigen::MatrixXd M, Eigen::MatrixXd Cov){
            Eigen::MatrixXd Ans(M);
            for(int i = 0; i < M.rows(); i++){
                normal_distribution<double> normal(M(i, 0), Cov(i, i));
                Ans(i, 0) = normal(gen);
            }
            return Ans;
            
        }   

        void update_null(){
            Eigen::Matrix <double, N_X, 1> X_predict;
            X_predict = F * X  + U;
            X = multiNormalDistribution(X_predict, F * P * F.transpose() + Q);
        }
};

int main(){
    ifstream fin("../dollar.txt");
    double tmp;
    while(fin >> tmp){
        predicts.push_back(tmp);
    }
    ofstream fout("../result.txt");
    KalmanFilter<3, 1> filter;
    fout << "前30日的滤波预测数据：" << endl;
    for(int i = 3; i < predicts.size(); i++){
        cnt++;
        filter.update(predicts[i]);
        fout << "X: " << filter.X[0] << endl;
        // cout << "X: " << filter.X << endl;
        // cout << "Z: " << filter.Z << endl;
    }
    fout << "对后10日的预测：" << endl;
    for(int i = 0; i < 10; i++){
        filter.update_null();
        fout << "X: " << filter.X[0] << endl;
        cout << "X: " << filter.X << endl;
    }
    return 0;
}