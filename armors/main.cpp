#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>  
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>

#include "big_armor_scale.hpp"

using namespace std;

class PNP{
    public:
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
        vector <cv::Point2d> pointsCamera;
        vector <cv::Point3d> pointsWorld;
        Eigen::Quaterniond q;
        cv::Mat rvec, tvec;
    
        PNP(){
            cv::FileStorage reader("../f_mat_and_c_mat.yml", cv::FileStorage::READ);
            reader["F"] >> camera_matrix;
            reader["C"] >> dist_coeffs;
            pointsWorld = PW_BIG;
            pointsCamera = {{575.508, 282.175},
                            {573.93, 331.819},
                            {764.518, 337.652},
                            {765.729, 286.741}};
            q = Eigen::Quaterniond(-0.0816168, 0.994363, -0.0676645, -0.00122528);
        };

        void work(){
            // cout << pointsCamera.size() << endl;
            cv::solvePnP(pointsWorld, pointsCamera, camera_matrix, dist_coeffs, rvec, tvec);
            // cout << rvec << ' ' << tvec << endl;
        }
};

PNP* pnp;

class Camera{
    private:
        Eigen::Matrix <double, 3, 3> K; //内参矩阵
        Eigen::Quaterniond q; //位姿四元数
        Eigen::Matrix3d R; //旋转矩阵
        Eigen::Vector3d T; //位移向量，即相机位置
        Eigen::Matrix4d W; //外参矩阵
        double f = 100.0;
        double alpha = 1.0;
        double beta = 1.0;
        double fx = alpha * f;
        double fy = beta * f;

    public:
        Eigen::Vector3d center_world;
        Eigen::Vector3d center_camera;
        Eigen::Vector3d center_inertial;

        Camera(){
            K = Eigen::Map<Eigen::Matrix3d>((pnp -> camera_matrix).ptr<double>(), (pnp -> camera_matrix).rows, (pnp -> camera_matrix).cols);
            cv::Mat R_cv;
            cv::Rodrigues((pnp -> rvec), R_cv);
            R = Eigen::Map<Eigen::Matrix3d>(R_cv.ptr<double>(), R_cv.rows, R_cv.cols);
            T = Eigen::Map<Eigen::Vector3d>((pnp -> tvec).ptr<double>(), (pnp -> tvec).rows, (pnp -> tvec).cols);
            // W = Eigen::Matrix4d::Zero();
            // W.block(0, 0, 3, 3) = R;
            // W.block(0, 3, 3, 1) = -R * T;
            // W(3, 3) = 1; 

            for(auto point : (pnp -> pointsWorld)){
                Eigen::Vector3d point_eigen;
                point_eigen[0] = point.x;
                point_eigen[1] = point.y;
                point_eigen[2] = point.z;
                center_world += point_eigen;
            }

            center_world /= 4;
        }

        void work(){
            center_camera = R * center_world + T;
            center_inertial = (pnp -> q).toRotationMatrix() * center_camera;
            ofstream fout("../result.txt");
            fout << center_inertial << endl;
        }
};

Camera* camera;

int main(){
    pnp = new PNP();
    pnp -> work();
    camera = new Camera();
    camera -> work();
    delete pnp;
    delete camera;
    return 0;
}