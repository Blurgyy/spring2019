#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
const double eps = 1e-8;
const double pi = acos(-1);

template<class T> 
T min(T a, T b, T c){return min(a, min(b, c));}
double sq(double x){return x * x;}
template<class T>
bool inrange(T x, T l, T r){return l <= x && x <= r;}
int comp(double a, double b){
    double dif = a - b;
    if(fabs(dif) <= eps) return 0;
    if(dif < 0) return -1;
    return 1;
}
int unify(double x, double range = 1.0, int dest = 255){  /// unifies value x from [0, range] into [0, dest]
    if(!inrange(x, 0.0, range)){
        printf("x = %f, range = %f, x is not in range [0, %f]\n", x, range, range);
        assert(inrange(x, 0.0, range));
    }
    double ratio = x / range;
    int ret = (int)(1.0 * dest * ratio + 0.5);
    assert(inrange(ret, 0, dest));
    // printf("unified: %d\n", ret);
    return ret;
}

Mat rgb2hsi(Mat &x){    // where x is an r-g-b image
    int rows = x.rows;
    int cols = x.cols;
    Mat ret(rows, cols, x.type());
    for(int i = 0; i < rows; ++ i){
        for(int j = 0; j < rows; ++ j){
            int B = x.at<Vec3b>(i, j)[0];
            int G = x.at<Vec3b>(i, j)[1];
            int R = x.at<Vec3b>(i, j)[2];
            double r = 1.0 * R / 255;
            double g = 1.0 * G / 255;
            double b = 1.0 * B / 255;
            double H, S, I, theta;
            I = (r + g + b) / 3;
            if(comp(I, 0) == 0){
                S = 0;
            }
            else{
                S = 1 - (3 * min(r, g, b)) / (r + g + b);
            }
            if(comp(r, g) == 0 && comp(g, b) == 0){
                H = 0;
            }
            else{
                assert(sq(r-g) + (r-b) * (g-b) >= 0);
                double param = (0.5 * (r-g+r-b)) / sqrt(sq(r-g) + (r-b) * (g-b));
                if(comp(param, 1) >= 0){    /// IMPORTANT
                    param = 1.0;
                }
                if(comp(param, 0) <= 0){
                    param = 0.0;
                }
                theta = acos(param);
                if(!inrange(theta, 0.0, pi)){
                    printf("param = %f\n", param);
                    printf("theta = %f\n", acos(param));
                    assert(inrange(theta, 0.0, pi));
                }
                if(comp(b, g) <= 0){
                    H = theta;
                }
                else {
                    H = 2 * pi - theta;
                }
            }
            // printf("H = %f, S = %f, I = %f\n", H, S, I);
            int hh, ss, ii;
            hh = unify(H, 2 * pi, 180);
            ss = unify(S);
            ii = unify(I);
            ret.at<Vec3b>(i, j)[0] = hh;
            ret.at<Vec3b>(i, j)[1] = ss;
            ret.at<Vec3b>(i, j)[2] = ii;
        }
    }
    return ret;
}

Mat hsi2rgb(Mat &x){    // where x is an h-s-i image
    int rows = x.rows;
    int cols = x.cols;
    Mat ret(rows, cols, x.type());
    for(int i = 0; i < rows; ++ i){
        for(int j = 0; j < cols; ++ j){
            int hh = x.at<Vec3b>(i, j)[0];
            int ss = x.at<Vec3b>(i, j)[1];
            int ii = x.at<Vec3b>(i, j)[2];
            double H, S, I, r, g, b;
            if(0 <= hh && hh <= 60){    // RG sector
                
            }
            else if(60 < hh <= 120){    // GB sector
                hh -= 60;
            }
            else if(120 < hh <= 180){   // BR sector
                hh -= 120;
            }
            else {
                printf("wtf\n");
                assert(0);
            }
            I = 1.0 * ii / 255;
            S = 1.0 * ss / 255;
            H = 2 * pi * hh / 180;
            double t1 = I * (1 + (S * cos(H)) / (cos(pi/3 - H)));
            double t2 = I * (1 - S);
            double t3 = 3 * I - (t1 + t2);
            // printf("t1 = %f, t2 = %f, t3 = %f\n", t1, t2, t3);
            char sector[5] = {0};
            if(0 <= hh && hh <= 60){    // RG sector
                // printf("RG sector\n");
                sector[0] = 'R';
                sector[1] = 'G';
                r = t1;
                g = t3;
                b = t2;
            }
            else if(60 < hh <= 120){    // GB sector
                // printf("GB sector\n");
                sector[0] = 'G';
                sector[1] = 'B';
                r = t2;
                g = t1;
                b = t3;
            }
            else if(120 < hh <= 180){   // BR sector
                // printf("BR sector\n");
                sector[0] = 'B';
                sector[1] = 'R';
                r = t3;
                g = t2;
                b = t1;
            }
            if(!inrange(r, 0.0, 1.0)){
                printf("assertion failed: r = %f\n", r);
                printf("I = %f\n", I);
                printf("in %s sector:\nt1 = %f\nt2 = %f\nt3 = %f\n\n", sector, t1, t2, t3);
                assert(inrange(r, 0.0, 1.0));
            }
            int R = unify(r, 1.0, 255);
            if(!inrange(g, 0.0, 1.0)){
                printf("assertion failed: g = %f\n", g);
                printf("I = %f\n", I);
                printf("in %s sector:\nt1 = %f\nt2 = %f\nt3 = %f\n\n", sector, t1, t2, t3);
                assert(inrange(g, 0.0, 1.0));
            }
            int G = unify(g, 1.0, 255);
            if(!inrange(b, 0.0, 1.0)){
                printf("assertion failed: b = %f\n", b);
                printf("I = %f\n", I);
                printf("in %s sector:\nt1 = %f\nt2 = %f\nt3 = %f\n\n", sector, t1, t2, t3);
                assert(inrange(b, 0.0, 1.0));
            }
            int B = unify(b, 1.0, 255);

            ret.at<Vec3b>(i, j)[0] = B;
            ret.at<Vec3b>(i, j)[1] = G;
            ret.at<Vec3b>(i, j)[2] = R;
        }
    }
    return ret;
}

int main(int argc, char const *argv[]){
    if(argc != 2){
        printf("Usage:\n    rgb $IMG_FILE_NAME\n");
        return -1;
    }
    Mat src = imread(argv[1], IMREAD_COLOR);
    Mat hsi = rgb2hsi(src);
    imshow("hsi", hsi);
    waitKey(0);
    Mat rgb = hsi2rgb(hsi);
    imshow("rgb", rgb);
    waitKey(0);
    return 0;
}
