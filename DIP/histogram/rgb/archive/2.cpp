#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
const double eps = 1e-8;
const double PI = acos(-1.0);

double sq(double x){return x * x;}
int comp(double x, double y){
    double diff = x - y;
    if(fabs(diff) < eps) return 0;
    if(diff > 0) return 1;
    return -1;
}
Mat rgb2hsi(Mat x){
    Mat ret(x.rows, x.cols, CV_64F);
    printf("xD = %d, retD = %d\n", x.dims, ret.dims);
    int B, G, R;
    double theta;
    double b, g, r, H, S, I;
    for(int i = 0; i < x.rows; ++ i){
        for(int j = 0; j < x.cols; ++ j){
            // printf("(%d, %d)\n", i, j);
            B = x.at<Vec3b>(i, j)[0];
            G = x.at<Vec3b>(i, j)[1];
            R = x.at<Vec3b>(i, j)[2];
            r = R / 255.0;
            g = G / 255.0;
            b = B / 255.0;
            I = (r + g + b) / 3;
            if(comp(I, 0) != 0){
                theta = acos(0.5 * ((r-g) + (r-b)) / (sqrt(sq(r-b) + (r-b) * (g-b))));
                S = 1 - ((3 * min(r, min(g, b))) / (r + g + b));
                if(comp(S, 0) == 1){
                    if(comp(b, g) <= 0){
                        H = theta;
                    }
                    else {
                        H = 2 * PI - theta;
                    }
                }
                else {
                    H = 0;
                }
            }
            else {
                H = S = 0;
            }
            ret.at<Vec3d>(i, j)[0] = H;
            ret.at<Vec3d>(i, j)[1] = S;
            ret.at<Vec3d>(i, j)[2] = I;
        }
    }
    return ret;
}

Mat hsi2rgb(Mat x){
    Mat ret(x.rows, x.cols, CV_8U);
    double r, g, b, H, S, I;
    int B, G, R;
    for(int i = 0; i < x.rows; ++ i){
        for(int j = 0; j < x.cols; ++ j){
            H = x.at<Vec3d>(i, j)[0];
            S = x.at<Vec3d>(i, j)[1];
            I = x.at<Vec3d>(i, j)[2];
            if(H <= PI*2/3){
                r = I * (1 + (S * cos(H)) / (cos(PI/3 - H)));
                b = I * (1 - S);
                g = 1 - (r + b);
            }
            else if(H <= PI*4/3){
                H -= PI*2/3;
                g = I * (1 + (S * cos(H)) / (cos(PI/3 - H)));
                r = I * (1 - S);
                b = 1 - (r + g);
            }
            else {
                H -= PI*4/3;
                b = I * (1 + (S * cos(H)) / (cos(PI/3 - H)));
                g = I * (1 - S);
                r = 1 - (g + b);
            }
            ret.at<Vec3b>(i, j)[0] = B = b * 255;
            ret.at<Vec3b>(i, j)[1] = G = g * 255;
            ret.at<Vec3b>(i, j)[2] = R = r * 255;
        }
    }
    return ret;
}

int main(int argc, char const *argv[]){
    if(argc != 2){
        printf("Usage:\n\t./rgb $IMG_FILE_NAME\n");
        return -1;
    }
    Mat src = imread(argv[1], IMREAD_COLOR);
    Mat hsi = rgb2hsi(src);
    cout << hsi << endl;
    Mat rgb = hsi2rgb(hsi);
    printf("test\n");
    imshow("src", src);
    imshow("hsi", hsi);
    imshow("rgb", rgb);
    waitKey(0);
    return 0;
}
