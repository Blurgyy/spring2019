#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
const double eps = 1e-8;
const double PI = acos(-1.0);
const size_t max_uchar = 255;

double sq(double x){return x * x;}
double hue(int R, int G, int B){
    double sum = (R + G + B);
    double r = 1.0 * R / sum;
    double g = 1.0 * G / sum;
    double b = 1.0 * B / sum;
    assert(sq(r-g) + (r-b) * (g-b) > 0);
    double ret = acos((0.5 * (r - g + r - b)) / sqrt(sq(r-g) + (r-b) * (g-b)));
    if(g < b) ret = PI * 2 - ret;
    return ret;
}
double saturation(int r, int g, int b){return 1 - (3.0 * min(r, min(g, b))) / (r + g + b);}
int intensity(int r, int g, int b){return (r + g + b) / 3;}
double deg2rad(int x){return PI * x / 180;}
int rad2deg(double x){return x * 180 / PI;}

int comp(double a, double b){
    double diff = a - b;
    if(fabs(diff) < eps) return 0;
    if(diff > 0) return 1;
    return -1;
}

int main(int argc, char const *argv[]){
    if(argc != 2){
        printf("Usage:\n\t./rgb $IMG_FILE_NAME\n");
        return -1;
    }
    Mat src = imread(argv[1], IMREAD_COLOR);
    Mat img = Mat(src.rows, src.cols, src.type());
    /** img = rgb2hsi(src); **/
    for(int i = 0; i < src.rows; ++ i){
        for(int j = 0; j < src.cols; ++ j){
            int b = src.at<Vec3b>(i, j)[0];
            int g = src.at<Vec3b>(i, j)[1];
            int r = src.at<Vec3b>(i, j)[2];
            double Hue, Saturation;
            int Intensity;
            // assert(g >= b);
            Intensity = intensity(r, g, b);
            if(Intensity > 0){
                Saturation = saturation(r, g, b);
                if(comp(Saturation, 0) == 1){
                    Hue = hue(r, g, b);
                }
                else {
                    Saturation = 0;
                }
            }
            else {
                Saturation = Hue = 0;
            }
            int hh = img.at<Vec3b>(i, j)[0] = rad2deg(Hue) / 2;     // in deg
            int ss = img.at<Vec3b>(i, j)[1] = Saturation * 255;     // normalize
            int ii = img.at<Vec3b>(i, j)[2] = Intensity;
            // printf("Hue = %d, Saturation = %d, Intensity = %d\n", hh, ss, ii);
        }
    }
    // imshow("test", img);
    // waitKey();
    /** Operate on intensity channel **/
    /************************/
    // vector<Mat> channels;
    // split(img, channels);
    // // imshow("src", src);
    // imshow("test intensity", channels.at(0));
    // waitKey();
    // // return 0;

    // int mp[max_uchar+1];
    // memset(mp, 0, sizeof(mp));
    // for(int i = 0; i < img.rows; ++ i){
    //     for(int j = 0; j < img.cols; ++ j){
    //         int Intensity = img.at<Vec3b>(i, j)[2];
    //         mp[Intensity] ++;
    //     }
    // }
    // for(int i = 0; i <= max_uchar; ++ i){
    //     if(mp[i]){
    //         printf("min = %d\n", i);
    //         break;
    //     }
    // }
    // for(int i = max_uchar; i >= 0; -- i){
    //     if(mp[i]){
    //         printf("max = %d\n", i);
    //         break;
    //     }
    // }
    // for(int i = 1; i <= max_uchar; ++ i)
    //     mp[i] += mp[i-1];
    // printf("%d\n", mp[max_uchar]);
    // assert(mp[max_uchar] == img.rows * img.cols);
    // for(int i = 0; i <= max_uchar; ++ i){
    //     mp[i] = 1.0 * max_uchar * mp[i] / (1.0 * mp[max_uchar]) + 0.5;
    // }
    // for(int i = 0; i < img.rows; ++ i){
    //     for(int j = 0; j < img.cols; ++ j){
    //         img.at<Vec3b>(i, j)[2] = mp[img.at<Vec3b>(i, j)[2]];
    //     }
    // }
    // for(int i = 0; i <= max_uchar; ++ i){
    //     if(mp[i]){
    //         printf("min = %d\n", i);
    //         break;
    //     }
    // }
    // for(int i = max_uchar; i >= 0; -- i){
    //     if(mp[i]){
    //         printf("max = %d\n", i);
    //         break;
    //     }
    // }
    // split(img, channels);
    // imshow("test2 intensity", channels.at(0));
    /***************************/
    /** nimg = hsi2rgb(img) **/
    Mat nimg(img.rows, img.cols, img.type());
    for(int i = 0; i < img.rows; ++ i){
        for(int j = 0; j < img.cols; ++ j){
            int H_deg = img.at<Vec3b>(i, j)[0];
            H_deg *= 2;
            double ss = 1.0 * img.at<Vec3b>(i, j)[1] / 255;
            int ii = img.at<Vec3b>(i, j)[2];
            int b, g, r;
            double H_rad = deg2rad(H_deg);
            if(H_deg <= 120){
                b = 1.0 * ii * (1 - ss);
                r = 1.0 * ii * (1 + (ss * cos(H_rad)) / cos(PI/3 - H_rad));
                g = 3 * ii - (b + r);
            }
            else if(H_deg <= 240){
                r = 1.0 * ii * (1 - ss);
                g = 1.0 * ii * (1 + (ss * cos(H_rad - PI*2/3)) / cos(PI - H_rad));
                b = 3 * ii - (r + g);
            }
            else {
                g = 1.0 * ii * (1 - ss);
                b = 1.0 * ii * ((ss * cos(H_rad - PI*4/3)) / cos(PI*5/3 - H_rad));
                r = 3 * ii - (g + b);
            }
            nimg.at<Vec3b>(i, j)[0] = b;
            nimg.at<Vec3b>(i, j)[1] = g;
            nimg.at<Vec3b>(i, j)[2] = r;
        }
    }
    imshow("src", src);
    imshow("test", nimg);
    waitKey();
    return 0;
}
