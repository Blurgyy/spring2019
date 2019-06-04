#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
const int max_grayscale = 256;

int mp[max_grayscale];

int main(int argc, char **argv){
    if(argc != 2){
        printf("Usage:\n\thistogram $IMG_FILE_NAME\n");
        return -1;
    }
    Mat img = imread(argv[1], IMREAD_GRAYSCALE);
    int rows = img.rows;
    int cols = img.cols;
    Mat nimg = Mat(rows, cols, CV_8UC1);
    memset(mp, 0, sizeof(mp));
    for(int i = 0; i < rows; ++ i){
        uchar *x = img.ptr<uchar>(i);
        for(int j = 0; j < cols; ++ j){
            mp[x[j]] ++;
        }
    }
    for(int i = 1; i < max_grayscale; ++ i){
        mp[i] += mp[i-1];
    }
    for(int i = 0; i < max_grayscale; ++ i){
        mp[i] = (255.0 * mp[i]) / (1.0 * mp[max_grayscale-1]) + 0.5;
    }
    for(int i = 0; i < rows; ++ i){
        uchar *x = img.ptr<uchar>(i);
        uchar *y = nimg.ptr<uchar>(i);
        for(int j = 0; j < cols; ++ j){
            y[j] = mp[x[j]];
        }
    }
    Mat showimg = Mat(rows, cols * 2, CV_8UC1);
    for(int i = 0; i < rows; ++ i){
        uchar *x = img.ptr<uchar>(i);
        uchar *y = nimg.ptr<uchar>(i);
        uchar *z = showimg.ptr<uchar>(i);
        for(int j = 0; j < cols * 2; ++ j){
            if(j < cols) {
                z[j] = x[j];
            }
            else {
                z[j] = y[j];
            }
        }
    }
    imshow("histogram", showimg);
    waitKey();
    return 0;
}
