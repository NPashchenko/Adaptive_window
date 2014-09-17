#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

using namespace std;
using namespace cv;

const int ITER_MAX_NUM = 8;

float adaptedWindowDisparity(Mat img1, Mat img2, int row, int col, Mat derivatives, Mat disp);

int main(int argc, char **argv)
{
    Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);
    StereoBM bm;
    Mat init_disp, derivatives;
    bm(img1, img2, init_disp, CV_32F);
    //normalize(init_disp, init_disp, 0, 255, CV_MINMAX);
    Sobel(img1, derivatives, CV_16S, 1, 0);

    int iter = 0;
    Mat disp(init_disp.size(), init_disp.type());
    while (iter < ITER_MAX_NUM)
    {
    	for(int i = 0; i < img1.rows; i++) {
            float* disp_ptr = disp.ptr<float>(i);
    		for(int j = 0; j < img1.cols; j++) {
                disp_ptr[j] = adaptedWindowDisparity(img1, img2, i, j, derivatives, init_disp);
    		}
    	}
        Mat changed;
        compare(abs(disp - init_disp), 0.1, changed, CMP_GE);
        if(!countNonZero(changed))
            break;
    }

    normalize(disp, disp, 0, 255, CV_MINMAX);
    imwrite(argv[3], disp);
    return 0;
}

float adaptedWindowDisparity(Mat img1, Mat img2, int row, int col, Mat derivatives, Mat disp)
{
    return disp.at<float>(row, col);
}
