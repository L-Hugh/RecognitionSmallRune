#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ml;

int main(int argc, char** argv)
{
	Mat dst, dst2, dst3, dst4;
	Mat element;
	Mat srcImage;
	double x1, x2, y1, y2;
	dst.create(srcImage.size(), srcImage.type());
	dst2.create(srcImage.size(), srcImage.type());
	dst3.create(srcImage.size(), srcImage.type());
	dst4.create(srcImage.size(), srcImage.type());
	dst = Scalar::all(0);
	dst2 = Scalar::all(0);
	dst3 = Scalar::all(0);
	dst4 = Scalar::all(0);

	const int imageRows = 16;
	const int imageCols = 24;
	//读取训练结果
	Ptr<ANN_MLP> model = StatModel::load<ANN_MLP>("MLPModel.xml");

	VideoCapture cap("12.avi");
	MatND dstHist;
	int dims = 1;
	float hranges[] = { 0,255 };
	const float *ranges[] = { hranges };
	int size = 256;
	int channels = 0;
	int frame = 0;
	for (;;)
	{
		frame++;
		cap >> srcImage;
		if (srcImage.empty())
			break;
		imshow("Original", srcImage);
		element = getStructuringElement(MORPH_RECT, Size(10, 10));
		dilate(srcImage, dst, element);
		//imshow("膨胀", dst);
		erode(dst, dst2, element);
		//imshow("再腐蚀", dst2);
		Mat img;
		vector<Mat> channelsImage;
		split(dst2, channelsImage);
		img = channelsImage.at(1);
		//imshow("绿色通道", img);

		for (int x = 0; x < srcImage.rows; x++)
			for (int y = 0; y < srcImage.cols; y++)
			{
				if (
					srcImage.at<cv::Vec3b>(x, y)[0] > 1.3 * srcImage.at<cv::Vec3b>(x, y)[1] ||
					srcImage.at<cv::Vec3b>(x, y)[0] > 1.3 * srcImage.at<cv::Vec3b>(x, y)[2] ||
					srcImage.at<cv::Vec3b>(x, y)[1] > 1.3 * srcImage.at<cv::Vec3b>(x, y)[2] ||
					srcImage.at<cv::Vec3b>(x, y)[1] > 1.3 * srcImage.at<cv::Vec3b>(x, y)[0] ||
					srcImage.at<cv::Vec3b>(x, y)[2] > 1.3 * srcImage.at<cv::Vec3b>(x, y)[1] ||
					srcImage.at<cv::Vec3b>(x, y)[2] > 1.3 * srcImage.at<cv::Vec3b>(x, y)[0]
					)
					if (x > 4 && y > 4)
					{
						img.at<uchar>(x, y) = 0;// img.at<uchar>(x - 1, y);
					}
					else
					{
						img.at<uchar>(x, y) = 0;
					}

			}

		//imshow("过滤后", img);

		element = getStructuringElement(MORPH_RECT, Size(10, 5));
		dilate(img, dst, element);
		//imshow("过滤后膨胀", dst);
		erode(dst, dst2, element);
		//imshow("过滤后再腐蚀", dst2);

		//imshow("过滤后腐蚀后", dst2);

		calcHist(&dst2, 1, &channels, Mat(), dstHist, dims, &size, ranges);
		int scale = 1;

		Mat dstImage(size *scale, size, CV_8U, Scalar(0));
		double minValue = 0;
		double maxValue = 0;
		minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);

		int hpt = saturate_cast<int>(0.9 * size);
		for (int i = 0; i < 256; i++)
		{
			float binValue = dstHist.at<float>(i);
			int realValue = saturate_cast<int>(binValue * hpt / maxValue);
			rectangle(dstImage, Point(i*scale, size - 1), Point((i + 1)*scale - 1, size - realValue), Scalar(255));
		}
		//imshow("一维直方图", dstImage);

		threshold(dst2, dst, 0, 255, CV_THRESH_OTSU);
		//imshow("大津算法", dst);

		element = getStructuringElement(MORPH_RECT, Size(10, 5));
		erode(dst, dst2, element);
		//imshow("大津后腐蚀", dst2);
		dilate(dst2, dst, element);
		//imshow("大津后膨胀", dst);

		element = getStructuringElement(MORPH_RECT, Size(15, 10));
		dilate(dst, dst2, element);
		//imshow("大津后膨胀2", dst2);
		erode(dst2, dst, element);
		//imshow("大津后腐蚀2", dst);
		

		Canny(dst, dst2, 20, 50, 3);
		//imshow("canny", dst2);

		//////////////轮廓检测
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(dst2, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		if (contours.size() != 9)
			continue;
		int i = 0;
		dst3 = srcImage.clone();
		for (i = 0; i < 9; i++)
		{
			RotatedRect box = minAreaRect(contours[i]);
			Point2f vertex[4];

			box.points(vertex);
			for (int x = 0; x < 4; x++)
			{
				line(dst3, vertex[x], vertex[(x + 1) % 4], Scalar(255, 50, 50), 1);
			}
		}
		//imshow("轮廓检测", dst3);


		


		Mat numbers[9], tmpNumber, tmpNumber1;
		Point2f vertexNumber[9];
		for (int i = 0; i < 9; i++)
		{
			RotatedRect box = minAreaRect(contours[i]);
			Point2f vertex[4];
			box.points(vertex);
			x1 = vertex[0].x;
			x2 = vertex[0].x;
			y1 = vertex[0].y;
			y2 = vertex[0].y;
			for (int x = 1; x < 4; x++)
			{
				if (vertex[x].x > x2)
					x2 = vertex[x].x;
				if (vertex[x].x < x1)
					x1 = vertex[x].x;
				if (vertex[x].y > y2)
					y2 = vertex[x].y;
				if (vertex[x].y < y1)
					y1 = vertex[x].y;
			}
			vertexNumber[i].x = x1;
			vertexNumber[i].y = y2;
			tmpNumber = srcImage(Range(y1, y2), Range(x1, x2)).clone();
			cvtColor(tmpNumber, tmpNumber1, CV_BGR2GRAY);
			resize(tmpNumber1, tmpNumber, Size(24,16));
			numbers[i].create(Size(24, 16), tmpNumber1.type());
			threshold(tmpNumber, numbers[i], 0, 255, CV_THRESH_OTSU);

		}
		/*
			for (int x = 0; x < 9; x++)
		{
			if(frame%10!=1&& frame % 10 != 5)
				break;
			stringstream ss;
			ss << "img/" << frame << x << ".jpg";
			imwrite(ss.str(), numbers[x]);
		}
		*/

		/*
		imshow("数字1", numbers[0]);
		imshow("数字2", numbers[1]);
		imshow("数字3", numbers[2]);
		imshow("数字4", numbers[3]);
		imshow("数字5", numbers[4]);
		imshow("数字6", numbers[5]);
		imshow("数字7", numbers[6]);
		imshow("数字8", numbers[7]);
		imshow("数字9", numbers[8]);
		*/

		Mat_<float> testMat(1, imageRows*imageCols);
		Mat predictDst;
		double maxVal = 0;
		Point maxLoc;
		for (int x = 0; x < 9; x++)
		{
			
			for (int i = 0; i < imageRows*imageCols; i++)
			{
				testMat.at<float>(0, i) = (float)numbers[x].data[i];//at<uchar>(i / 24, i % 24);
			}
			model->predict(testMat, predictDst);
			
			minMaxLoc(predictDst, NULL, &maxVal, NULL, &maxLoc);
			//std::cout << "测试结果：" << maxLoc.x << "置信度:" << maxVal * 100 << "%" << std::endl;
			stringstream ss;
			ss << maxLoc.x + 1;
			putText(dst3, ss.str(), vertexNumber[x], 4,1, Scalar(255, 150, 150), 1);
		}

		imshow("result", dst3);
		//四角检测
		/*
		Point leftMaxP = contours[0][0];
		Point leftMinP = contours[0][0];
		Point rightMaxP = contours[0][0];
		Point rightMinP = contours[0][0];
		double leftMax = abs((double)contours[0][0].x - contours[0][0].y + 10000) / 1.41421;
		double leftMin = abs((double)contours[0][0].x - contours[0][0].y + 10000) / 1.41421;
		double rightMax = abs((double)contours[0][0].x + contours[0][0].y) / 1.41421;
		double rightMin = abs((double)contours[0][0].x + contours[0][0].y) / 1.41421;
		double left;
		double right;
		for (i = 0; i < contours.size(); i++)
		{
			for (int y = 0; y < contours[i].size(); y++)
			{
				left = abs((double)contours[i][y].x - contours[i][y].y + 10000) / 1.41421;
				right = abs((double)contours[i][y].x + contours[i][y].y) / 1.41421;

				if (left < leftMin)
				{
					leftMin = left;
					leftMinP = contours[i][y];
				}
				if (left > leftMax)
				{
					leftMax = left;
					leftMaxP = contours[i][y];
				}
				if (right < rightMin)
				{
					rightMin = right;
					rightMinP = contours[i][y];
				}
				if (right > rightMax)
				{
					rightMax = right;
					rightMaxP = contours[i][y];
				}
			}
		}


		Point tmpP;
		tmpP = leftMinP;   //右边距离最短是左边最短
		leftMinP = rightMinP;
		rightMinP = tmpP;


		dst3 = srcImage.clone();
		line(dst3, leftMaxP, rightMaxP, Scalar(255, 0, 0), 1);
		line(dst3, rightMaxP, rightMinP, Scalar(255, 0, 0), 1);
		line(dst3, rightMinP, leftMinP, Scalar(255, 0, 0), 1);
		line(dst3, leftMinP, leftMaxP, Scalar(255, 0, 0), 1);
		imshow("四角检测", dst3);
		*/


		/*
		Point2f srcTri[4] = {leftMinP, rightMinP, leftMaxP,rightMaxP};
		Point2f dstTri[4];

		//创建仿射变换目标图像与原图像尺寸类型相同
		Mat warp_dstImage = Mat::zeros(srcImage.rows, srcImage.cols, srcImage.type());

		dstTri[0] = Point2f(0,0);
		dstTri[1] = Point2f(srcImage.cols - 1, 0);
		dstTri[2] = Point2f(0, srcImage.rows - 1);
		dstTri[3] = Point2f(srcImage.cols - 1, srcImage.rows - 1);

		//计算仿射变换矩阵
		Mat warp_mat = getAffineTransform(srcTri, dstTri);

		//对加载图形进行仿射变换操作
		warpAffine(srcImage, warp_dstImage, warp_mat, warp_dstImage.size());

		imshow("仿射变换", warp_dstImage);
		*/

		waitKey(1);
	}
	waitKey();
	return 0;
}