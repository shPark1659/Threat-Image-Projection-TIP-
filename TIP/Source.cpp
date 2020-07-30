#include <ctime>
#include <cstdlib>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <filesystem>

#include <direct.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>

#include "Header.h"

#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>

using namespace cv;
using namespace std;


int main() {
	
	ThreatImageAugment New;
	for (int n = 0; n < 10000; n++)
	{
		int i = rand() % 11;
		New._run(i, n);
	}
		

	return 0;
}

void ThreatImageAugment::_run(int Index_BG, int nRst) {

	// IG BG = ReadImg("background_" + to_string(Index_BG), 0, READIMG_TXT_SEPERATE);
	IG BG = ReadImg("background_" + to_string(Index_BG), 0, READIMG_PNG);
	IG T, Result;
	int Total = 0;

	if (_mkdir(dst)) {
		cout << "already exist dst folder!!" << endl;
	}

	for (int classList = 1; classList < 5; classList++) {
		for (int No = 0; No < ChoiceNo[classList]; No++) {
			random_device rng;
			int Index_C = rng() % No_per_Class[classList];
			//T = ReadImg(Class[classList] + "_" + to_string(Index_C), classList, READIMG_TXT_UNITE);
			T = ReadImg(Class[classList] + "_" + to_string(Index_C), classList, READIMG_PNG);
			T = Vol_N_Den(T);
			T = Rotation(T);

			if (Total == 0)
			{
				if ((BG.Img_high.cols <= T.Img_high.cols) || (BG.Img_high.rows <= T.Img_high.rows))
					continue;
				Result = Translation(BG, T);
			}				
			else 
			{
				if ((BG.Img_high.cols <= T.Img_high.cols) || (BG.Img_high.rows <= T.Img_high.rows))
					continue;
				Result = Translation(Result, T);
			}
				

			Total++;
		}
	}
	
	Result = InvNormalize(Result);

	SaveIG(Result, "Result_" + to_string(nRst));

	cout << "#" << nRst << " OK" << endl;

}

// low 영상, high 영상를 read해서 IG 출력
IG ThreatImageAugment::ReadImg(string filename, int classnum, int type) {

	Mat Lsrc, Hsrc;
	string name;
	int val, height, width;
	
	// png파일로 low 영상, high 영상 따로 read
	if (type == 0) {																 

		string name_low = "src" + backslash + Class[classnum] + backslash + filename + "_low" + extension_png;
		string name_high = "src" + backslash + Class[classnum] + backslash + filename + "_high" + extension_png;
		Lsrc = cv::imread(name_low, IMREAD_ANYDEPTH);
		Hsrc = cv::imread(name_high, IMREAD_ANYDEPTH);

		Hsrc.convertTo(Hsrc, CV_16UC1);
		Lsrc.convertTo(Lsrc, CV_16UC1);
		if (((ushort *)Hsrc.data)[0] < 256)
			Hsrc = Hsrc * 65535 / 255;
		if (((ushort *)Lsrc.data)[0] < 256)
			Lsrc = Lsrc * 65535 / 255;
	}

	// txt 파일로 low 영상, high 영상 따로 read
	else if (type == 1) {

		ifstream file_low;
		name = "src" + backslash + Class[classnum] + backslash + filename + "_low" + extension_txt;
		file_low.open(name);

		if (file_low.is_open()) {
			int n = 0;

			//file_low >> height;
			//file_low >> width;
			height = HEIGHT;
			width = WIDTH;

			int *rawdata = new int[sizeof(int) * height * width * 2];

			Lsrc = Mat::zeros(height, width, CV_16UC1);

			while (file_low >> val)
				rawdata[n++] = val;

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					Lsrc.at<ushort>(i, j) = rawdata[j * height + i];
				}
			}
			delete rawdata;
		}
		file_low.close();

		ifstream file_high;
		name = "src" + backslash + Class[classnum] + backslash + filename + "_high" + extension_txt;
		file_high.open(name);

		if (file_high.is_open()) {
			int n = 0;

			//file_high >> height;
			//file_high >> width;
			height = HEIGHT;
			width = WIDTH;

			int *rawdata = new int[sizeof(int) * height * width * 2];

			Hsrc = Mat::zeros(height, width, CV_16UC1);

			while (file_high >> val)
				rawdata[n++] = val;

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					Hsrc.at<ushort>(i, j) = rawdata[j * height + i];
				}
			}
			delete rawdata;
		}
		file_high.close();
	}

	// txt 파일로 low 영상, high 영상 한번에 read
	else if (type == 2)
	{

		ifstream file;
		name = "src" + backslash + Class[classnum] + backslash + filename + extension_txt;
		file.open(name);

		if (file.is_open()) {
			int n = 0;

			file >> height;
			file >> width;

			Lsrc = Mat::zeros(height, width, CV_16UC1);
			Hsrc = Mat::zeros(height, width, CV_16UC1);

			int *rawdata = new int[sizeof(int) * height * width * 2];

			while (file >> val)
				rawdata[n++] = val;

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					Lsrc.at<ushort>(i, j) = rawdata[i * (2 * width) + j];
					Hsrc.at<ushort>(i, j) = rawdata[i * (2 * width) + j + width];
				}
			}

			delete rawdata;
		}


		file.close();
	}
	else if (type == 3)
	{
		string name_low = "src" + backslash + Class[classnum] + backslash + filename + "_low" + extension_tiff;
		string name_high = "src" + backslash + Class[classnum] + backslash + filename + "_high" + extension_tiff;
		Lsrc = cv::imread(name_low, IMREAD_ANYDEPTH);
		Hsrc = cv::imread(name_high, IMREAD_ANYDEPTH);
	}

	if (classnum) {
		Lsrc = Saturation(Lsrc);
		Hsrc = Saturation(Hsrc);
	}
	Lsrc = Norm(Lsrc);
	Hsrc = Norm(Hsrc);

	IG output = MakeIG(Lsrc, Hsrc, classnum);

	return output;
}

// 위험물 영상의 background를 max value로 saturation
Mat ThreatImageAugment::Saturation(Mat src) {

	Mat output;
	src.copyTo(output);

	ushort *data_input = (ushort *)output.data;
	for (int c = 0; c < output.cols; c++)
		for (int r = 0; r < output.rows; r++)
			if (data_input[r * output.cols + c] > threshold_value)
				data_input[r * output.cols + c] = maxVal;

	return output;
}

// ushort type(0 ~ 65535)인 Input 영상(Mat)를 normalize해서 float type(0 ~ 1)으로 변환(normalize)
Mat ThreatImageAugment::Norm(Mat src) {

	Mat input;
	src.copyTo(input);

	input.convertTo(input, CV_32FC1);

	Mat output(src.rows, src.cols, CV_32FC1);
	output = input / maxVal;

	return output;
}

// low-energy 영상, high-energy 영상과 classnum으로 GT를 만들어서 하나의 IG로 합침
IG ThreatImageAugment::MakeIG(Mat low, Mat high, int classnum) {
	GT zero_GT = { 0,0,0,0,0 };
	IG output = { Mat::zeros(low.rows, low.cols, CV_16UC1), Mat::zeros(high.rows, high.cols, CV_16UC1), vector <GT> { zero_GT } };
	output.GroundTruth[0].Nclass = classnum;
	low.copyTo(output.Img_low);
	high.copyTo(output.Img_high);
	return output;
}

// 위험물의 class가 폭발물이나 도검류일 경우, 밀도와 크기를 무작위 난수로 정해진 범위내에서 조정
IG ThreatImageAugment::Vol_N_Den(IG Threat) {

	IG output = Threat;
	if ((Threat.GroundTruth[0].Nclass == 3)||(Threat.GroundTruth[0].Nclass == 4)) {
		srand((uint)time(NULL));

		Mat Resize_low, Resize_high;

		float rate_Vol = ((float)(rand()) / 32767) * (Volume_max - Volume_min) + Volume_min;
		float rate_Den = ((float)(rand()) / 32767) * (Density_max - Density_min) + Density_min;

		output.Img_low.copyTo(Resize_low);
		output.Img_high.copyTo(Resize_high);

		float *data_Resize_low = (float *)Resize_low.data;
		float *data_Resize_high = (float *)Resize_high.data;

		for (int c = 0; c < Resize_low.cols; c++) {
			for (int r = 0; r < Resize_low.rows; r++) {
				data_Resize_low[r * Resize_low.cols + c] = pow(data_Resize_low[r * Resize_low.cols + c], rate_Den);
				data_Resize_high[r * Resize_high.cols + c] = pow(data_Resize_high[r * Resize_high.cols + c], rate_Den);
			}
		}

		resize(Resize_low, Resize_low, Size((int)(Threat.Img_low.cols * rate_Vol), (int)(Threat.Img_low.rows * rate_Vol)));
		resize(Resize_high, Resize_high, Size((int)(Threat.Img_high.cols * rate_Vol), (int)(Threat.Img_high.rows * rate_Vol)));

		Resize_low.copyTo(output.Img_low);
		Resize_high.copyTo(output.Img_high);
	}
	return output;
}

// 위험물 영상을 무작위 난수로 회전
IG ThreatImageAugment::Rotation(IG Threat) {

	IG output = Threat;
	Mat RotatedTI_low, RotatedTI_high;

	int angle = RndAngle();
	int row = Threat.Img_low.rows;
	int col = Threat.Img_low.cols;
	int dia = (int)sqrt(col * col + row * row);
	int offsetX = (dia - col) / 2;
	int offsetY = (dia - row) / 2;
	int rowRotated, colRotated;

	Mat targetMat_low(dia, dia, Threat.Img_low.type(), Scalar(1));
	Mat targetMat_high(dia, dia, Threat.Img_high.type(), Scalar(1));
	Point2f Threat_center(targetMat_low.cols / 2.0F, targetMat_low.rows / 2.0F);

	Threat.Img_low.copyTo(targetMat_low.rowRange(offsetY, offsetY + Threat.Img_low.rows).colRange(offsetX, offsetX + Threat.Img_low.cols));
	Threat.Img_high.copyTo(targetMat_high.rowRange(offsetY, offsetY + Threat.Img_high.rows).colRange(offsetX, offsetX + Threat.Img_high.cols));
	Mat rot_mat = getRotationMatrix2D(Threat_center, angle, 1.0);
	warpAffine(targetMat_low, RotatedTI_low, rot_mat, targetMat_low.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(1));
	warpAffine(targetMat_high, RotatedTI_high, rot_mat, targetMat_high.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(1));

	Rect bound_Rect(Threat.Img_low.cols, Threat.Img_low.rows, 0, 0);

	Mat co_Ordinate = (Mat_<double>(3, 4) << offsetX, offsetX + Threat.Img_low.cols, offsetX, offsetX + Threat.Img_low.cols,
		offsetY, offsetY, offsetY + Threat.Img_low.rows, offsetY + Threat.Img_low.rows,
		1, 1, 1, 1);
	Mat RotCo_Ordinate = rot_mat * co_Ordinate;

	for (int i = 0; i<4; i++) {
		if (RotCo_Ordinate.at<double>(0, i)<bound_Rect.x)
			bound_Rect.x = (int)RotCo_Ordinate.at<double>(0, i);
		if (RotCo_Ordinate.at<double>(1, i)<bound_Rect.y)
			bound_Rect.y = RotCo_Ordinate.at<double>(1, i);
	}

	for (int i = 0; i<4; i++) {
		if (RotCo_Ordinate.at<double>(0, i)>bound_Rect.width)
			bound_Rect.width = (int)RotCo_Ordinate.at<double>(0, i);
		if (RotCo_Ordinate.at<double>(1, i)>bound_Rect.height)
			bound_Rect.height = RotCo_Ordinate.at<double>(1, i);
	}

	bound_Rect.width = bound_Rect.width - bound_Rect.x;
	bound_Rect.height = bound_Rect.height - bound_Rect.y;

	cout << "Threat.Img_low.rows: " << Threat.Img_low.rows << "\t\tThreat.Img_low.cols: " << Threat.Img_low.cols << endl;
	cout << "RotatedTI_low.rows: "<<RotatedTI_low.rows << "\t\tRotatedTI_low.cols: " << RotatedTI_low.cols << endl;
	cout << "x: "<<bound_Rect.x << "\ty: " << bound_Rect.y << "\theight " << bound_Rect.height << "\twidth: " << bound_Rect.width << endl;
	
	if (bound_Rect.x < 0)
	{
		bound_Rect.x = 0;
		cout << "ROI region error" << endl;
	}
	if (bound_Rect.y < 0)
	{
		bound_Rect.y = 0;
		cout << "ROI region error" << endl;
	}
	if (bound_Rect.x + bound_Rect.width > RotatedTI_low.cols)
	{
		bound_Rect.width = RotatedTI_low.cols - bound_Rect.x - 1;
		cout << "ROI region error" << endl;
	}
	if (bound_Rect.y + bound_Rect.height > RotatedTI_low.rows)
	{
		bound_Rect.height = RotatedTI_low.rows - bound_Rect.y - 1;
		cout << "ROI region error" << endl;
	}

	Mat ROI_low = RotatedTI_low(bound_Rect);
	Mat ROI_high = RotatedTI_high(bound_Rect);
	ROI_low.copyTo(output.Img_low);
	ROI_high.copyTo(output.Img_high);

	return output;
}

// 무작위 난수로 각도 출력
int ThreatImageAugment::RndAngle() {

	random_device rn;
	mt19937_64 generator(rn());
	uniform_int_distribution<int> distribution(0, 360);
	int angle = distribution(generator);
	return angle;
}

// 배경 영상에 위험물 영상과 GT를 합성
IG ThreatImageAugment::Translation(IG src, IG Threat) {
	IG output = src;
	Mat AugImg_low, AugImg_high;
	src.Img_low.copyTo(AugImg_low);
	src.Img_high.copyTo(AugImg_high);

	try {
		Point2i top_left = RndPoint(src, Threat);
		Point2i bottom_right = top_left + Point2i(Threat.Img_low.size());;

		float *data_src_low = (float *)src.Img_low.data;
		float *data_src_high = (float *)src.Img_high.data;
		float *data_threat_low = (float *)Threat.Img_low.data;
		float *data_threat_high = (float *)Threat.Img_high.data;
		float *data_AugImg_low = (float *)AugImg_low.data;
		float *data_AugImg_high = (float *)AugImg_high.data;

		int n = 0;
		for (int c = top_left.x; c < bottom_right.x; c++) {
			for (int r = top_left.y; r < bottom_right.y; r++) {
				if ((r < AugImg_low.rows) && (c < AugImg_low.cols) && (r > -1) && (c > -1)) {
					data_AugImg_low[r * AugImg_low.cols + c] = data_src_low[r * src.Img_low.cols + c] * data_threat_low[(r - top_left.y) * Threat.Img_low.cols + (c - top_left.x)];
					data_AugImg_high[r * AugImg_high.cols + c] = data_src_high[r * src.Img_high.cols + c] * data_threat_high[(r - top_left.y) * Threat.Img_high.cols + (c - top_left.x)];
				}
			}
		}

		AugImg_low.copyTo(output.Img_low);
		AugImg_high.copyTo(output.Img_high);

		AddGT(&output, Threat, top_left);

	}

	catch (Mat error) {
		cout << "위험물 영상의 크기가 적합하지 않습니다." << endl << "배경 영상의 크기 :" << src.Img_low.size() << endl;
	}

	return output;
}

// 배경 영상을 erosion해서 위험물 영상을 자연스럽게 합성할 위치를 난수로 출력
Point2i ThreatImageAugment::RndPoint(IG BG, IG threat) {

	Mat thresholding, uintmat, eroded;

	BG.Img_low.copyTo(uintmat);
	uintmat = uintmat * 256;
	uintmat.convertTo(uintmat, CV_8UC1);
	//threshold(uintmat, thresholding, threshold_value / 256, 255, 0);
	int ithreshold = 220;
	vector<Point2i> ROI;
	Mat erosion_mask = Mat::zeros(2 * threat.Img_low.rows + 1, 2 * threat.Img_low.cols + 1, CV_8UC1);

	////////////////0619 수정
	for (int c = 0; c < BG.Img_low.cols-threat.Img_low.cols; c++)
		for (int r = 0; r < BG.Img_low.rows-threat.Img_low.rows; r++)
			ROI.push_back(Point2d(c, r));

	//uchar *data_erosion_mask = (uchar *)erosion_mask.data;
	//for (int c = 0; c < erosion_mask.cols; c++)
	//	for (int r = 0; r < erosion_mask.rows; r++)
	//		if (r >(threat.Img_low.rows))
	//			if (c >(threat.Img_low.cols))
	//				data_erosion_mask[r * erosion_mask.cols + c] = 255;
	//
	//while (1) {
	//	threshold(uintmat, thresholding, ithreshold, 255, 0);
	//	filter2D(thresholding, eroded, -1, erosion_mask);

	//	uchar *data_eroded = (uchar *)eroded.data;
	//	////////////////0619 수정
	//	for (int c = 0; c < eroded.cols-threat.Img_low.cols; c++)
	//		for (int r = 0; r < eroded.rows-threat.Img_low.rows; r++)
	//			if (data_eroded[r * eroded.cols + c] < 128)
	//				ROI.push_back(Point2d(c, r));

	//	cout << "threshold: " << ithreshold << "\tROI: " << ROI.size() << endl;
	//	if (ROI.size() != 0) break;
	//	ithreshold -= 10;
	//	if (ithreshold == 0) {
	//		ROI.push_back(Point2d(eroded.cols / 2 - threat.Img_low.cols / 2, eroded.rows / 2 - threat.Img_low.rows / 2));
	//		return ROI[0];
	//	}
	//}

	random_device rn;
	mt19937_64 generator(rn());
	uniform_int_distribution<int> distribution(0, ROI.size());
	int random = distribution(generator);
	return ROI[random];
}

// 원래 있던 GT에 새로 추가된 위험물 영상의 GT를 추가
void ThreatImageAugment::AddGT(IG* src, IG threat, Point2f top_left) {

	GT new_GT;	
	GT zero_GT = { 0,0,0,0,0 };
	
	new_GT.Nclass = threat.GroundTruth[0].Nclass;
	new_GT.width = (float)(threat.Img_low.cols) / (float)(src->Img_low.cols);
	new_GT.height = (float)(threat.Img_low.rows) / (float)(src->Img_low.rows);
	new_GT.x = (float)(top_left.x + (float)(threat.Img_low.cols) / 2.0) / (float)(src->Img_low.cols);
	new_GT.y = (float)(top_left.y + (float)(threat.Img_low.rows) / 2.0) / (float)(src->Img_low.rows);

	if (src->GroundTruth[0] == zero_GT) 
		src->GroundTruth[0] = new_GT;
	else 
		src->GroundTruth.push_back(new_GT);
}

// float type(0 ~ 1)인 Input 영상(IG)를 normalize해서 ushort type(0 ~ 65535)으로 변환(inverse normalize)
IG ThreatImageAugment::InvNormalize(IG src) {

	Mat InvNormMat_low, InvNormMat_high;
	IG output;

	src.Img_low.copyTo(InvNormMat_low);
	src.Img_high.copyTo(InvNormMat_high);

	InvNormMat_low = InvNormMat_low * maxVal;
	InvNormMat_high = InvNormMat_high * maxVal;

	InvNormMat_low.convertTo(InvNormMat_low, CV_16UC1);
	InvNormMat_high.convertTo(InvNormMat_high, CV_16UC1);

	output.Img_low = InvNormMat_low;
	output.Img_high = InvNormMat_high;
	output.GroundTruth = src.GroundTruth;

	return output;
}

// 합성한 영상과 GT를 txt, png 형태로 dst 폴더에 저장
void ThreatImageAugment::SaveIG(IG src, string filename) {
	/*string Resulttxt = "dst" + backslash + "Result_txt" + backslash + filename + extension_txt;
	string GTName = "dst" + backslash + "Result_GT" + backslash + filename + extense_GT + extension_txt;
	string LowImg = "dst" + backslash + "Result_png" + backslash + filename + "_low.png";
	string HighImg = "dst" + backslash + "Result_png" + backslash + filename + "_high.png";
	string LowTiff = "dst" + backslash + "Result_tiff" + backslash + filename + "_low.tiff";
	string HighTiff = "dst" + backslash + "Result_tiff" + backslash + filename + "_high.tiff";*/
	//string Resulttxt = "dst" + backslash + "Result_txt" + backslash + filename + extension_txt;

	string GTName = dst + backslash + filename + "_high" + extension_txt;
	string LowImg = dst + backslash + filename + "_low.png";
	string HighImg = dst + backslash + filename + "_high.png";
	//string LowTiff = "dst" + backslash + "Result_tiff" + backslash + filename + "_low.tiff";
	//string HighTiff = "dst" + backslash + "Result_tiff" + backslash + filename + "_high.tiff";

	//ofstream Result_txt(Resulttxt);

		//ushort *data_low = (ushort *)src.Img_low.data;
		//ushort *data_high = (ushort *)src.Img_high.data;

		//Result_txt << src.Img_low.rows;
		//Result_txt << '	';
		//Result_txt << src.Img_low.cols;
		//Result_txt << endl;

		//for (int r = 0; r < src.Img_low.rows; r++) {
		//	for (int c = 0; c < src.Img_low.cols; c++)
		//		Result_txt << data_low[r * src.Img_low.cols + c] << ' ';
		//	Result_txt << endl;
		//	for (int c = 0; c < src.Img_high.cols; c++)
		//		Result_txt << data_high[r * src.Img_high.cols + c] << ' ';
		//	Result_txt << endl;
		//}
		//Result_txt.close();

	imwrite(LowImg, src.Img_low);
	imwrite(HighImg, src.Img_high);
	/*imwrite(LowTiff, src.Img_low);
	imwrite(HighTiff, src.Img_high);*/

	ofstream AugImgGT(GTName);

		for (int n = 0; n < src.GroundTruth.size(); n++) 
			AugImgGT << src.GroundTruth[n].Nclass - 1 << ' ' << src.GroundTruth[n].x << ' ' << src.GroundTruth[n].y << ' ' << src.GroundTruth[n].width << ' ' << src.GroundTruth[n].height << endl;

	AugImgGT.close();
}

IG ThreatImageAugment::Reduce_boxsize(IG Threat) {
	return Threat;
}