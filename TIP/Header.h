#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>

#define maxVal 65535
#define PI 3.141592
#define rad PI/180
#define extense_GT "_GT"
#define extension_txt ".txt"
#define extension_png ".png"
#define extension_tiff ".tiff"
#define threshold_value 55000

#define HEIGHT 1552
#define WIDTH 3000

// user parameter
// Vol_N_Den 함수의 변환 범위
#define Volume_max 1.2
#define Volume_min 0.8
#define Density_max 1.15
#define Density_min 0.85

// dst name
#define dst "200721train"

using namespace cv;
using namespace std;

// Ground Truth를 포함하는 struct로 순서대로 class 번호(int), center x좌표(float), center y좌표(float), ROI width(float), ROI height(float)로 구성되어 있습니다.
struct GT {
	int Nclass;
	float x;
	float y;
	float width;
	float height;

	bool operator==(const GT& a) const {
		return(Nclass == a.Nclass && x == a.x && y == a.y && width == a.width && height == a.height);
	};

	GT& operator=(const GT &a)
	{
		Nclass = a.Nclass;
		x = a.x;
		y = a.y;
		width = a.width;
		height = a.height;
		return *this;
	};
};

// low-energy 영상, high-energy 영상과 GT type vector로 구성된 struct입니다.
struct IG {
	Mat Img_low;
	Mat Img_high;
	vector<GT> GroundTruth;

	IG& operator=(const IG &a)
	{
		a.Img_low.copyTo(Img_low);
		a.Img_high.copyTo(Img_high);
		GroundTruth = a.GroundTruth;
		return *this;
	};
};

class ThreatImageAugment {
public:
	void _run(int Index_BG, int nRst);

	// READIMG의 flags
	enum READIMG_flags{
		READIMG_PNG,				// PNG image를 읽어 옵니다.
		READIMG_TXT_SEPERATE,		// low-energy 영상, high-energy 영상을 각각 다른 파일에서 읽어 옵니다.
		READIMG_TXT_UNITE,			// low-energy 영상, high-energy 영상을 하나의 파일에서 읽어 옵니다.
		READIMG_TIFF				// TIFF image를 읽어 옵니다.
	};

private:

	// user parameter			  
	// class의 name & number로 class를 추가하시려면 아래있는 모든 배열의 크기를 늘리고 추가하시면 됩니다.
	// 특정 class의 number는 해당 class 이름의 (배열 Class에서의) index입니다.
	string Class[5] = { "background", "knife", "gun", "rifle", "explosive" };

	// 각 class 폴더에 들어 있는 위험물 & 배경 영상의 수입니다.
	// No_per_Class[0]은 배경 영상의 갯수이고
	// No_per_Class[n]은 class number가 n인 위험물 영상의 갯수입니다.
	int No_per_Class[5] = { 11, 101, 57, 19, 0 };

	// 하나의 배경 영상에 합성할 class 별 위험물 영상의 갯수입니다.
	// ChoiceNo[0]은 의미X,
	// ChoiceNo[n]은 각 class number가 n일때 합성하고 싶은 위험물 영상의 갯수입니다.
	int ChoiceNo[5] = { NULL, 2, 1, 1, 0 };
	
	string backslash = "\\";

	IG ReadImg(string filename, int classnum, int type);
		Mat Norm(Mat src);
		Mat Saturation(Mat src); 
		IG MakeIG(Mat low, Mat high, int classnum);
	IG Vol_N_Den(IG Threat);
	IG Rotation(IG Threat);
		int RndAngle();
	IG Translation(IG src, IG Threat);
		Point2i RndPoint(IG BG, IG Threat);
		void AddGT(IG* src, IG Threat, Point2f top_left);
	IG InvNormalize(IG src);
	void SaveIG(IG src, string filename);
};