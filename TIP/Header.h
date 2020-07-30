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
// Vol_N_Den �Լ��� ��ȯ ����
#define Volume_max 1.2
#define Volume_min 0.8
#define Density_max 1.15
#define Density_min 0.85

// dst name
#define dst "200721train"

using namespace cv;
using namespace std;

// Ground Truth�� �����ϴ� struct�� ������� class ��ȣ(int), center x��ǥ(float), center y��ǥ(float), ROI width(float), ROI height(float)�� �����Ǿ� �ֽ��ϴ�.
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

// low-energy ����, high-energy ����� GT type vector�� ������ struct�Դϴ�.
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

	// READIMG�� flags
	enum READIMG_flags{
		READIMG_PNG,				// PNG image�� �о� �ɴϴ�.
		READIMG_TXT_SEPERATE,		// low-energy ����, high-energy ������ ���� �ٸ� ���Ͽ��� �о� �ɴϴ�.
		READIMG_TXT_UNITE,			// low-energy ����, high-energy ������ �ϳ��� ���Ͽ��� �о� �ɴϴ�.
		READIMG_TIFF				// TIFF image�� �о� �ɴϴ�.
	};

private:

	// user parameter			  
	// class�� name & number�� class�� �߰��Ͻ÷��� �Ʒ��ִ� ��� �迭�� ũ�⸦ �ø��� �߰��Ͻø� �˴ϴ�.
	// Ư�� class�� number�� �ش� class �̸��� (�迭 Class������) index�Դϴ�.
	string Class[5] = { "background", "knife", "gun", "rifle", "explosive" };

	// �� class ������ ��� �ִ� ���蹰 & ��� ������ ���Դϴ�.
	// No_per_Class[0]�� ��� ������ �����̰�
	// No_per_Class[n]�� class number�� n�� ���蹰 ������ �����Դϴ�.
	int No_per_Class[5] = { 11, 101, 57, 19, 0 };

	// �ϳ��� ��� ���� �ռ��� class �� ���蹰 ������ �����Դϴ�.
	// ChoiceNo[0]�� �ǹ�X,
	// ChoiceNo[n]�� �� class number�� n�϶� �ռ��ϰ� ���� ���蹰 ������ �����Դϴ�.
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