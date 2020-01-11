#pragma once
#define COMPILER_MSVC
#define NOMINMAX
#include<string.h>
#include<stdlib.h>
#include<map>
#include<vector>
#include<list>
#include<fstream>
#include<iostream>
#include<sstream>
#include<thread>
//#include <unistd.h>//usleep

#include <string>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <utility>
#include <thread>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include<direct.h>
#include<io.h>
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session.h"

//#include "platform.h"

//#include "speech.h"

//time mkdir
//#include <sys/time.h>
#include <sys/stat.h>
#include <chrono>
using namespace std::chrono;

using namespace std;
using namespace cv;


#define _WIN32 
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#elif _LINUX
#include <stdarg.h>
#include <sys/stat.h>
#endif

#ifdef _WIN32
#define ACCESS _access
#define MKDIR(a) _mkdir((a))
#elif _LINUX
#define ACCESS access
#define MKDIR(a) mkdir((a),0755)
#endif


typedef uint64_t millitime_t;

enum CardType { COMMON_CARD = 0, BIG_CHARACTER_CARD = 1 };

namespace std
{
	template <typename T>
	std::string to_string(T value)
	{
		std::ostringstream os;
		os << value;
		return os.str();
	}
}

#ifdef MY_JNI

#include <jni.h>
#include <android/log.h>
#define LOG_TAG "AlgDealer"
#define myprint(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace std
{
	template <typename T>
	std::string to_string(T value)
	{
		std::ostringstream os;
		os << value;
		return os.str();
	}

}
#endif


#ifdef MY_IOS
#define myprint(...)
#endif



/*
struct MsecMat
{
	millitime_t time;
	Mat image;
};
*/

struct IdxMat
{
	int idx;
	Mat image;
	millitime_t time;

};


struct SPokerAlgParam
{
	SPokerAlgParam()
	{
		debug_save_mode = 0;

	}

	int GetCameraPoistion()
	{
		int _camera_position;
		mutex_camera_position.lock();
		_camera_position = camera_position;
		mutex_camera_position.unlock();
		return _camera_position;

	}
	void SetCameraPoistion(int _camera_position)
	{
		mutex_camera_position.lock();
		camera_position = _camera_position;
		mutex_camera_position.unlock();

	}

	int GetCameraDistaine()
	{
		int _camera_distance;
		mutex_camera_distance.lock();
		_camera_distance = camera_distance;
		mutex_camera_distance.unlock();
		return _camera_distance;

	}
	void SetCameraDistance(int _camera_distance)
	{
		mutex_camera_distance.lock();
		camera_distance = _camera_distance;
		mutex_camera_distance.unlock();

	}

	int need_joker;
	int result_mode;
	int mult_card_take_mode;
	int mult_card_take_mumb;

	int card_type;

	int button_camera_mode;
	int raise_mode;


	int debug_save_mode;
private:
	Mutex mutex_camera_position;
	int camera_position;
	Mutex mutex_camera_distance;
	int camera_distance;





};




int GetSpeech(string& speech);

void PutSpeech(string speech);

void ClearSpeech();

void SetCameraDistance(int distance);

void SetCameraPosition(int position);

bool PokerInit(const int type, const string read_path, const string write_path);

bool Start(SPokerAlgParam poker_alg_param);

bool Stop();

int GetResult();

int Process(Mat frame, unsigned long long timestamp);

void SetInningImmedPushResultFlag(bool _immed_push_result);



void CalibrateCamera(IdxMat& idx_img);

int CalibrateComb(vector<int>& vec_comb_status);

bool CalibrateJudgeBegin(Mat& frame, vector<Rect>& vec_org_extn_rect);

bool GetIsCalibBegin(int mode);

bool DebugStart(int save_mode);

bool DebugStop(int save_mode);



//debug_info

void DebugInfoSavePath(string write_path);

void DebugResetCollectStatus(bool collectStatus);

bool DebugGetExtnAchieveTargetNumb();

void DebugSetExtnAchieveTargetNumb(bool _is_achieve_numb);


void DebugIsNeedSaveImg(int isNeedSaveImgMode);