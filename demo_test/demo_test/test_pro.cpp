#include "test_pro.h"
#define WindowsTime // windows上用的时间
#define pi 3.14159

bool s_flag=false;
int j_count=1;
bool e_flag = false;

std::string g_writepath;
cv::CascadeClassifier num_cascade;
cv::CascadeClassifier flower_cascade;


tensorflow::Session *p_session;
tensorflow::Session *p2_session;

mutex input_pic_lock;
mutex l_extract_recogn_lock;
mutex l_extract_inside_cls_lock;
mutex g_res_lock;
mutex g_front_lock;

mutex l_extract_inside_cls_rect_lock;
mutex l_extract_inside_cls_num_lock;

struct MsecMat
{	
	int p_id;
	std::string p_time;
	std::string res_dirname;
	cv::Mat image;
};


struct InfoNumFlower {
	int p_id;
	cv::Mat p_obj;
	vector<std::string> nfinfovec;
	int nf_count;
};
//存的每一帧结果 hongk meiQ fang5
vector<InfoNumFlower> ObjNumFlowerVec;

// 前一帧结果
vector<string> front_poker_vec;

list<MsecMat> PicInfoVec;


list<Rect> gobal_numflowerpokerslist;
//vector<int> gobal_numbersvec;

list<Rect> gobal_res_numflowerpokerslist;
vector<int> gobal_res_numbersvec;


// 打印信息
void myprint(std::string a) {
	std::cout << "-----"<< a << std::endl;
}

// 结构化输出时间
std::string GetFormatTime()
{
	namespace chrono = std::chrono;
	auto time_now = chrono::system_clock::now();
	auto duration_in_ms = chrono::duration_cast<chrono::milliseconds>(time_now.time_since_epoch());
	auto ms_part = duration_in_ms - chrono::duration_cast<chrono::seconds>(duration_in_ms);

	tm local_time_now;
	time_t raw_time = chrono::system_clock::to_time_t(time_now);
	#ifdef WindowsTime
	//windwos用
	localtime_s(&local_time_now, &raw_time);
	#else
	// linux用
	localtime_r(&raw_time, &local_time_now);
	#endif 

	
	
	//cout << std::put_time(&local_time_now, "%Y%m%d %H:%M:%S,") << std::setfill('0') << std::setw(3) << ms_part.count() << "\n";

	char sTime[256];
	memset(sTime, 0, 256);
	sprintf_s(sTime, "%04d-%02d-%02d-%02d-%02d-%02d-%03lld",
		local_time_now.tm_year + 1900,
		local_time_now.tm_mon + 1,
		local_time_now.tm_mday,
		local_time_now.tm_hour,
		local_time_now.tm_min,
		local_time_now.tm_sec,
		ms_part.count());
	//printf("%s\n",sTime);

	return std::string(sTime);
}

// 单步时间
/*
inline long long single_time_cost() {
	struct  timeval beginTime = { 0,0 };
	gettimeofday(&beginTime, NULL);
	long long tm_begin = beginTime.tv_sec * 1000 + beginTime.tv_usec / 1000;
	return tm_begin;
}
*/


// 结果组合
string NFRes(int num, int flower)
{
	string characternum, characterflower;
	switch (num)
	{
	case 1:
		characternum = "A";
		break;
	case 2:
		characternum = "2";
		break;
	case 3:
		characternum = "3";
		break;
	case 4:
		characternum = "4";
		break;
	case 5:
		characternum = "5";
		break;
	case 6:
		characternum = "6";
		break;
	case 7:
		characternum = "7";
		break;
	case 8:
		characternum = "8";
		break;
	case 9:
		characternum = "9";
		break;
	case 10:
		characternum = "10";
		break;
	case 11:
		characternum = "J";
		break;
	case 12:
		characternum = "Q";
		break;
	case 13:
		characternum = "K";
		break;
	}

	switch (flower)
	{
	case 1:
		characterflower = "hei";
		break;
	case 2:
		characterflower = "hong";
		break;
	case 3:
		characterflower = "mei";
		break;
	case 4:
		characterflower = "fang";
		break;
	}
	return characterflower + characternum;
}

// 开启计算图
bool CreateGraph(std::string model_path, int flag = 1)
{
	tensorflow::GraphDef graph;
	//tensorflow::Session  tmpsession;
	tensorflow::Status ret = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph);
	if (!ret.ok())
	{
		myprint("模型加载失败");
		return false;
	}
	if (flag == 1)
	{
		NewSession(tensorflow::SessionOptions(), &p_session);
		ret = p_session->Create(graph);
	}
	else
	{
		NewSession(tensorflow::SessionOptions(), &p2_session);
		ret = p2_session->Create(graph);
	}


	if (!ret.ok())
	{
		return false;
	}
	return true;
}

// 分类结果
int CalcClassify(Mat transrc, int whsize, int flag = 1)
{

	int image_width = whsize;  //32
	int image_height = whsize; //32
	int image_channel = 1;
	int image_batch = 1;
	//ReadPicToVec(img_path, image_height,image_width);
	//cv::Mat src = imread(img_path, 0);
	cv::Mat src = transrc.clone();
	//Mat dst(Size(image_width,image_height),CV_8UC1,Scalar::all(0));
	//myprint("奔溃了");
	//cv::imshow("src", src);
	//cv::waitKey(0);
	cv::Mat dst(image_height, image_width,CV_8UC1);
	cv::resize(src, dst, Size(image_width, image_height), 0, 0, INTER_LINEAR);
	if (dst.empty())
	{
		myprint("奔溃了");
	}
	
	//imshow("123", dst);
	//waitKey(0);
	int row = dst.rows;
	int col = dst.cols*dst.channels();
	//std::cout << row << "," << col << std::endl;
	vector<float> src_data;
	for (int i = 0; i < row; ++i)
	{
		uchar *data = dst.ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			src_data.push_back((float(data[j]) / 255 - 0.5));
		}
	}

	/*
	std::vector<float> src_data;
	for (size_t i = 0; i < image_width*image_height*image_batch*image_channel; i++)
	{
	src_data.push_back(0.5);
	}
	*/

	tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ image_batch,image_height,image_width,image_channel }));
	std::copy_n(src_data.begin(), src_data.size(), image_tensor.flat<float>().data());
	const std::string input_layer_name = "input:0";
	const std::string output_layer_name = "MobilenetV1/Predictions/Reshape_1:0";

	std::vector < std::pair<std::string, tensorflow::Tensor>> inputs;
	inputs.push_back(std::pair <std::string, tensorflow::Tensor>(input_layer_name, image_tensor));
	std::vector<tensorflow::Tensor> outputs;
	tensorflow::Status run_status;
	if (flag == 1)
	{
		run_status = p_session->Run(inputs, { output_layer_name }, {}, &outputs);

	}
	else
	{
		run_status = p2_session->Run(inputs, { output_layer_name }, {}, &outputs);
	}

	if (!run_status.ok())
	{
		//tensorflow::LogAllRegisteredKernels();
		std::cout << "tensorflow running model failed \n" << std::endl;
		//return 0;
	}


	tensorflow::Tensor *ts_out = &outputs[0];
	const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned> prediction = ts_out->flat<float>();

	//std::cout << "predict:" << std::endl;

	float MaxScore = 0.0;
	int MaxLoc = -1;
	for (size_t i = 0; i < prediction.size(); i++)
	{
		//cout << prediction(i) << " ";
		//printf("%e ", prediction(i));

		if (prediction(i) > MaxScore)
		{
			MaxScore = prediction(i);
			MaxLoc = i;
		}
	}

	return MaxLoc;
}

// 初始化
bool PokerInit(const int type, const std::string read_path, const std::string write_path)
{
	g_writepath = write_path;	
	std::string numxmlpath = read_path + "/20191226_num_cascade.xml";
	std::string flowerxmlpath = read_path + "/20191226_flower_cascade.xml";
	std::string numpbpath = read_path + "/20191228_num_14_frozen_graph.pb";
	//std::string flowerpbpath = read_path + "/20191226_flower_5_frozen_graph.pb";
	std::string flowerpbpath = "E:\\Classify_Data\\Models\\20200109_poker_flower_2424_gray_cls_5_mobivenet\\20200111_flower_5_frozen_graph.pb";
	if (!num_cascade.load(numxmlpath)) {
		myprint("numxmlpath loading fail");
		return false;
	}
	else {
		myprint("numxmlpath loading success");
	}

	if (!flower_cascade.load(flowerxmlpath)) {
		myprint("flowerxmlpath loading fail");
		return false;
	}
	else {
		myprint("flowerxmlpath loading success");
	}

	bool numflag = CreateGraph(numpbpath, 1);
	bool flowerflag = CreateGraph(flowerpbpath, 2);

	if (!numflag)
	{
		myprint("numpbpath load fail");
		return false;
	}
	if (!flowerflag)
	{
		myprint("flowerflag load fail");
		return false;
	}
	myprint("--------numpbpath load success---------");
	myprint("---------flowerflag load success----------");
	return true;
}

// 二值化
bool drawContPic(Mat &frame)
{
	Mat src = frame.clone();
	int maxnum = 0;
	int thres_index = 0;
	vector<int> thresold_num = { 65,85,100,115,135 };
	//cv::imshow("frame", src);
	//cv::waitKey(0);
	int num_c = 0;
	int innum_max = -1;
	int innum_count = 0;
	for (size_t i = 0; i<5; i++)
	{
		Mat dst(Size(540, 360), CV_8SC1, Scalar(0));
		resize(src, dst, Size(540, 360), 0, 0);
		thres_index = thresold_num.at(i);
		threshold(dst, dst, thres_index, 255, THRESH_BINARY_INV);

		//cv::imshow("二值化", dst);
		//cv::waitKey(0);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(dst, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

		Mat result;

		
		//cout << hierarchy.size() << endl;
		for (int i = 0; i < hierarchy.size(); i++)
		{
			Rect rect = boundingRect(contours.at(i));
			//if(rect.area()>300 and rect.area()<1200)
			if (rect.area()>64 && rect.area()<625)
			{
				if ((rect.height / rect.width)<2 && (rect.width / rect.height)<2)
				{
					if (rect.width>8 && rect.height>8)
					{	
						//cv::rectangle(dst, rect, Scalar(255), 3, 8);
						num_c++;
					}
				}
			}
		}
		//cv::imshow("rect ", dst);
		//cv::waitKey(0);
		/*
		if (num_c >= 30)
		{
			return true;
		}
		*/
		if (num_c >= innum_max)
		{
			innum_max = num_c;
		}
		if (num_c > 30)
		{
			innum_count+=1;
		}
		if (innum_count >= 2)
		{
			return true;
		}
	}
	return false;
}

// 过程
int Process(Mat frame, unsigned long long time_stamp)
{	
	if (s_flag)
	{	
		Mat per_frame = frame.clone();
		if (drawContPic(per_frame))
		{
			MsecMat PicInfo;
			PicInfo.image = per_frame;
			PicInfo.p_id = j_count;
			PicInfo.p_time = GetFormatTime();
			PicInfo.res_dirname = to_string(time_stamp);
			
			input_pic_lock.lock();
			//允许传入的多少张
			if (PicInfoVec.size() <=300)
			{				
				PicInfoVec.push_back(PicInfo); //一个写操作
				j_count += 1;
			}
			input_pic_lock.unlock();
			
		}
	}
	return 1;
}

// 获取中心点
Point RetPoint(Rect &BoxRect) {
	Point CenterPoint = Point(BoxRect.x + BoxRect.width / 2, BoxRect.y + BoxRect.height / 2);
	return CenterPoint;
}

//计算三点之间的夹角
float AngleDegree(Point srcp1, Point dstp1, Point dstp2)
{
	float angle = 180 * acos(((srcp1.x - dstp1.x)*(srcp1.x - dstp1.x) + (srcp1.y - dstp1.y)*(srcp1.y - dstp1.y) + (srcp1.x - dstp2.x)*(srcp1.x - dstp2.x) + (srcp1.y - dstp2.y)*(srcp1.y - dstp2.y) - (dstp1.x - dstp2.x)*(dstp1.x - dstp2.x) - (dstp1.y - dstp2.y)*(dstp1.y - dstp2.y)) / (2 * (sqrt((srcp1.x - dstp1.x)*(srcp1.x - dstp1.x) + (srcp1.y - dstp1.y)*(srcp1.y - dstp1.y))*sqrt((srcp1.x - dstp2.x)*(srcp1.x - dstp2.x) + (srcp1.y - dstp2.y)*(srcp1.y - dstp2.y))))) / pi;
	return angle;
}

//计算俩点之间的距离
double disBetweenPoint(Point SrcCenterPoint, Point DstCenterPoint)
{
	double dis = sqrt((SrcCenterPoint.y - DstCenterPoint.y)*(SrcCenterPoint.y - DstCenterPoint.y) + (SrcCenterPoint.x - DstCenterPoint.x)*(SrcCenterPoint.x - DstCenterPoint.x));
	return dis;
}

//斜率对应的角度
float TanAngleDegree(Point a, Point b)
{
	if ((a.x - b.x) == 0)
	{
		return 90;
	}
	float angle = atan(double(a.y - b.y) / double(a.x - b.x))*float(180) / pi;
	return angle;
}

// 创建多级目录
int CreatDir(char *pDir)
{	
	try
	{
		//char *pDir = (char*)path.c_str();
		int i = 0;
		int iRet;
		int iLen;
		char* pszDir;

		if (NULL == pDir)
		{
			return 0;
		}

		//pszDir = strdup(pDir);
		//strcpy(pszDir,pDir);
		pszDir = pDir;
		iLen = strlen(pszDir);

		// 创建中间目录
		for (i = 0; i < iLen; i++)
		{
			if (pszDir[i] == '\\' || pszDir[i] == '/')
			{
				pszDir[i] = '\0';

				//如果不存在,创建
				iRet = ACCESS(pszDir, 0);
				if (iRet != 0)
				{
					iRet = MKDIR(pszDir);
					if (iRet != 0)
					{
						return -1;
					}
				}
				//支持linux,将所有\换成/
				pszDir[i] = '/';
			}
		}

		iRet = MKDIR(pszDir);
		free(pszDir);
		return iRet;
	}
	catch (const std::exception &e)
	{
		std::cout << e.what() << std::endl;
	}
	
	
}


// 存放图片
void RestorePic(MsecMat &tmppicinfo, vector<Rect> &numvec_commonpokers, vector<Rect> &flowervec_commonpokers, std::string param, std::string str_text, std::string k_str_text)
{	
	Mat frame = tmppicinfo.image;
	Mat temp_frame_copy = frame.clone();
	Mat temp_frame;
	cv::cvtColor(temp_frame_copy, temp_frame, cv::COLOR_GRAY2BGR);
	std::string name = GetFormatTime();
	std::string tmp = name.substr(0, name.find_last_of("-"));
	std::string tmp2 = tmp.substr(0, tmp.find_last_of("-"));
	std::string dirname = g_writepath + "/" + "/detect/"+tmppicinfo.res_dirname;
	// 创建保存图片的文件夹

	#ifdef WindowsTime
	if (_access(dirname.c_str(), 0) == -1)
	{
		//cout << dirname << " is not existing" << endl;
		int flag = _mkdir(dirname.c_str());
	}
	#else
	if (0 != access(dirname.c_str(), 0))
	{
		mkdir(dirname.c_str());
}
	#endif // 

	
	std::string res_path = dirname + "/" +name+"_"+param + "_picture" + ".jpg";

	for (size_t i = 0; i < numvec_commonpokers.size(); i++)
	{
		Rect SrcRextBox = numvec_commonpokers[i];
		rectangle(temp_frame, SrcRextBox, Scalar(0, 255, 0), 1, 8);
	}

	for (size_t j = 0; j < flowervec_commonpokers.size(); j++)
	{
		Rect SrcRextBox = flowervec_commonpokers[j];
		rectangle(temp_frame, SrcRextBox, Scalar(0, 0, 255), 1, 8);
	}
	
	putText(temp_frame, str_text, Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 1, 8);
	putText(temp_frame, k_str_text, Point(50, 100), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 1, 8);

	/*
	if (top_point.x != 0 && top_point.y != 0)
	{
		rectangle(temp_frame, Rect(top_point.x, top_point.y, bottom_point.x - top_point.x, bottom_point.y - top_point.y), Scalar(0, 255, 255), 1, 8);
	}
	*/

	imwrite(res_path, temp_frame);

}

//判断
bool JudgeEmpty(MsecMat &stmpframe)
{
	input_pic_lock.lock();
	if (!PicInfoVec.empty())
	{
		stmpframe = PicInfoVec.front();
		PicInfoVec.pop_front();
	}
	input_pic_lock.unlock();

	if ((stmpframe.image).empty())
	{
		return false;
	}
	else
	{
		return true;
	}
}

// 排序
void SortCommon(vector<Rect> &vec_commonpokers) {
	for (size_t i = 0; i<vec_commonpokers.size(); i++)
	{
		for (size_t j = i; j<vec_commonpokers.size(); j++)
		{
			//int srcx = vec_commonpokers[i].x;
			//int dstx = vec_commonpokers[j].x;
			int srcx = RetPoint(vec_commonpokers[i]).x;
			int dstx = RetPoint(vec_commonpokers[j]).x;
			if (srcx > dstx)
			{
				Rect a;
				a = vec_commonpokers[i];
				vec_commonpokers[i] = vec_commonpokers[j];
				vec_commonpokers[j] = a;
			}
		}
	}
}

// 删除相同位置
void DeleteRowCommon(vector<Rect> &vec_commonpokers, vector<int> &list_numpokers)
{
	map<int, int> in_isExistVec;
	vector<Rect> vec_commonpokers_copy;
	vector<int> list_numpokers_copy;
	for (size_t j = 0; j < vec_commonpokers.size(); j++)
	{
		in_isExistVec[j] = 1;
	}

	for (size_t i = 0; i < vec_commonpokers.size(); i++)
	{
		Rect SrcBoxRect = vec_commonpokers[i];
		Point SrcCenterPoint = RetPoint(SrcBoxRect);
		double srcps = SrcBoxRect.area();
		if (in_isExistVec[i])
		{
			for (size_t j = 0; j < vec_commonpokers.size(); j++)
			{
				if (i != j)
				{
					Rect DstBoxRect = vec_commonpokers[j];
					Point DstCenterPoint = RetPoint(DstBoxRect);
					Rect IntersectRect = SrcBoxRect&DstBoxRect;
					double dstps = DstBoxRect.area();
					double ps = 0;
					ps = IntersectRect.area();
					double dis = disBetweenPoint(SrcCenterPoint, DstCenterPoint);
					bool inside_flag = false;
					inside_flag = SrcBoxRect.contains(DstCenterPoint);

					if (ps > ((SrcBoxRect.width / 2)*(DstBoxRect.width / 2)))
					{	
						if (srcps > dstps)
						{
							in_isExistVec[j] = 0;
						}
						else
						{
							in_isExistVec[i] = 0;
						}		
					}
					else if (dis < (SrcBoxRect.width*(0.6)))
					{
						//in_isExistVec[j] = 0;
						if (srcps > dstps)
						{
							in_isExistVec[j] = 0;
						}
						else
						{
							in_isExistVec[i] = 0;
						}
					}
					else if (inside_flag)
					{
						//in_isExistVec[j] = 0;
						if (srcps > dstps)
						{
							in_isExistVec[j] = 0;
						}
						else
						{
							in_isExistVec[i] = 0;
						}
					}
				}
			}
		}
	}

	for (size_t k = 0; k < vec_commonpokers.size(); k++)
	{
		if (in_isExistVec[k])
		{
			vec_commonpokers_copy.push_back(vec_commonpokers[k]);
			list_numpokers_copy.push_back(list_numpokers[k]);
		}
	}
	vec_commonpokers.clear();
	list_numpokers.clear();
	vec_commonpokers.assign(vec_commonpokers_copy.begin(), vec_commonpokers_copy.end());
	list_numpokers.assign(list_numpokers_copy.begin(), list_numpokers_copy.end());
}

void DeleteRowCommon_edit2(vector<Rect> &vec_commonpokers, vector<int> &list_numpokers)
{
	map<int, int> in_isExistVec;
	vector<Rect> vec_commonpokers_copy;
	vector<int> list_numpokers_copy;

	for (size_t j = 0; j < vec_commonpokers.size(); j++)
	{
		in_isExistVec[j] = 1;
	}

	for (size_t i = 0; i < vec_commonpokers.size(); i++)
	{
		Rect SrcBoxRect = vec_commonpokers[i];
		Point SrcCenterPoint = RetPoint(SrcBoxRect);
		if (in_isExistVec[i])
		{
			for (size_t j = 0; j < vec_commonpokers.size(); j++)
			{
				if (i != j)
				{
					Rect DstBoxRect = vec_commonpokers[j];
					Point DstCenterPoint = RetPoint(DstBoxRect);
					Rect IntersectRect = SrcBoxRect&DstBoxRect;
					double ps = 0;
					ps = IntersectRect.area();
					double dis = disBetweenPoint(SrcCenterPoint, DstCenterPoint);
					bool inside_flag = false;
					inside_flag = SrcBoxRect.contains(DstCenterPoint);
					if (ps > ((SrcBoxRect.width / 2)*(DstBoxRect.width / 2)))
					{
						in_isExistVec[j] = 0;
					}
					else if (dis < (SrcBoxRect.width*(0.6)))
					{
						in_isExistVec[j] = 0;
					}
					else if (inside_flag)
					{
						in_isExistVec[j] = 0;
					}
				}
			}
		}
	}

	for (size_t k = 0; k < vec_commonpokers.size(); k++)
	{
		if (in_isExistVec[k])
		{
			vec_commonpokers_copy.push_back(vec_commonpokers[k]);
			list_numpokers_copy.push_back(list_numpokers[k]);
		}
	}
	vec_commonpokers.clear();
	list_numpokers.clear();
	vec_commonpokers.assign(vec_commonpokers_copy.begin(), vec_commonpokers_copy.end());
	list_numpokers.assign(list_numpokers_copy.begin(), list_numpokers_copy.end());
}


// 合并边框
void MergeBorder(vector<Rect> &vec_commonpokers, vector<int> &list_numpokers) {

	map<int, int> in_isExistVec;
	//pair<int, Rect> tmppair;
	vector<Rect> vec_commonpokers_copy;
	vector<int> list_numpokers_copy;
	for (size_t j = 0; j < vec_commonpokers.size(); j++)
	{
		in_isExistVec[j] = 1; // 合并哪俩个
	}

	for (size_t i = 0; i < vec_commonpokers.size(); i++)
	{
		Rect SrcBoxRect = vec_commonpokers[i];
		Point SrcCenterPoint = RetPoint(SrcBoxRect);
		double srcps = SrcBoxRect.area();
		vector<int > vec_sameblank;
		
		if (in_isExistVec[i])
		{	
			vec_sameblank.push_back(i);
			// i 比较每一个j 标记与i相同的位置
			// 取出每一个rect 得出xmin，ymin，xmax，ymax
			for (size_t j = 0; j < vec_commonpokers.size(); j++)
			{
				if (i != j)
				{
					Rect DstBoxRect = vec_commonpokers[j];
					Point DstCenterPoint = RetPoint(DstBoxRect);
					double dstps = DstBoxRect.area();
					Rect IntersectRect = SrcBoxRect&DstBoxRect;
					double ps = 0;
					ps = IntersectRect.area();
					double dis = disBetweenPoint(SrcCenterPoint, DstCenterPoint);
					bool inside_flag = false;
					inside_flag = SrcBoxRect.contains(DstCenterPoint);
					
					//if(ps>0.25)


					#if 1
					if (ps > ((SrcBoxRect.width / 2)*(DstBoxRect.width / 2)))
					{
						in_isExistVec[j] = 0;
						vec_sameblank.push_back(j);
					}
					else if (dis < (SrcBoxRect.width*(0.6)))
					{
						in_isExistVec[j] = 0;
						vec_sameblank.push_back(j);
					}
				
					else if (inside_flag)
					{
						in_isExistVec[j] = 0;
						vec_sameblank.push_back(j);
					}
					#endif			
				}
			}
			
			if (vec_sameblank.size() >= 2)
			{
				int x_max_num = 0;
				int x_min_num = 2000;
				int y_max_num = 0;
				int y_min_num = 2000;
				for (size_t k = 0; k < vec_sameblank.size(); k++)
				{	
					int kz = vec_sameblank[k];
					if (vec_commonpokers[kz].x>x_max_num)
					{
						x_max_num = vec_commonpokers[kz].x;
					}
					if (vec_commonpokers[kz].x<x_min_num)
					{
						x_min_num = vec_commonpokers[kz].x;
					}
					if (vec_commonpokers[kz].y>y_max_num)
					{
						y_max_num = vec_commonpokers[kz].y;
					}
					if (vec_commonpokers[kz].y<y_min_num)
					{
						y_min_num = vec_commonpokers[kz].y;
					}
				}
				Rect recttmp;

				recttmp=Rect(x_min_num, y_min_num, x_max_num - x_min_num, y_max_num - y_min_num);
				cout << recttmp.tl()<<" "<<recttmp.br() << endl;
				if (recttmp.width > 15 && recttmp.height > 15)
				{	
					
					vec_commonpokers_copy.push_back(recttmp);
				}
				

			 }
			else if(vec_sameblank.size()==1)
			{	
				vec_commonpokers_copy.push_back(vec_commonpokers[i]);
			}			
		}
	}

	vec_commonpokers.clear();
	vec_commonpokers.assign(vec_commonpokers_copy.begin(), vec_commonpokers_copy.end());
	/*
	for (size_t k = 0; k < vec_commonpokers.size(); k++)
	{
		if (in_isExistVec[k])
		{
			vec_commonpokers_copy.push_back(vec_commonpokers[k]);
			list_numpokers_copy.push_back(list_numpokers[k]);
		}
	}
	vec_commonpokers.clear();
	list_numpokers.clear();
	vec_commonpokers.assign(vec_commonpokers_copy.begin(), vec_commonpokers_copy.end());
	list_numpokers.assign(list_numpokers_copy.begin(), list_numpokers_copy.end());
	*/
}

//删除远距离的边框
#if 0
void DeleteDisCommon(vector<Rect> &vec_commonpokers, int matchnum)
{
	for (size_t j = 0; j < vec_commonpokers.size(); )
	{
		int w;
		if (j == (vec_commonpokers.size() - 1))
		{
			w = j - 1;
		}
		else
		{
			w = j + 1;
		}

		Rect SrcRectBox = vec_commonpokers[j];
		Rect SrcNextRectBox = vec_commonpokers[w];

		//Point SrcCenterPoint = Point(SrcRectBox.x + SrcRectBox.width / 2, SrcRectBox.y + SrcRectBox.height / 2);
		Point SrcCenterPoint = RetPoint(SrcRectBox);
		//Point SrcNextCenterPoint = Point(SrcNextRectBox.x + SrcNextRectBox.width / 2, SrcNextRectBox.y + SrcNextRectBox.height / 2);
		Point SrcNextCenterPoint = RetPoint(SrcNextRectBox);
		double S_D_Distance = 0;
		//S_D_Distance= sqrt((SrcCenterPoint.x - SrcNextCenterPoint.x)*(SrcCenterPoint.x - SrcNextCenterPoint.x) + (SrcCenterPoint.y - SrcNextCenterPoint.y)*(SrcCenterPoint.y - SrcNextCenterPoint.y));
		S_D_Distance = disBetweenPoint(SrcCenterPoint, SrcNextCenterPoint);
		if (0 < j && j < vec_commonpokers.size() - 1)
		{
			Rect SrcFrontRectBox = vec_commonpokers[j - 1];
			//Point SrcFrontCenterPoint = Point(SrcFrontRectBox.x + SrcFrontRectBox.width / 2, SrcFrontRectBox.y + SrcFrontRectBox.height / 2);
			Point SrcFrontCenterPoint = RetPoint(SrcFrontRectBox);;
			//double S_D_Distance2 = sqrt((SrcCenterPoint.x - SrcFrontCenterPoint.x)*(SrcCenterPoint.x - SrcFrontCenterPoint.x) + (SrcCenterPoint.y - SrcFrontCenterPoint.y)*(SrcCenterPoint.y - SrcFrontCenterPoint.y));
			double S_D_Distance2 = disBetweenPoint(SrcCenterPoint, SrcFrontCenterPoint);
			if (S_D_Distance >(SrcRectBox.width * matchnum) && S_D_Distance2>(SrcRectBox.width * matchnum))
			{
				vec_commonpokers.erase(vec_commonpokers.begin() + j);
			}
			else
			{
				j++;
			}
		}
		else
		{
			if (S_D_Distance > SrcRectBox.width * matchnum)
			{
				vec_commonpokers.erase(vec_commonpokers.begin() + j);
			}
			else
			{
				j++;
			}
		}
	}
}
#endif 

//删除远距离边框_edi2
void DeleteDisCommon(vector<Rect> &vec_commonpokers, int matchnum)
{

	map<string, int> in_isExistVec;
	vector<Rect> vec_commonpokers_copy;
	bool in_isFlag;
	for (size_t j = 0; j < vec_commonpokers.size(); j++)
	{
		in_isExistVec[to_string(j)] = 0;
	}
	for (size_t j = 0; j < vec_commonpokers.size(); j++)
	{
		in_isFlag = false;

		Rect SrcRectBox = vec_commonpokers[j];
		Point SrcCenterPoint = RetPoint(SrcRectBox);
		for (size_t k = 0; k < vec_commonpokers.size(); k++)
		{
			if (j != k)
			{
				Rect DstRectBox = vec_commonpokers[k];
				Point DstCenterPoint = RetPoint(DstRectBox);
				double S_D_Distance;
				S_D_Distance = disBetweenPoint(SrcCenterPoint, DstCenterPoint);

				if (S_D_Distance < ((SrcRectBox.width)*matchnum))
				{
					in_isFlag = true;
					break;
				}
			}
		}

		if (in_isFlag)
		{
			in_isExistVec[to_string(j)] = 1;
		}
	}


	for (size_t j = 0; j < vec_commonpokers.size(); j++)
	{
		if (in_isExistVec[to_string(j)] != 0)
		{
			vec_commonpokers_copy.push_back(vec_commonpokers.at(j));
		}
	}

	vec_commonpokers.clear();
	vec_commonpokers.assign(vec_commonpokers_copy.begin(), vec_commonpokers_copy.end());

}

void DeleteDisCommon_edit2(vector<Rect> &vec_commonpokers, vector<int> &list_numpokers, int matchnum)
{

	map<string, int> in_isExistVec;
	vector<Rect> vec_commonpokers_copy;
	vector<int> list_numpokers_copy;
	bool in_isFlag;
	for (size_t j = 0; j < vec_commonpokers.size(); j++)
	{
		in_isExistVec[to_string(j)] = 0;
	}
	for (size_t j = 0; j < vec_commonpokers.size(); j++)
	{
		in_isFlag = false;

		Rect SrcRectBox = vec_commonpokers[j];
		Point SrcCenterPoint = RetPoint(SrcRectBox);
		for (size_t k = 0; k < vec_commonpokers.size(); k++)
		{
			if (j != k)
			{
				Rect DstRectBox = vec_commonpokers[k];
				Point DstCenterPoint = RetPoint(DstRectBox);
				double S_D_Distance;
				S_D_Distance = disBetweenPoint(SrcCenterPoint, DstCenterPoint);

				if (S_D_Distance < ((SrcRectBox.width)*matchnum))
				{
					in_isFlag = true;
					break;
				}
			}
		}

		if (in_isFlag)
		{
			in_isExistVec[to_string(j)] = 1;
		}
	}


	for (size_t j = 0; j < vec_commonpokers.size(); j++)
	{
		if (in_isExistVec[to_string(j)] != 0)
		{
			vec_commonpokers_copy.push_back(vec_commonpokers.at(j));
			if (list_numpokers.size()>0)
			{
				list_numpokers_copy.push_back(list_numpokers[j]);
			}
		}
	}

	vec_commonpokers.clear();
	vec_commonpokers.assign(vec_commonpokers_copy.begin(), vec_commonpokers_copy.end());
	if (list_numpokers.size()>0)
	{
		list_numpokers.clear();
		list_numpokers.assign(list_numpokers_copy.begin(), list_numpokers_copy.end());
	}


}

//读取每一个边框
bool readRecognpic(list<Rect> &vec_commonpokers_copy, Rect &tmpRect)
{
	int tmpflag = 0;
	l_extract_recogn_lock.lock();
	if (!vec_commonpokers_copy.empty())
	{
		tmpRect = vec_commonpokers_copy.front();
		vec_commonpokers_copy.pop_front();
		tmpflag = 1;
	}
	l_extract_recogn_lock.unlock();

	if (!tmpflag)
	{
		return false;
	}
	return true;
}

bool readRecognpic_copy(Rect &tmpRect)
{
	int tmpflag = 0;
	l_extract_recogn_lock.lock();
	if (!gobal_numflowerpokerslist.empty())
	{
		tmpRect = gobal_numflowerpokerslist.front();
		gobal_numflowerpokerslist.pop_front();
		tmpflag = 1;
	}
	l_extract_recogn_lock.unlock();

	if (!tmpflag)
	{
		return false;
	}
	return true;
}

// 分类多线程
void multthreadRecogn(MsecMat &tmppicinfo, list<Rect>& vec_commonpokers_copy, vector<Rect>& tmp_vec_commonpokers, vector<int>& list_commonpokers, int flag = 1)
{
	Mat frame_gray = tmppicinfo.image;
	while (1)
	{
		Rect in_tmpRect;
		if (readRecognpic(vec_commonpokers_copy, in_tmpRect))
		{	
			// classnumber 分类的结果 数字:0 1 2 3 *** 13 花色:0-4
			int classnumber = 0;
			Mat copy_frame = frame_gray.clone();
			Rect tmp_a = in_tmpRect;


			// 扩充4个像素
			/*
			if(flag==1)
			{
			tmp_a = tmp_a +Point(-2,-2);
			tmp_a = tmp_a + Size(4,4);
			}
			*/
			Mat tmpframe = copy_frame(tmp_a);
			string belongcls = "";
			if (flag == 1)
			{
				belongcls = "num";
				classnumber = CalcClassify(tmpframe, 32, flag);
			}
			else {
				belongcls = "flower";
				classnumber = CalcClassify(tmpframe, 24, flag);
			}
			// classnumber=0的话就不要
			if (classnumber != 0)
			{
				l_extract_inside_cls_lock.lock();
				list_commonpokers.push_back(classnumber);
				tmp_vec_commonpokers.push_back(in_tmpRect);
				l_extract_inside_cls_lock.unlock();
			}
			//string picname = g_writepath+"";
			string timename = GetFormatTime();
			string tmp = timename.substr(0, timename.find_last_of("-"));
			string tmp2 = tmp.substr(0, tmp.find_last_of("-"));
			string dirname = g_writepath  +"/class/"+ tmppicinfo.res_dirname;
			//string dirname2 = g_writepath + "/" + tmppicinfo.res_dirname;
			#ifdef WindowsTime
			if (_access(dirname.c_str(), 0) == -1)
			{
				//cout << dirname << " is not existing" << endl;
				int flag = _mkdir(dirname.c_str());
			}
			#else
			if (0 != access(dirname.c_str(), 0))
			{
				mkdir(dirname.c_str());
			}
			#endif // 
			string picname = dirname + "/(" + to_string(tmppicinfo.p_id) + ")_" + belongcls + "[" + to_string(classnumber) + "]_" + timename + ".jpg";
			//保存每一个边框的识别结果 
			imwrite(picname, tmpframe);
		}
		else
		{
			break;
		}

	}

}

void multthreadRecogn_copy(MsecMat &tmppicinfo,  int flag = 1)
{
	Mat frame_gray = tmppicinfo.image;
	while (1)
	{
		Rect in_tmpRect;
		if (readRecognpic_copy(in_tmpRect))
		{
			// classnumber 分类的结果 数字:0 1 2 3 *** 13 花色:0-4
			int classnumber = 0;
			Mat copy_frame = frame_gray.clone();
			Rect tmp_a = in_tmpRect;


			// 扩充4个像素
			/*
			if(flag==1)
			{
			tmp_a = tmp_a +Point(-2,-2);
			tmp_a = tmp_a + Size(4,4);
			}
			*/
			Mat tmpframe = copy_frame(tmp_a);
			string belongcls = "";
			if (flag == 1)
			{
				belongcls = "num";
				classnumber = CalcClassify(tmpframe, 32, flag);
			}
			else {
				belongcls = "flower";
				classnumber = CalcClassify(tmpframe, 24, flag);
			}
			// classnumber=0的话就不要
			if (classnumber != 0)
			{
				l_extract_inside_cls_rect_lock.lock();
				gobal_res_numbersvec.push_back(classnumber);
				l_extract_inside_cls_rect_lock.unlock();

				l_extract_inside_cls_num_lock.lock();
				gobal_res_numflowerpokerslist.push_back(in_tmpRect);
				l_extract_inside_cls_num_lock.unlock();
			}
			//string picname = g_writepath+"";
			string timename = GetFormatTime();
			string tmp = timename.substr(0, timename.find_last_of("-"));
			string tmp2 = tmp.substr(0, tmp.find_last_of("-"));
			string dirname = g_writepath + "/class/" + tmppicinfo.res_dirname;
			//string dirname2 = g_writepath + "/" + tmppicinfo.res_dirname;
#ifdef WindowsTime
			if (_access(dirname.c_str(), 0) == -1)
			{
				//cout << dirname << " is not existing" << endl;
				int flag = _mkdir(dirname.c_str());
			}
#else
			if (0 != access(dirname.c_str(), 0))
			{
				mkdir(dirname.c_str());
			}
#endif // 
			string picname = dirname + "/(" + to_string(tmppicinfo.p_id) + ")_" + belongcls + "[" + to_string(classnumber) + "]_" + timename + ".jpg";
			//保存每一个边框的识别结果 
			imwrite(picname, tmpframe);
		}
		else
		{
			break;
		}
	}
}

//开多线程识别
void StartRecogn(MsecMat &tmppicinfo, vector<Rect> &vec_commonpokers, vector<int> &list_commonpokers, int flag = 1)
{
	Mat frame_gray = (tmppicinfo.image).clone();
	vector<Rect>::iterator it;
	vector<Rect> tmp_vec_commonpokers;
#if 1
	list<Rect> inside_vec_commonpokers;
	for (size_t vec_i = 0; vec_i<vec_commonpokers.size(); vec_i++)
	{
		inside_vec_commonpokers.push_back(vec_commonpokers[vec_i]);
	}

	// 注意ref
	thread in_t1(multthreadRecogn, std::ref(tmppicinfo), std::ref(inside_vec_commonpokers), std::ref(tmp_vec_commonpokers), std::ref(list_commonpokers), flag);
	thread in_t2(multthreadRecogn, std::ref(tmppicinfo), std::ref(inside_vec_commonpokers), std::ref(tmp_vec_commonpokers), std::ref(list_commonpokers), flag);
	thread in_t3(multthreadRecogn, std::ref(tmppicinfo), std::ref(inside_vec_commonpokers), std::ref(tmp_vec_commonpokers), std::ref(list_commonpokers), flag);
	thread in_t4(multthreadRecogn, std::ref(tmppicinfo), std::ref(inside_vec_commonpokers), std::ref(tmp_vec_commonpokers), std::ref(list_commonpokers), flag);
	thread in_t5(multthreadRecogn, std::ref(tmppicinfo), std::ref(inside_vec_commonpokers), std::ref(tmp_vec_commonpokers), std::ref(list_commonpokers), flag);

	in_t1.join();
	in_t2.join();
	in_t3.join();
	in_t4.join();
	in_t5.join();



	for (size_t i = 0; i<tmp_vec_commonpokers.size(); i++)
	{
		for (size_t j = 0; j<tmp_vec_commonpokers.size(); j++)
		{
			int srcx = RetPoint(tmp_vec_commonpokers[i]).x;
			int dstx = RetPoint(tmp_vec_commonpokers[j]).x;
			if (srcx < dstx)
			{
				Rect a;
				a = tmp_vec_commonpokers[i];
				tmp_vec_commonpokers[i] = tmp_vec_commonpokers[j];
				tmp_vec_commonpokers[j] = a;

				int b;
				b = list_commonpokers[i];
				list_commonpokers[i] = list_commonpokers[j];
				list_commonpokers[j] = b;
			}
		}
	}

	vec_commonpokers.clear();
	vec_commonpokers.assign(tmp_vec_commonpokers.begin(), tmp_vec_commonpokers.end());

#endif

}

void StartRecogn_copy(MsecMat &tmppicinfo, vector<Rect> &vec_commonpokers, vector<int> &list_commonpokers, int flag = 1)
{
	Mat frame_gray = (tmppicinfo.image).clone();
	vector<Rect>::iterator it;
	vector<Rect> tmp_vec_commonpokers;
	


#if 1
	//list<Rect> inside_vec_commonpokers;
	for (size_t vec_i = 0; vec_i<vec_commonpokers.size(); vec_i++)
	{
		gobal_numflowerpokerslist.push_back(vec_commonpokers[vec_i]);
	}
	

	// 注意ref
	thread in_t1(multthreadRecogn_copy, std::ref(tmppicinfo), flag);
	thread in_t2(multthreadRecogn_copy, std::ref(tmppicinfo), flag);
	thread in_t3(multthreadRecogn_copy, std::ref(tmppicinfo), flag);
	thread in_t4(multthreadRecogn_copy, std::ref(tmppicinfo), flag);
	thread in_t5(multthreadRecogn_copy, std::ref(tmppicinfo), flag);

	in_t1.join();
	in_t2.join();
	in_t3.join();
	in_t4.join();
	in_t5.join();

	tmp_vec_commonpokers.assign(gobal_res_numflowerpokerslist.begin(), gobal_res_numflowerpokerslist.end());

	for (size_t i = 0; i<tmp_vec_commonpokers.size(); i++)
	{
		for (size_t j = 0; j<tmp_vec_commonpokers.size(); j++)
		{
			int srcx = RetPoint(tmp_vec_commonpokers[i]).x;
			int dstx = RetPoint(tmp_vec_commonpokers[j]).x;
			if (srcx < dstx)
			{
				Rect a;
				a = tmp_vec_commonpokers.at(i);
				tmp_vec_commonpokers.at(i) = tmp_vec_commonpokers.at(j);
				tmp_vec_commonpokers.at(j) = a;

				int b;
				b = gobal_res_numbersvec.at(i);
				gobal_res_numbersvec.at(i) = gobal_res_numbersvec.at(j);
				gobal_res_numbersvec.at(j) = b;
			}
		}
	}

	
	vec_commonpokers.clear();
	vec_commonpokers.assign(tmp_vec_commonpokers.begin(), tmp_vec_commonpokers.end());

	list_commonpokers.clear();
	list_commonpokers.assign(gobal_res_numbersvec.begin(), gobal_res_numbersvec.end());

	
	gobal_res_numbersvec.clear();	
	gobal_res_numflowerpokerslist.clear();



#endif
}

// 转正6，9做识别
int Single69Recogn(Mat &RectNum, double angle)
{
	Mat tmp_RectNum = RectNum.clone();
	Mat dst_warpRotateScale;
	Point center = Point(tmp_RectNum.cols / 2, tmp_RectNum.rows / 2);

	Mat M2 = getRotationMatrix2D(center, angle, 1);

	// 仿射变换
	cv::warpAffine(tmp_RectNum, dst_warpRotateScale, M2, Size(tmp_RectNum.cols, tmp_RectNum.rows), INTER_LINEAR);
	int num = CalcClassify(dst_warpRotateScale, 32, 1);
	return num;
}

//组合数字和花色
void CombineCommon(MsecMat &tmppicinfo, vector<Rect> &numvec_pokers, vector<Rect> &flowervec_pokers, vector<int> &list_numpokers, vector<int> &list_flowerpokers)
{
	Mat srcframe = tmppicinfo.image;
	Mat tmpframe = srcframe.clone();
	vector<pair<int, int>> nf_pair;
	//vector<double> nf_kangle;
	vector<float> nf_kangle;
	vector<int> tmpflag_flower(flowervec_pokers.size());

	vector<Rect> incopy_numvec_pokers;
	vector<Rect> incopy_flowervec_pokers;

	int numpoker_len = numvec_pokers.size();
	int flowerpoker_len = flowervec_pokers.size();

	float front_k = 0.0;
	for (size_t i = 0; i < numvec_pokers.size(); i++)
	{
		Rect NumRectBox;
		NumRectBox = numvec_pokers.at(i);
		Point NumCenterPoint;
		NumCenterPoint = RetPoint(NumRectBox);
		int min_loc = 0;
		int sec_loc = 0;
		float min_dis = 1000.0;
		float sec_dis = 1000.0;
		float min_k1 = -10;
		float sec_k2 = -10;
		bool isUsedSec = false;
		for (size_t j = 0; j < flowervec_pokers.size(); j++)
		{
			if (tmpflag_flower[j] != 1)
			{
				Rect FlowerRectBox;
				FlowerRectBox = flowervec_pokers.at(j);
				Point FlowerCenterPoint;
				FlowerCenterPoint = RetPoint(FlowerRectBox);
				float point_distance = 0.0;
				//point_distance= sqrt((NumCenterPoint.x - FlowerCenterPoint.x)*(NumCenterPoint.x - FlowerCenterPoint.x) + (NumCenterPoint.y - FlowerCenterPoint.y)*(NumCenterPoint.y - FlowerCenterPoint.y));
				point_distance = (float)disBetweenPoint(NumCenterPoint, FlowerCenterPoint);
				if (point_distance < min_dis)
				{
					min_dis = point_distance;
					min_loc = j;
					min_k1 = abs(TanAngleDegree(NumCenterPoint, FlowerCenterPoint));
				}
			}
		}

		for (size_t j = 0; j < flowervec_pokers.size(); j++)
		{
			if (tmpflag_flower[j] != 1)
			{
				Rect FlowerRectBox;
				FlowerRectBox = flowervec_pokers.at(j);
				Point FlowerCenterPoint;
				FlowerCenterPoint = RetPoint(FlowerRectBox);
				float point_distance = 0.0;
				//point_distance= sqrt((NumCenterPoint.x - FlowerCenterPoint.x)*(NumCenterPoint.x - FlowerCenterPoint.x) + (NumCenterPoint.y - FlowerCenterPoint.y)*(NumCenterPoint.y - FlowerCenterPoint.y));
				point_distance = disBetweenPoint(NumCenterPoint, FlowerCenterPoint);

				if (point_distance<sec_dis && point_distance > min_dis)
				{
					if (point_distance<1.3*NumRectBox.width)
					{
						isUsedSec = true;
						sec_dis = point_distance;
						sec_loc = j;
						sec_k2 = abs(TanAngleDegree(NumCenterPoint, FlowerCenterPoint));
					}
				}
			}
		}


		// 保证踢掉最远的框
		if (min_dis > 1.5*NumRectBox.width)
		{
			continue;
		}

#if 1
		//if (min_dis <= 1.2*NumRectBox.width)
		{
			if (!isUsedSec)
			{
				pair<int, int> tmppair;
				tmppair = make_pair(i, min_loc);
				tmpflag_flower[min_loc] = 1;
				nf_pair.push_back(tmppair);
				nf_kangle.push_back(min_k1);
				//myprint("添加了1000的序号:i=%d,j=%d",i,min_loc);
			}
			else
			{
				//myprint("1----------min loc:%d , min dis:%.2f ,k1:%.2f, sec loc:%d ,sec len:%.2f ,k2:%.2f",min_loc,min_dis,min_k1,sec_loc,sec_dis,sec_k2);
				Rect FlowerRectBox_o1 = flowervec_pokers.at(min_loc);
				Point FlowerCenterPoint_o1 = RetPoint(FlowerRectBox_o1);

				Rect FlowerRectBox_o2 = flowervec_pokers.at(sec_loc);
				Point FlowerCenterPoint_o2 = RetPoint(FlowerRectBox_o2);

				float NumTwoFlowerDgree = 0.0;
				NumTwoFlowerDgree = AngleDegree(NumCenterPoint, FlowerCenterPoint_o1, FlowerCenterPoint_o2);
				//myprint("2------------i=%d,min loc:%d,sec loc:%d,度数:%.2f",i,min_loc,sec_loc,NumTwoFlowerDgree);
				if (NumTwoFlowerDgree >= 20)
				{
					int next_i = 10000;
					if (i == (numpoker_len - 1))
					{
						int tran_a = i;
						next_i = tran_a - 1;
					}
					else
					{
						int tran_a = i;
						next_i = tran_a + 1;
					}

					Rect NumBoxRectNext = numvec_pokers[next_i];
					Point NextNumCenterPoint = RetPoint(NumBoxRectNext);
					float ps1 = 0.0;
					ps1 = AngleDegree(NumCenterPoint, NextNumCenterPoint, FlowerCenterPoint_o1);
					float ps2 = 0.0;
					ps2 = AngleDegree(NumCenterPoint, NextNumCenterPoint, FlowerCenterPoint_o2);
					pair<int, int> tmppair;
					// myprint("3-----------度数1:%.2f, 度数2：%.2f",ps1,ps2);
					if (ps2 >= ps1)
					{
						tmppair = make_pair(i, sec_loc);
						tmpflag_flower[sec_loc] = 1;

						nf_pair.push_back(tmppair);
						nf_kangle.push_back(sec_k2);
						// myprint("添加了的序号:i=%d,j=%d",i,sec_loc);
					}
					else
					{
						tmppair = make_pair(i, min_loc);
						tmpflag_flower[min_loc] = 1;
						nf_pair.push_back(tmppair);
						nf_kangle.push_back(min_k1);
						//myprint("添加了的序号:i=%d,j=%d",i,min_loc);
					}
				}
				else
				{
					pair<int, int> tmppair;
					tmppair = make_pair(i, min_loc);
					tmpflag_flower[min_loc] = 1;
					nf_pair.push_back(tmppair);
					nf_kangle.push_back(min_k1);
					// myprint("<30 添加了的序号:i=%d,j=%d",i,min_loc);
				}
			}
		}

			
		}
		
#endif

	


#if 1
	InfoNumFlower pinfo;
	pinfo.p_obj = tmpframe;
	pinfo.nf_count = numvec_pokers.size();
	pinfo.p_id = tmppicinfo.p_id;
	//pinfo.nfslope.assign(nf_kangle.begin(),nf_kangle.end());
#endif

	//myprint("第%d张识别结果", tmppicinfo.p_id);
	//myprint("第" + to_string(tmppicinfo.p_id) + "张识别结果：");
	string str_text = "";
	
	for (size_t k = 0; k < nf_pair.size(); k++)
		{	
			incopy_numvec_pokers.push_back(numvec_pokers[nf_pair[k].first]);
			incopy_flowervec_pokers.push_back(flowervec_pokers[nf_pair[k].second]);

			int recgon_num = 0;
			int modify_k = 0;
			if (list_numpokers[nf_pair[k].first] == 6 || list_numpokers[nf_pair[k].first] == 9)
			{
				if (abs(nf_kangle[k]) <= 30)
				{
					Point num_p = RetPoint(numvec_pokers[nf_pair[k].first]);
					Point flower_p = RetPoint(flowervec_pokers[nf_pair[k].second]);
					Rect tmp_a = numvec_pokers[nf_pair[k].first];
					tmp_a = tmp_a + Point(-3, -3);
					tmp_a = tmp_a + Size(7, 5);
					Mat tmp_RectNum = srcframe(tmp_a);

					if (num_p.x < flower_p.x)
					{
						recgon_num = Single69Recogn(tmp_RectNum, -60);
						if (recgon_num != 0)
						{
							modify_k = k;
						}
					}
					else if (num_p.x > flower_p.x)
					{
						recgon_num = Single69Recogn(tmp_RectNum, 60);
						if (recgon_num != 0)
						{
							modify_k = k;
						}
					}
				}
			}

			if (recgon_num != 0)
			{
				list_numpokers[nf_pair[modify_k].first] = recgon_num;
			}
			//myprint("对应序号 %d，，%d",list_numpokers[nf_pair[k].first],list_flowerpokers[nf_pair[k].second]);
			string s1 = NFRes(list_numpokers[nf_pair[k].first], list_flowerpokers[nf_pair[k].second]);
			//cout << s1 << endl;
			//myprint("结果%s",s1.c_str());
			pinfo.nfinfovec.push_back(s1);
			str_text = str_text + s1 + "  ";
		}
	

	string k_str_text = "";

	for (size_t m = 0; m<nf_kangle.size(); m++)
	{
		k_str_text = k_str_text + to_string(int(nf_kangle[m])) + " ";
	}

	//myprint("第%d张:%s", tmppicinfo.p_id,str_text.c_str());
	//myprint("第" + to_string(tmppicinfo.p_id) + "张:" + str_text);


#if 1
	g_res_lock.lock();
	if (pinfo.nfinfovec.size() >= 13)
	{
		ObjNumFlowerVec.push_back(pinfo);
		g_res_lock.unlock();
	}
	else {
		g_res_lock.unlock();
	}
#endif

	// numvec和poker需要改
	RestorePic(tmppicinfo, incopy_numvec_pokers, incopy_flowervec_pokers, "last_" + to_string(tmppicinfo.p_id), str_text, k_str_text);

}

// 下一轮与上一轮对比
bool CurrentNextResMatch(vector<string> &current_poker_vec) {

	int same_count = 0;
	for (size_t i = 0; i<front_poker_vec.size(); i++)
	{
		for (size_t j = 0; j<current_poker_vec.size(); j++)
		{
			if (front_poker_vec.at(i) == current_poker_vec.at(j))
			{
				same_count++;
				if (same_count >= 8)
				{
					return true;
				}
			}
		}
	}
	return false;
}

// 合并3张结果
bool MergeMultipleImagesRes(vector<InfoNumFlower> &ObjNumFlowerVecCopy) {

	int maxlen = 0;
	for (size_t i = 0; i < ObjNumFlowerVecCopy.size(); i++)
	{
		if (ObjNumFlowerVecCopy[i].nfinfovec.size() > maxlen)
		{
			maxlen = ObjNumFlowerVecCopy[i].nfinfovec.size();
		}
	}

	vector<vector<string>> colcls(maxlen);
	for (size_t i = 0; i < ObjNumFlowerVecCopy.size(); i++)
	{
		for (size_t j = 0; j < ObjNumFlowerVecCopy[i].nfinfovec.size(); j++)
		{
			colcls[j].push_back(ObjNumFlowerVecCopy[i].nfinfovec[j]);
		}
	}



#if 1
	vector<string> everyturnresvec;
	for (size_t i = 0; i < colcls.size(); i++)
	{
		map<string, int> percolclsmap;
		int maxnum = 0;
		string maxloc;

		for (size_t j = 0; j < colcls[i].size(); j++)
		{
			percolclsmap[colcls[i][j]]++;

			if (percolclsmap[colcls[i][j]] > maxnum && percolclsmap[colcls[i][j]] >= 2)
			{
				maxnum = percolclsmap[colcls[i][j]];
				maxloc = colcls[i][j];
			}
		}

		if (percolclsmap[maxloc] >= 2)
		{
			everyturnresvec.push_back(maxloc);
		}
	}


	string str_text = "";
	for (size_t k = 0; k < everyturnresvec.size(); k++)
	{
		str_text += everyturnresvec[k] + " ";
	}

	g_front_lock.lock();
	if (front_poker_vec.empty())
	{
		front_poker_vec.assign(everyturnresvec.begin(), everyturnresvec.end());
		g_front_lock.unlock();
		if (everyturnresvec.size() >= 13)
		{
			myprint("3张图片的结果------------------------\n");
			//myprint("%s\n", str_text.c_str());
			myprint(str_text);
			//time_cost("one_turn", 1);
			//speech_broadcast_file((char*)"duang");
		}

	}
	else {
		// 当前与上一个结果的对比
		bool local_flag = CurrentNextResMatch(everyturnresvec);
		if (!local_flag)
		{
			if (everyturnresvec.size() >= 13)
			{
				front_poker_vec.clear();
				front_poker_vec.assign(everyturnresvec.begin(), everyturnresvec.end());
				myprint("3张图片的结果------------------------\n");
				//myprint("%s\n", str_text.c_str());
				myprint(str_text);
				//time_cost("one_turn", 1);
				//speech_broadcast_file((char*)"duang");
			}
		}
		g_front_lock.unlock();
	}

	return true;

#endif

}

void DetectCls()
{
	while (s_flag)
	{	
		MsecMat picinfo;
		if(JudgeEmpty(picinfo))
		{	
			myprint("流程开启--------------------------");
			//long long start_time1 = single_time_cost();
			
			// 截取的小图片 数字和花色
			std::vector<Rect> numpokers;
			std::vector<Rect> flowerpokers;
			
			// 截取的小图片 对应分类 数字 0-13  花色 0-4 (其中0表示数字，花色中的负样本)
			// 其他 数字对应法则 1-A 2-2 3-3 ****11-J 12-Q 13-K 
			// 花色对应法则 1-黑桃 2-红桃 3-梅花 4方块
			std::vector<int> list_numpokers;
			std::vector<int> list_flowerpokers;
			
			Mat roi_srcframe;
			roi_srcframe = picinfo.image.clone();
			std::string filepath = picinfo.res_dirname;

			// 开启检测
			num_cascade.detectMultiScale(roi_srcframe, numpokers, 1.1, 1, 1, Size(25, 25), Size(45, 45));  // 缩放尺寸
			flower_cascade.detectMultiScale(roi_srcframe, flowerpokers, 1.1, 1, 1, Size(18, 18), Size(32, 32)); //缩放尺寸要改
			
			//long long end_time1 = single_time_cost();
			// 至少13张才进入
			if (numpokers.size() >= 13 && flowerpokers.size() >= 13)
			{
				// 给数字或花色按左上角x大小排序
				SortCommon(numpokers);
				SortCommon(flowerpokers);
				RestorePic(picinfo, numpokers, flowerpokers, "start" + to_string(picinfo.p_id), "", "");
				myprint("1111111111");

				myprint("删除前" + to_string(numpokers.size()));
				myprint(to_string(flowerpokers.size()));
				
				//MergeBorder(numpokers, list_numpokers);
				//MergeBorder(flowerpokers, list_flowerpokers);

				

				// 删除远距的方框
				DeleteDisCommon_edit2(numpokers,list_numpokers, 3);
				DeleteDisCommon_edit2(flowerpokers,list_flowerpokers, 3);

				RestorePic(picinfo, numpokers, flowerpokers, "delete" + to_string(picinfo.p_id), "", "");


				//保存图片
				// 参1:结构体对象 参2:数字框 参3:花色框 参4:对应第几张图片 参5:数字分类结果，6花色分类结果
				

				// 打印
				myprint("删除后"+to_string(numpokers.size()));
				myprint(to_string(flowerpokers.size()));

				myprint("222222222");
				
				// 开启数字 花色识别
				StartRecogn_copy(picinfo, numpokers, list_numpokers, 1);
				StartRecogn_copy(picinfo, flowerpokers, list_flowerpokers, 2);
				myprint("3333333333333");
				std::string paramnum = "";
				std::string paramflower = "";
				for (int i = 0; i < list_numpokers.size(); i++)
				{
					paramnum += to_string(list_numpokers.at(i)) + " ";
				}
				for (int j = 0; j < list_flowerpokers.size(); j++)
				{
					paramflower += to_string(list_flowerpokers.at(j)) + " ";
				}
				//保存识别完的结果图片
				RestorePic(picinfo, numpokers, flowerpokers, "middle" + to_string(picinfo.p_id), paramnum, paramflower);
				
				#if 1
				//DeleteRowCommon(numpokers, list_numpokers);
				//DeleteRowCommon(flowerpokers, list_flowerpokers);

				MergeBorder(numpokers, list_numpokers);
				MergeBorder(flowerpokers, list_flowerpokers);

				DeleteDisCommon_edit2(numpokers,list_numpokers, 3);
				DeleteDisCommon_edit2(flowerpokers, list_flowerpokers,3);
				#endif 

				RestorePic(picinfo, numpokers, flowerpokers, "deletetwo" + to_string(picinfo.p_id), "", "");

				if (numpokers.size() >= 13 && flowerpokers.size() >= 13)
				{	
					// 组合数字和花色
					CombineCommon(picinfo, numpokers, flowerpokers, list_numpokers, list_flowerpokers);
				}

			}
		}

		if (!e_flag)
		{
			g_res_lock.lock();
			int ObjNumFlowerCount = ObjNumFlowerVec.size();
			if (ObjNumFlowerCount >= 3)
			{
				vector<InfoNumFlower> ObjNumFlowerVecCopy;
				ObjNumFlowerVecCopy.assign(ObjNumFlowerVec.begin(), ObjNumFlowerVec.end());
				g_res_lock.unlock();
				// 三张出一个结果
				e_flag = MergeMultipleImagesRes(ObjNumFlowerVecCopy);
				s_flag = false;

				input_pic_lock.lock();
				PicInfoVec.clear();
				input_pic_lock.unlock();

				g_res_lock.lock();
				ObjNumFlowerVec.clear();
				g_res_lock.unlock();

			}
			else
			{
				g_res_lock.unlock();
			}
		}
	}
}

bool Start(SPokerAlgParam poker_alg_param) {

	thread t1(DetectCls);
	t1.join();
	return 0;
}

void click_start()
{	
	// xml pb 模型所在路径
	std::string read_path = "E:\\Android_Data\\PokerDealer\\AndroidEyesDealer\\app\\src\\main\\assets\\jni";
	// 保存图片的路径
	std::string write_path = "E:\\first_edition_add_20191211-cut-pic\\others";
	//初始化 xml pb模型的对象
	bool tmpa=PokerInit(0, read_path, write_path);
	
	s_flag = true;
	// 加载图片
	//std::string filepath = "E:\\cut_pic\\20191023pics\\2019-10-22-14-45-08";
	// 加载图片
	std::string filepath = "E:\\cut_pic\\20191130test\\2019-12-16-09-44-26";
	vector<cv::String> filesvec;

	cv::glob(filepath, filesvec);

	for (int i = 0; i < filesvec.size(); i++)
	{
		cv::Mat frame = cv::imread(filesvec[i],0);
		if (!frame.data)
		{
			continue;
		}
		else
		{
			// 开启图片的传入 参1 picobj 参2时间戳 （我这边随便起的）
			int tmpb = Process(frame, 478);
		}
		
	}
	SPokerAlgParam poker_alg_param;
	//开启线程读取
	int ctmp=Start(poker_alg_param); 
}


void Statistics_res()
{
	vector<string> flower = {"hei","hong","mei","fang"};
	vector<string> num = {"A","2","3","4","5","6","7","8","9","10","J","Q","K"};
	
	map<string, int> tmpmap;
	vector<map<string, int> >vec_map_numflower;
	for (int i = 0; i < flower.size(); i++)
	{	
		for (int j = 0; j < num.size(); j++)
		{	
			string nf="";
			nf = flower.at(i) + num.at(j);
			tmpmap[nf] = 0;
			vec_map_numflower.push_back(tmpmap);
		}
	}

}


int main() {
	std::string time_sj = GetFormatTime();
	click_start();
	std::string time_sj2 = GetFormatTime();
	std::cout << time_sj << std::endl;
	std::cout << time_sj2 << std::endl;

	//std::string time_sj = GetFormatTime();
	//std::cout << time_sj << std::endl;
	/*
	string path = "E:\\first_edition_add_20191211-cut-pic\\liyang\\a\\b\\c";
	char *pDir = (char*)path.c_str();
	int flag=CreatDir(pDir);

	std::cout << flag << std::endl;
	*/
	system("pause");
}


bool DebugStart(int save_mode) { return 0; }
void DebugResetCollectStatus(bool collectStatus) {}
bool DebugGetExtnAchieveTargetNumb() { return false; }
void DebugSetExtnAchieveTargetNumb(bool _is_achieve_numb) {}
void DebugIsNeedSaveImg(int isNeedSaveImgMode) {}
void DebugInfoSavePath(std::string write_path) {}
bool CalibrateJudgeBegin(Mat& frame, vector<Rect>& vec_org_extn_rect) { return false; }
bool DebugStop(int save_mode) { return 0; }
void SetCameraPosition(int position) {}
void SetCameraDistance(int distance) {}
void SetInningImmedPushResultFlag(bool _immed_push_result) {}
void SetInningEnd(bool _inning_end) {}
int GetResult() { return 0; }
void CalibrateCamera(IdxMat& idx_img) {}
int CalibrateComb(vector<int>& vec_comb_status) { return 1; }
bool GetIsCalibBegin(int mode) { return false; }
int GetSpeech(std::string& speech) { return 0; }