#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <ctime>
#include <sstream>
#include <time.h>
#include <unistd.h>
#include <termios.h>
#include <signal.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/face.hpp"
//#include <opencv2\core\core.hpp>
//#include<opencv2\face\facerec.hpp>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <sys/time.h>


using namespace cv;
using namespace cv::face;
using namespace std;

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // depth of framebuffer
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
};

cv::Mat frame;
cv::VideoCapture camera;
std::ofstream ofs;
struct termios old_attr,new_attr;


/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye.xml";
String face_model_name = "face_model.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);
//CvCapture* capture;
//IplImage* img;
bool time_flag;
struct timeval stop, start;


std::vector<Rect> faces;
std::vector<Rect> eyes;
std::vector<int> labels;
struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path );
	

void sighandler(int sig){
	if(sig != SIGINT)
		return;
	while(tcsetattr(0,TCSANOW,&old_attr));	
	exit(0);
	return;
}


/*static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}*/

void Detect( Ptr<FaceRecognizer> model)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    // equalizeHist(frame_gray, frame_gray);
    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
    std::cout << faces.size()<<std::endl;
    for (int i = 0; i < faces.size() ; i++)
    {
        Mat faceROI = frame_gray(faces[i]);
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(.5, .5));
        cv::resize(faceROI, faceROI, cv::Size(100, 100));
        int predicted_label = model->predict(faceROI);
        labels.push_back(predicted_label);
    }

}

void display()
{
    int first_rad=0;
    string box_text = format("new");
    for (int i = 0; i < faces.size(); i++)
    {
        first_rad = 0;
        int pos_x = max(faces[i].x - 10, 0);
        int pos_y = max(faces[i].y - 10, 0);
        Point p1(faces[i].x, faces[i].y);
        Point p2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        rectangle(frame, p1, p2, Scalar(255, 0, 0), 3, 8, 0);
        std::cout << "eyes: " << eyes.size() << std::endl;

        for (int j = 0; j < eyes.size(); j++)
        {
            Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5, faces[i].y + eyes[j].y + eyes[j].height * 0.5);
            int radius = cvRound((eyes[j].width + eyes[i].height) * 0.25);
            std::cout << radius << std::endl;
            // if (eyes[j].y < faces[i].height / 2 && eyes[j].y > faces[i].height / 5 && radius < 15 && radius > 8)
            if (radius <15 && radius > 8)
                /*if (first_rad == 0)
                {
                    circle(frame, center, radius, Scalar(255, 0, 0), 3, 8, 0);
                    first_rad = radius;
                }
                else if( abs(radius - first_rad)<3 )
                {
                    circle(frame, center, radius, Scalar(255, 0, 0), 3, 8, 0);
                    break;
                }*/
                circle(frame, center, radius, Scalar(255, 0, 0), 3, 8, 0);

        }

        if (labels[i] > -1)
            switch (labels[i])
            {
            case 0:box_text = format("jason");
                break;
            case 1:box_text = format("kevin");
                break;
            }
        else
            box_text = format("unknown");
        
        putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0, 255, 0), 1, CV_AA);
    }

}
void detectAndDisplay(Ptr<LBPHFaceRecognizer> model)
//void detectAndDisplay (Ptr<FaceRecognizer> model)
//void detectAndDisplay(Mat &frame)
{
    //do stuff

    Mat frame_gray;
    string box_text = format("new");
    //string time_txt = format("0 ms");
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
    for (int i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
        //ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
        int pos_x = max(faces[i].x - 10, 0);
        int pos_y = max(faces[i].y - 10, 0);
        Point p1(faces[i].x, faces[i].y);
        Point p2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        rectangle(frame, p1, p2, Scalar(255, 0, 0), 1, 8, 0);
        Mat faceROI = frame_gray(faces[i]);
        if (faces.size() < 2) {
            

            //-- In each face, detect eyes
            eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(.5, .5));

            for (int j = 0; j < eyes.size(); j++)
            {
                Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5, faces[i].y + eyes[j].y + eyes[j].height * 0.5);
                int radius = cvRound((eyes[j].width + eyes[i].height) * 0.25);
                circle(frame, center, radius, Scalar(255, 0, 0), 1, 8, 0);
            }
        }
       cv::resize(faceROI, faceROI, cv::Size(100, 100));

       int predicted_label = -1;
       double confidence = 0.0;
       model->predict(faceROI, predicted_label, confidence);
       //gettimeofday(&stop, NULL);
       std::cout << "predict: " << predicted_label << " confidence: " << confidence << std::endl;
       if (predicted_label > -1 && confidence < 90)
           switch (predicted_label)
           {
           case 2:box_text = format("jason");
               break;
           case 4:box_text = format("kevin");
               break;
           default:box_text = format("unknown");
               break;
           }
       else 
           box_text = format("unknown");
       

        if (time_flag) {
            //time_txt = format(" %lu ms", ((stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec)/1000);
            box_text = box_text;
        }
        putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0, 255, 0), 1, CV_AA);

    }

    //-- Show what you got
    //imshow(window_name, frame);
    //imwrite("result.jpg", frame);
}




int main ( int argc, const char *argv[] )
{
    // set tty
    if(tcgetattr(0,&old_attr) < 0){
	perror("tcgetattr");
	return -1;
    }
    signal(SIGINT,sighandler);
    new_attr = old_attr;
    new_attr.c_lflag &= ~ICANON;
    new_attr.c_cc[VMIN] = 0;
    new_attr.c_cc[VTIME] = 0;

    if(tcsetattr(0,TCSANOW,&new_attr) < 0){
	perror("tcsetattr");
	return -1;
    }
    time_flag = false;
    
    //-- 1.train model
    /*String fn_csv = "data.txt";
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    }
    catch (const cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if (images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }
    
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::LBPHFaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();*/
    //cout << images.size();
    //cout << labels .size();
    /* */
    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    //Ptr<FaceRecognizer> model = EigenFaceRecognizer::create();
    model->read(face_model_name);
    //std::cout << 3 << std::endl;
    //-- 2. Load the cascades
    if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
    if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

    // variable to store the frame get from video stream
    pid_t pid;
    cv::Size2f frame_size;
    int ret;

    // open video stream device
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a5d5f5dacb77bbebdcbfb341e3d4355c1
    camera = cv::VideoCapture( 2 );
    camera.set(CV_CAP_PROP_FRAME_WIDTH, 160);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, 120);
    // get info of the framebuffer
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");

    // open the framebuffer device
    // http://www.cplusplus.com/reference/fstream/ofstream/ofstream/
    ofs.open("/dev/fb0");

    // check if video stream device is opened success or not
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a9d2ca36789e7fcfe7a7be3b328038585
    if( !camera.isOpened() )
     {
         std::cerr << "Could not open video device." << std::endl;
         return 1;
     }

    // set propety of the frame
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a8c6d8c2d37505b5ca61ffd4bb54e9a7c
    // https://docs.opencv.org/3.4.7/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    string time_txt = format("0 ms");
    int count = 0;
    while ( true )
    {
        gettimeofday(&start, NULL);
        count++;
        // get video frame from stream
        // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1
        // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a199844fb74226a28b3ce3a39d1ff6765
        camera >> frame;
	    char c;
	    ret = read(0,&c,1);

	    if(ret > 0 && c == 'c'){
		    c = 0;
            time_flag = true;
	    }
        // get size of the video frame
        // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a146f8e8dda07d1365a575ab83d9828d1
        //resize(frame, frame, Size(600, 300));
        /*if (!frame.empty() && count == 1)
        {
            Detect(model);
            count = 0;
            //detectAndDisplay(frame);
        }
        display();
        */
        if (time_flag == true)
        {
            detectAndDisplay(model);
            gettimeofday(&stop, NULL);
            time_txt = format(" %lu ms", ((stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec) / 1000);
            putText(frame, time_txt, Point(10, 10), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0, 255, 0), 1, CV_AA);
        }
        cv::resize(frame, frame, cv::Size(640, 480));
        frame_size = frame.size();


        // transfer color space from BGR to BGR565 (16-bit image) to fit the requirement of the LCD
        // https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
        // https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0
        cv::cvtColor(frame,frame,cv::COLOR_BGR2BGR565);

        // output the video frame to framebufer row by row
        for ( int y = 0; y < frame_size.height; y++ )
        {
            // move to the next written position of output device framebuffer by "std::ostream::seekp()"
            // http://www.cplusplus.com/reference/ostream/ostream/seekp/


            ofs.seekp(y*(fb_info.xres_virtual)*2 + (fb_info.xres_virtual-frame_size.width));
            // write to the framebuffer by "std::ostream::write()"
            // you could use "cv::Mat::ptr()" to get the pointer of the corresponding row.
            // you also need to cacluate how many bytes required to write to the buffer
            // http://www.cplusplus.com/reference/ostream/ostream/write/
            // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a13acd320291229615ef15f96ff1ff738

            ofs.write(reinterpret_cast<char*>(frame.ptr(y)),frame_size.width*2);
        }
        /*if (time_flag == true)
        {
            gettimeofday(&stop, NULL);
            time_txt = format(" %lu ms", ((stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec) / 1000);
            putText(frame, time_txt, Point(10, 10), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0, 255, 0), 1, CV_AA);
        }*/
    }

    // closing video stream
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#afb4ab689e553ba2c8f0fec41b9344ae6
    camera.release ( );
    return 0;
}

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path )
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.

    // open deive with linux system call "open( )"
    // https://man7.org/linux/man-pages/man2/open.2.html
    int fdfb = open("/dev/fb0", O_RDWR);

    // get attributes of the framebuffer device thorugh linux system call "ioctl()"
    // the command you would need is "FBIOGET_VSCREENINFO"
    // https://man7.org/linux/man-pages/man2/ioctl.2.html
    // https://www.kernel.org/doc/Documentation/fb/api.txt
    if(ioctl(fdfb,FBIOGET_VSCREENINFO,&screen_info)){
        printf("Error reading fixed information\n");
        exit(1);
    }
    // put the required attributes in variable "fb_info" you found with "ioctl() and return it."
    fb_info.xres_virtual = screen_info.xres;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;

    return fb_info;
};
