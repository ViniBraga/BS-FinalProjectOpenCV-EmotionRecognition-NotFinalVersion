#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

bool readImages(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels);
std::string decodeEmotion(int prediction);

int main() {

    cv::CascadeClassifier haarCascade;
    cv::Mat frame;

    std::string haarXml = "lbpcascade_frontalface.xml";
    std::string imagesTxt = "images.txt";

    std::vector<cv::Mat> images;
    std::vector<int> labels;

    std::cout << "Reading images..." << std::endl;
    if(!readImages(imagesTxt, images, labels)){
      std::cout << "Error opening file!" << std::endl;
      exit(1);
    }

    int im_width = images[1].cols;
    int im_height = images[1].rows;

    std::cout << "Trainning machine..." << std::endl;
    cv::Ptr<cv::FaceRecognizer> model = cv::createFisherFaceRecognizer();
    model->train(images, labels);

    std::cout << "Loading Haar Cascade..." << std::endl;
    haarCascade.load(haarXml);

    std::cout << "Creating socket connection..." << std::endl;
    //create socket connection

    std::cout << "Starting camera..." << std::endl;
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
        std::cout << "Capture device not found!" << std::endl;
        return -1;
    }

    while(1) {
        cap >> frame;
        // Clone the current frame:
        cv::Mat original = frame.clone();
        // Convert the current frame to grayscale:
        cv::Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        std::vector< cv::Rect_<int> > faces;
        haarCascade.detectMultiScale(gray, faces);
        // At this point you have the position of the faces in
        // faces. Now we'll get the faces, make a prediction and
        // annotate it in the video. Cool or what?
        for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            cv::Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            cv::Mat face = gray(face_i);
            // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
            // verify this, by reading through the face recognition tutorial coming with OpenCV.
            // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
            // input data really depends on the algorithm used.
            //
            // I strongly encourage you to play around with the algorithms. See which work best
            // in your scenario, LBPH should always be a contender for robust face recognition.
            //
            // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
            // face you have just found:
            cv::Mat face_resized;
            cv::resize(face, face_resized, cv::Size(im_width, im_height), 1.0, 1.0, cv::INTER_CUBIC);
            // Now perform the prediction, see how easy that is:
            int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            // Create the text we will annotate the box with:
            std::string mystring = decodeEmotion(prediction);
            char emotion[1024];
            strcpy(emotion, mystring.c_str());
            std::string box_text = cv::format(emotion);

            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            putText(original, box_text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
        // Show the result:
        imshow("face_recognizer", original);
        // And display it:
        char key = (char) cv::waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
    return 0;
}

bool readImages(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels) {
    const char separator = ';';
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file) {
        return false;
    }
    std::string line, path, classlabel;
    while (getline(file, line)) {
        std::stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(cv::imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
    return true;
}

std::string decodeEmotion(int prediction){
  switch (prediction) {
    case 0: return "Anger";
    break;
    case 1: return "Happy";
    break;
    default: return "Neutral";
    break;
  }
}
