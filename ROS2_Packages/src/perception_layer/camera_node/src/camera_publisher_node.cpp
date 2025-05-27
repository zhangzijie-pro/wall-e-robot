#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>

class CameraPublisherNode: public rclcpp::Node
{
public:
    CameraPublisherNode()
    :Node("camera_publisher_node")
    {
        image_pub_ = image_transport::create_publisher(this, "image_raw");

        std::string pipeline = 
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=(string)BGRx ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink";

        cap_.open(pipeline, cv::CAP_GSTREAMER);

        if (!cap_.isOpened()){
            RCLCPP_ERROR(this->get_logger(), "Failed to open CSI camera using GSTREAMER pipeline");
            rclcpp::shutdown();
            return;
        }

        timer_ = this -> create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&CameraPublisherNode::timer_callback, this)
        );
    }

private:
    void time_callback(){
        cv::Mat frame;
        if (!cap_.read(frame)) {
            RCLCPP_WARN(this->get_logger(), "Failed to read frame");
            return;
        }

        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        msg->header.stamp = this->now();
        image_pub_.publish(msg);
    }

    image_transport::Publisher image_pub_;
    cv::VideoCapture cap_;
    rclcpp::TimerBase::SharedPtr timer_;
}

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraPublisherNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}