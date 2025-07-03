#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/header.hpp>
#include <format.h>

class CameraPublisherNode : public rclcpp::Node
{
public:
    CameraPublisherNode()
    : Node("camera_publisher_node")
    {
        this->declare_parameter<std::string>("camera_type", "CSI");
        this->declare_parameter<int>("fps", 30);
        this->declare_parameter<int>("resolution_H", 1080);
        this->declare_parameter<int>("resolution_W", 1920);

        std::string camera_type;
        int fps, resolution_H, resolution_W;
        this->get_parameter("camera_type", camera_type);
        this->get_parameter("fps", fps);
        this->get_parameter("resolution_H", resolution_H);
        this->get_parameter("resolution_W", resolution_W);

        image_pub_ = image_transport::create_publisher(this, "image_raw");

        if (camera_type == "USB") {
            cap_.open(0, cv::CAP_V4L2);
        } else if (camera_type == "CSI") {
            std::string pipeline = std::format(
                "nvarguscamerasrc ! video/x-raw(memory:memory:NVMM), width={}, height={}, framerate={}/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink",
            resolution_W, resolution_H, fps);

            cap_.open(pipeline, cv::CAP_G);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unsupported camera type: %s", camera_type.c_str());
            rclcpp::shutdown();
            return;
        }

        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open %s camera", camera_type.c_str());
            rclcpp::shutdown();
            return;
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / fps),
            std::bind(&CameraPublisherNode::timer_callback, this));
    }

private:
    void timer_callback()
    {
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
};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraPublisherNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}