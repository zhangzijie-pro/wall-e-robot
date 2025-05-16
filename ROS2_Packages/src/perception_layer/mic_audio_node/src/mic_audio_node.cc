#include "vosk_speech_recognizer.h"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("mic_audio_node");

    RCLCPP_INFO(node->get_logger(),"this is a cpp node");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}