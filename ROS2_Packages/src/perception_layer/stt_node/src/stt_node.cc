#include "rclcpp/rclcpp.hpp"
#include "vosk_speech_recognizer.h"
#include "config_loader.h"
#include <filesystem>
#include "std_msgs/msg/int16_multi_array.hpp"

class STTNode : public rclcpp::Node
{
public:
    STTNode() : Node("stt_node")
    {
        YAML::Node config = shared_utils::ConfigLoader::load();
        
        VoskSpeechRecognizer::Config vosk_config;
        vosk_config.model_path = get_package_path("stt_node") + "/" + config["model_path"].as<std::string>();
        vosk_config.sample_rate = config["sample_rate"].as<int>();
        vosk_config.block_size = config["block_size"].as<int>();
        vosk_config.queue_size = config["queue_size"].as<int>();
        vosk_config.log_file = get_package_path("stt_node") + "/" + config["log_file"].as<std::string>();

        RCLCPP_INFO(this->get_logger(), "Model path: %s", vosk_config.model_path.c_str());
        publisher_.create_publisher<std_msgs::msgs::String>("input_text",10);

        recognizer_ = std::make_unique<VoskSpeechRecognizer>(vosk_config);
        recognizer_->set_result_callback([this](const std::string &text) {
            RCLCPP_INFO(this->get_logger(), "Recognized: %s", text.c_str());
        });
        recognizer_->start();
        RCLCPP_INFO(this->get_logger(), "Recognizer started.");

        subscription_ = this->create_subscription<std_msgs::msg::Int16MultiArray>(
            "raw_audio", 10,
            [this](const std_msgs::msg::Int16MultiArray::SharedPtr msg) {
                RCLCPP_DEBUG(this->get_logger(), "Received audio block of size: %zu", msg->data.size());
                recognizer_->push_audio(msg->data);
            });

        recognizer_.set_result_callback([this](const std::string &text) {
            RCLCPP_INFO(this->get_logger(), "Recognized: %s", text.c_str());
            std_msgs::msg::String result_msg;
            result_msg.data = text;
            publisher_->publish(result_msg);
        });
    }
    ~STTNode()
    {
        RCLCPP_INFO(this->get_logger(), "Stopping recognizer.");
        recognizer_->stop();
    }

private:
    std::string get_package_path(const std::string &package_name) {
        return ament_index_cpp::get_package_share_directory(package_name);
    }

    std::unique_ptr<VoskSpeechRecognizer> recognizer_;
    rclcpp::Subscription<std_msgs::msg::Int16MultiArray>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<STTNode>());
    rclcpp::shutdown();
    return 0;
}