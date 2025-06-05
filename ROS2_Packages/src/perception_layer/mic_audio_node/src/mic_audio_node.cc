#include <rclcpp/rclcpp.hpp>
#include <libserialport.h>
#include "config_loader.h"
#include "std_msgs/msg/int16_multi_array.hpp"
#include <vector>
#include <ament_index_cpp/get_package_share_directory.hpp>

class MicAudioNode : public rclcpp::Node
{
public:
    MicAudioNode(): Node("mic_audio_node"){
        auto config = shared_utils::ConfigLoader::load();
        
        std::string port_name;
        auto port_name_it = config.find("port_name");
        if (port_name_it != config.end()) {
            port_name = port_name_it->second;
        } else {
            throw std::runtime_error("Missing 'port_name' in config file");
        }
        
        int baud_rate;
        auto baud_rate_it = config.find("baud_rate");
        if (baud_rate_it != config.end()) {
            try {
                baud_rate = std::stoi(baud_rate_it->second);
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid 'baud_rate' format: " + baud_rate_it->second);
            }
        } else {
            throw std::runtime_error("Missing 'baud_rate' in config file");
        }
        
        int block_size;
        auto block_size_it = config.find("block_size");
        if (block_size_it != config.end()) {
            try {
                block_size = std::stoi(block_size_it->second);
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid 'block_size' format: " + block_size_it->second);
            }
        } else {
            throw std::runtime_error("Missing 'block_size' in config file");
        }

        RCLCPP_INFO(this->get_logger(), "Port name: %s", port_name.c_str());    
        RCLCPP_INFO(this->get_logger(), "Baud rate: %d", baud_rate);
        RCLCPP_INFO(this->get_logger(), "Block size: %d", block_size);

        publisher_ = this->create_publisher<std_msgs::msg::Int16MultiArray>("raw_audio", 10);
        sp_get_port_by_name(port_name.c_str(), &port_);
        sp_open(port_, SP_MODE_READ);
        sp_set_baudrate(port_, baud_rate);
        sp_set_bits(port_, 8);
        sp_set_parity(port_, SP_PARITY_NONE);   
        sp_set_stopbits(port_, 1);

        // timer_ = this->create_wall_timer(
        //     std::chrono::milliseconds(100),
        //     std::bind(&MicAudioNode::timer_callback, this)
        // );
        block_size_ = block_size;
        RCLCPP_INFO(this->get_logger(), "MicAudioNode initialized successfully.");
    }
    ~MicAudioNode(){
        if(port_){
            RCLCPP_INFO(this->get_logger(), "Closing serial port.");
            sp_close(port_);
            sp_free_port(port_);
        }
    }

private:
    void read_serial(){
        size_t bytes_to_read = block_size_*sizeof(int16_t);
        std::vector<int16_t> buffer(block_size_);
        int bytes_read = sp_nonblocking_read(port_, buffer.data(), bytes_to_read);
        if (bytes_read==int(bytes_to_read)){
            std_msgs::msg::Int16MultiArray msg;
            msg.data.resize(block_size_);
            std::memcpy(msg.data.data(), buffer.data(), bytes_to_read);
            publisher_->publish(msg);
        } else {
            RCLCPP_WARN(this->get_logger(), "Failed to read from serial port.");
        }
    }

    std::string get_package_path(const std::string &package_name)
    {
        std::string path = ament_index_cpp::get_package_share_directory(package_name);
        return path;
    }

    rclcpp::Publisher<std_msgs::msg::Int16MultiArray>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    struct sp_port *port_=nullptr;
    size_t block_size_=1600;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MicAudioNode>();
    RCLCPP_INFO(node->get_logger(),"starting in mic_audio_node");


    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}