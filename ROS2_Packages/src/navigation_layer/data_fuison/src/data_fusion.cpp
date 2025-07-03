#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"


class DataConversionNode : public rclcpp::Node{
public: 
    DataConversionNode() : Node("data_conversion_node") {
        scan_sub = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 
            rclcpp::QoS(rclcpp::KeepLast(10)),
            std::bind(&DataConversionNode::scan_callback, this, std::placeholders::_1)
        );
        
    }
private:
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        //  turn laserScan data to point cloud data
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub;

}