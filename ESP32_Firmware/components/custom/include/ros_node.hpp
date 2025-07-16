#pragma once

#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

#include <std_msgs/msg/u_int8_multi_array.h>
#include <std_msgs/msg/float32_multi_array.h>
#include <geometry_msgs/msg/twist.h>

class ROSNode {
public:
    void init();
    void spin_once();
    void publish_audio(const uint8_t* data, size_t size);

private:
    rcl_node_t node;
    rclc_support_t support;
    rcl_allocator_t allocator;
    rclc_executor_t executor;

    rcl_publisher_t audio_pub;
    std_msgs__msg__UInt8MultiArray audio_msg;

    rcl_subscription_t motion_sub;
    rcl_subscription_t servo_sub;

    static void motion_callback(const void *msgin);
    static void servo_callback(const void *msgin);
};
