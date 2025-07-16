#include "ros_node/ros_node.hpp"
#include <string.h>
#include <unistd.h>

void ROSNode::init() {
    allocator = rcl_get_default_allocator();
    rclc_support_init(&support, 0, NULL, &allocator);
    rclc_node_init_default(&node, "esp32_node", "", &support);

    // 发布器
    rclc_publisher_init_default(&audio_pub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, UInt8MultiArray),
        "raw_audio");

    // 订阅器
    rclc_subscription_init_default(&motion_sub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
        "motion_cmd");

    rclc_subscription_init_default(&servo_sub, &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
        "servo_cmd");

    // 执行器
    rclc_executor_init(&executor, &support.context, 2, &allocator);

    static geometry_msgs__msg__Twist motion_msg;
    static std_msgs__msg__Float32MultiArray servo_msg;

    rclc_executor_add_subscription(&executor, &motion_sub, &motion_msg, &motion_callback, ON_NEW_DATA);
    rclc_executor_add_subscription(&executor, &servo_sub, &servo_msg, &servo_callback, ON_NEW_DATA);

    // 初始化音频消息
    audio_msg.data.capacity = 2048;
    audio_msg.data.size = 0;
    audio_msg.data.data = (uint8_t *)malloc(2048);
}

void ROSNode::spin_once() {
    rclc_executor_spin_some(&executor, RCL_MS_TO_NS(20));
}

void ROSNode::publish_audio(const uint8_t* data, size_t size) {
    if (size > audio_msg.data.capacity) size = audio_msg.data.capacity;
    memcpy(audio_msg.data.data, data, size);
    audio_msg.data.size = size;
    rcl_publish(&audio_pub, &audio_msg, NULL);
}

// ---------- 回调函数 ----------
void ROSNode::motion_callback(const void *msgin) {
    auto msg = (const geometry_msgs__msg__Twist *)msgin;
    float vx = msg->linear.x;
    float vy = msg->linear.y;
    float w  = msg->angular.z;
    // 控制底盘：使用 vx, vy, w
}

void ROSNode::servo_callback(const void *msgin) {
    auto msg = (const std_msgs__msg__Float32MultiArray *)msgin;
    // 控制舵机数组：msg->data.data[i]，共 msg->data.size 个值
}
