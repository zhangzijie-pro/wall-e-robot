#ifndef PCA9685_H
#define PCA9685_H

#include "driver/i2c.h"
#include "esp_log.h"
#include "joint.hpp"
#include "motor.hpp"

#define PCA9685_SDA GPIO_NUM_17      // I2C SDA 引脚
#define PCA9685_SCL GPIO_NUM_18      // I2C SCL 引脚

#define PCA_PORT_NUM I2C_NUM_0       // I2C 端口号
#define I2C_MASTER_FREQ_HZ 100000    // I2C 主机频率 100kHz
#define SERVO_FREQ         48        // 舵机 PWM 频率 50Hz


#define PCA9685_ADDR_W 0x40          // PCA9685 写地址
#define PCA9685_ADDR_R 0x41          // PCA9685 读地址
static const char *PCA_TAG = "pca9685";

class PCA9685 {
private:
    RobotJointState body

    void i2c_master_write_custom(uint8_t reg, uint8_t* data, size_t len);
    uint8_t i2c_master_read_custom(uint8_t reg);
    PIDController* pid_controllers[TOTAL_JOINT_NUM];

public:
    PCA9685(RobotJointState body);
    ~PCA9685() ;
    void init();
    void set_pwm(uint8_t channel, uint16_t on, uint16_t off)
    void setServoAngle(uint8_t channel, float angle);
    void setArmsAngles(float* angles);
    void resetArms();

    // PID 
    void Set_PID_Tunings(float kp, float ki, float kd);
    void Update_PID();
};

#endif