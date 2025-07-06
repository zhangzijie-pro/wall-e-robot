#ifndef MOTOR_H
#define MOTOR_H

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/pcnt.h"
#include "driver/gpio.h"
#include "driver/ledc.h"

#define MOTORA_A GPIO_NUM_35
#define MOTORA_B GPIO_NUM_36
#define ENA_IN_A GPIO_NUM_37
#define ENA_IN_B GPIO_NUM_38
#define MOTORB_A GPIO_NUM_39
#define MOTORB_B GPIO_NUM_40
#define ENB_IN_A GPIO_NUM_41
#define ENB_IN_B GPIO_NUM_42

// 默认参数
#define PPR 1040              // 每转脉冲数
#define WHEEL_DIAMETER 0.1    // 轮子直径（米）
#define SAMPLE_RATE_MS 10     // 采样间隔（毫秒）
#define PWM_FREQ_HZ 1000      // PWM频率（Hz）
#define PWM_RESOLUTION LEDC_TIMER_8_BIT // PWM分辨率

class PIDController {
private:
    float kp;             // 比例增益
    float ki;             // 积分增益
    float kd;             // 微分增益
    float integral;       // 积分项
    float last_error;     // 上次误差
    float integral_limit; // 积分饱和限制
    float output_min;     // 输出下限
    float output_max;     // 输出上限

public:
    PIDController(float kp, float ki, float kd, float integral_limit, float output_min, float output_max);
    float Compute(float setpoint, float measured_value, float dt);
    void Set_Tunings(float kp, float ki, float kd);
    void Reset();
};

class SingleMotor {
private:
    gpio_num_t enc_a_pin;     // 编码器A相引脚
    gpio_num_t enc_b_pin;     // 编码器B相引脚
    gpio_num_t pwm_in1_pin;   // PWM IN1引脚
    gpio_num_t pwm_in2_pin;   // PWM IN2引脚
    pcnt_unit_t pcnt_unit;    // PCNT单元
    ledc_channel_t pwm_channel1; // PWM通道1
    ledc_channel_t pwm_channel2; // PWM通道2
    uint8_t current_speed;    // 当前速度（0-255）
    float target_rpm;     // 目标速度（0-255）
    PIDController* pid;       // PID控制器
    bool is_forward;          // 当前方向（true为正转，false为反转）

    void GPIO_Init(void);
    void PCNT_Init(void);
    void PWM_Init(void);

public:
    SingleMotor(gpio_num_t enc_a, gpio_num_t enc_b, gpio_num_t in1, gpio_num_t in2, 
                pcnt_unit_t unit, ledc_channel_t ch1, ledc_channel_t ch2);
    ~SingleMotor();
    int16_t Get_Encoder_Count(void);
    float Get_RPM(int16_t count, int16_t prev_count, float dt);
    void Calculate_Metrics(int16_t count, int16_t prev_count, float dt, const char* motor_name);
    void Set_Speed(uint8_t speed);
    void Forward();
    void Backward();
    void Stop();
    void Update_PID(int16_t count, int16_t prev_count, float dt);
};

class MotorController {
private:
    SingleMotor* motor_a;     // 电机A
    SingleMotor* motor_b;     // 电机B
    float global_speed;     // 全局速度

public:
    MotorController();
    ~MotorController();
    void Set_Speed(uint8_t speed);
    void Set_PID_Tunings(float kp, float ki, float kd);
    void Forward();
    void Backward();
    void Turn_Left();
    void Turn_Right();
    void Stop();
    void Debug_Metrics();
    void Update_PID();
};

#endif // MOTOR_H