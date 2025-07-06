#include "pca9685.hpp"


PCA9685::PCA9685(RobotJointState body) : body(body) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,                 // 主机模式
        .sda_io_num = PCA9685_SDA,               // SDA 引脚
        .scl_io_num = PCA9685_SCL,               // SCL 引脚
        .sda_pullup_en = GPIO_PULLUP_ENABLE,     // 启用上拉
        .scl_pullup_en = GPIO_PULLUP_ENABLE,     // 启用上拉
        .master = {.clk_speed=I2C_MASTER_FREQ_HZ},  // I2C 时钟
    };
    i2c_param_config(PCA_PORT_NUM, &conf);      // 配置 I2C 参数
    i2c_driver_install(PCA_PORT_NUM, conf.mode, 0, 0, 0); // 安装 I2C 驱动

    for (uint8_t i = 0; i < TOTAL_JOINT_NUM; i++) {
        pid_controllers[i] = new PIDController(1.0, 0.1, 0.01, 5000.0, 0.0, 180.0); // 默认PID参数
    }
    init(); // 初始化 PCA9685
}

PCA9685::~PCA9685() {
    for (uint8_t i = 0; i < TOTAL_JOINT_NUM; i++) {
        delete pid_controllers[i];
    }
    i2c_driver_delete(PCA_PORT_NUM);
}

void PCA9685::init(){
    uint8_t data[2] = {0x00, 0x10}; // 设置 MODE1 寄存器，进入休眠
    i2c_master_write_custom(0x00, data, 2);
    data[0] = 0xFE; // 设置预分频寄存器 PRE_SCALE
    data[1] = 121;  // 50Hz: round(25MHz / (4096 * 50)) - 1
    i2c_master_write_custom(0xFE, data, 2);
    data[0] = 0x00; // 设置 MODE1 寄存器，唤醒并开启自动递增
    data[1] = 0x20;
    i2c_master_write_custom(0x00, data, 2);
}

// 设置 PWM 输出，on/off 为计数器值
void PCA9685::set_pwm(uint8_t channel, uint16_t on, uint16_t off) {
    uint8_t data[4];
    data[0] = on & 0xFF;          // ON 低字节
    data[1] = (on >> 8) & 0xFF;   // ON 高字节
    data[2] = off & 0xFF;         // OFF 低字节
    data[3] = (off >> 8) & 0xFF;  // OFF 高字节
    i2c_master_write_custom(0x06 + 4 * channel, data, 4); // 写入对应通道的寄存器
}

// 设置指定通道舵机角度（0-180度）
void PCA9685::setServoAngle(uint8_t channel, float angle) {
    if (angle < 0) angle = 0;
    if (angle > 180) angle = 180;
    uint16_t pulse = 500 + (angle / 180.0) * (2500 - 500); // 500-2500us 脉宽
    uint16_t off = (pulse * 4096) / (1000000 / SERVO_FREQ); // 转换为 PWM 计数值
    set_pwm(channel, 0, off);
}

// 批量设置手臂舵机角度，angles 格式为 [id, angle, id, angle, ...]
void PCA9685::setArmsAngles(float* angles) {
    uint8_t length = body.length() * 2;
    for (uint8_t i = 0; i < length; i += 2) {
        uint8_t target_id = static_cast<uint8_t>(angles[i]);
        float target_angle = angles[i + 1];
        Joint* joint = body.findJointById(target_id);
        if (joint != nullptr) {
            joint->expect_angle = target_angle;
            setServoAngle(target_id, target_angle);
            pid_controllers[target_id]->Reset();
        }
    }
}

void PCA9685::resetArms() {
    for (Joint* joint : body.getAllJoints()) {
        if (joint->id < 16) {
            joint->expect_angle = joint->reset_angle;
            setServoAngle(joint->id, joint->reset_angle);
            pid_controllers[joint->id]->Reset();
        }
    }
}

void PCA9685::Set_PID_Tunings(float kp, float ki, float kd) {
    for (uint8_t i = 0; i < TOTAL_JOINT_NUM; i++) {
        pid_controllers[i]->Set_Tunings(kp, ki, kd);
    }
}

void PCA9685::Update_PID() {
    float dt = SAMPLE_RATE_MS / 1000.0;
    for (Joint* joint : body.getAllJoints()) {
        if (joint->id < 16) {
            float actual_angle = joint->current_angle; // 假设由外部传感器更新
            float target_angle = joint->expect_angle;
            float adjusted_angle = pid_controllers[joint->id]->Compute(target_angle, actual_angle, dt);
            setServoAngle(joint->id, adjusted_angle);
            ESP_LOGI(PCA_TAG, "Joint %d: Target Angle: %.2f, Actual Angle: %.2f, Adjusted Angle: %.2f",
                     joint->id, target_angle, actual_angle, adjusted_angle);
        }
    }
}

// I2C 写操作，向指定寄存器写入数据
void PCA9685::i2c_master_write_custom(uint8_t reg, uint8_t* data, size_t len) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (PCA9685_ADDR_W << 1) | I2C_MASTER_WRITE, true); // 发送设备地址（写）
    i2c_master_write_byte(cmd, reg, true); // 发送寄存器地址
    i2c_master_write(cmd, data, len, true); // 写入数据
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(PCA_PORT_NUM, cmd, 1000 / portTICK_PERIOD_MS);
    i2c_cmd_link_delete(cmd);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C write failed: %s", esp_err_to_name(ret));
    }
}

// I2C 读操作，从指定寄存器读取一个字节
uint8_t PCA9685::i2c_master_read_custom(uint8_t reg) {
    uint8_t data;
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (PCA9685_ADDR_W << 1) | I2C_MASTER_WRITE, true); // 发送设备地址（写）
    i2c_master_write_byte(cmd, reg, true); // 发送寄存器地址
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (PCA9685_ADDR_R << 1) | I2C_MASTER_READ, true); // 发送设备地址（读）
    i2c_master_read_byte(cmd, &data, I2C_MASTER_NACK); // 读取数据
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(PCA_PORT_NUM, cmd, 1000 / portTICK_PERIOD_MS);
    i2c_cmd_link_delete(cmd);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C read failed: %s", esp_err_to_name(ret));
    }
    return data;
}

