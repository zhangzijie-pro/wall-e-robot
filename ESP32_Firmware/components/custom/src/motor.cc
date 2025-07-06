#include "motor.hpp"

static const char *TAG = "MotorController";

PIDController::PIDController(float kp, float ki, float kd, float integral_limit, float output_min, float output_max) {
    this->kp = kp;
    this->ki = ki;
    this->kd = kd;
    this->integral_limit = integral_limit;
    this->output_min = output_min;
    this->output_max = output_max;
    integral = 0.0;
    last_error = 0.0;
}

float PIDController::Compute(float setpoint, float measured_value, float dt) {
    float error = setpoint - measured_value;
    integral += error * dt;
    if (integral > integral_limit) integral = integral_limit;
    if (integral < -integral_limit) integral = -integral_limit;
    float derivative = (error - last_error) / dt;
    float output = kp * error + ki * integral + kd * derivative;
    if (output > output_max) output = output_max;
    if (output < output_min) output = output_min;
    last_error = error;
    return output;
}

void PIDController::Set_Tunings(float kp, float ki, float kd) {
    this->kp = kp;
    this->ki = ki;
    this->kd = kd;
}

void PIDController::Reset() {
    integral = 0.0;
    last_error = 0.0;
}

SingleMotor::SingleMotor(
    gpio_num_t enc_a, gpio_num_t enc_b, 
    gpio_num_t in1, gpio_num_t in2, 
    pcnt_unit_t unit, 
    ledc_channel_t ch1, ledc_channel_t ch2)
{
    enc_a_pin = enc_a;
    enc_b_pin = enc_b;

    pwm_in1_pin = in1;
    pwm_in2_pin = in2;

    pcnt_unit = unit;

    pwm_channel1 = ch1;
    pwm_channel2 = ch2;

    current_speed = 0; target_rpm = 0.0;
    is_forward = true;
    pid = new PIDController(1.0, 0.1, 0.01, 5000.0, 0.0, 255.0); 

    GPIO_Init();
    PCNT_Init();
    PWM_Init();
    ESP_LOGI(TAG, "SingleMotor initialized for PCNT unit %d", unit);
}

SingleMotor::~SingleMotor() {
    delete pid;
    ledc_stop(LEDC_HIGH_SPEED_MODE, pwm_channel1, 0);
    ledc_stop(LEDC_HIGH_SPEED_MODE, pwm_channel2, 0);
    pcnt_counter_pause(pcnt_unit);
}

void SingleMotor::GPIO_Init(void) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << enc_a_pin) | (1ULL << enc_b_pin),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf);

    io_conf.pin_bit_mask = (1ULL << pwm_in1_pin) | (1ULL << pwm_in2_pin);
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    gpio_config(&io_conf);
}

void SingleMotor::PCNT_Init(void) {
    pcnt_config_t pcnt_config = {
        .pulse_gpio_num = enc_a_pin,
        .ctrl_gpio_num = enc_b_pin,
        .unit = pcnt_unit,
        .channel = PCNT_CHANNEL_0,
        .pos_mode = PCNT_COUNT_INC,
        .neg_mode = PCNT_COUNT_DEC,
        .lctrl_mode = PCNT_MODE_REVERSE,
        .hctrl_mode = PCNT_MODE_KEEP,
        .counter_h_lim = 32767,
        .counter_l_lim = -32767,
    };
    pcnt_unit_config(&pcnt_config);
    pcnt_set_filter_value(pcnt_unit, 100);
    pcnt_filter_enable(pcnt_unit);
    pcnt_counter_clear(pcnt_unit);
    pcnt_counter_pause(pcnt_unit);
    pcnt_counter_resume(pcnt_unit);
}

void SingleMotor::PWM_Init(void) {
    ledc_timer_config_t ledc_timer = {
        .speed_mode = LEDC_HIGH_SPEED_MODE,
        .duty_resolution = PWM_RESOLUTION,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = PWM_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK
    };
    ledc_timer_config(&ledc_timer);

    ledc_channel_config_t ledc_channel[2] = {
        {
            .gpio_num = pwm_in1_pin,
            .speed_mode = LEDC_HIGH_SPEED_MODE,
            .channel = pwm_channel1,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER_0,
            .duty = 0,
            .hpoint = 0
        },
        {
            .gpio_num = pwm_in2_pin,
            .speed_mode = LEDC_HIGH_SPEED_MODE,
            .channel = pwm_channel2,
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER_0,
            .duty = 0,
            .hpoint = 0
        }
    };
    for (int i = 0; i < 2; i++) {
        ledc_channel_config(&ledc_channel[i]);
    }
}

int16_t SingleMotor::Get_Encoder_Count(void) {
    int16_t count;
    pcnt_get_counter_value(pcnt_unit, &count);
    return count;
}

float SingleMotor::Get_RPM(int16_t count, int16_t prev_count, float dt) {
    return ((float)(count - prev_count) * 60.0) / (PPR * dt);
}

void SingleMotor::Calculate_Metrics(int16_t count, int16_t prev_count, float dt, const char* motor_name) {
    float rpm = ((float)(count - prev_count) * 60.0) / (PPR * dt);
    float angle = ((float)count / PPR) * 360.0;
    float circumference = 3.14159 * WHEEL_DIAMETER;
    float distance = ((float)count / PPR) * circumference;
    ESP_LOGI(TAG, "%s: Count: %d, RPM: %.2f, Angle: %.2f deg, Distance: %.2f m",
             motor_name, count, rpm, angle, distance);
}


void SingleMotor::Set_Speed(uint8_t speed) {
    target_rpm = rpm;
    pid->Reset();
}

void SingleMotor::Forward() {
    is_forward = true;
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, pwm_channel1, target_speed);
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, pwm_channel2, 0);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, pwm_channel1);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, pwm_channel2);
}

void SingleMotor::Backward() {
    is_forward = false;
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, pwm_channel1, 0);
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, pwm_channel2, target_speed);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, pwm_channel1);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, pwm_channel2);
}

void SingleMotor::Stop() {
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, pwm_channel1, 0);
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, pwm_channel2, 0);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, pwm_channel1);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, pwm_channel2);
}

void SingleMotor::Update_PID(int16_t count, int16_t prev_count, float dt) {
    float actual_rpm = Get_RPM(count, prev_count, dt);
    float signed_target_rpm = is_forward ? target_rpm : -target_rpm;
    float output = pid->Compute(signed_target_rpm, actual_rpm, dt);
    current_speed = (uint8_t)(output < 0 ? 0 : (output > 255 ? 255 : output));
    if (is_forward) {
        ledc_set_duty(LEDC_HIGH_SPEED_MODE, pwm_channel1, current_speed);
        ledc_set_duty(LEDC_HIGH_SPEED_MODE, pwm_channel2, 0);
    } else {
        ledc_set_duty(LEDC_HIGH_SPEED_MODE, pwm_channel1, 0);
        ledc_set_duty(LEDC_HIGH_SPEED_MODE, pwm_channel2, current_speed);
    }
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, pwm_channel1);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, pwm_channel2);
}

MotorController::MotorController() {
    motor_a = new SingleMotor(MOTORA_A, MOTORA_B, ENA_IN_A, ENA_IN_B, PCNT_UNIT_0, LEDC_CHANNEL_0, LEDC_CHANNEL_1);
    motor_b = new SingleMotor(MOTORB_A, MOTORB_B, ENB_IN_A, ENB_IN_B, PCNT_UNIT_1, LEDC_CHANNEL_2, LEDC_CHANNEL_3);
    global_speed = 0.0;
    ESP_LOGI(TAG, "MotorController initialized");
}

MotorController::~MotorController() {
    delete motor_a;
    delete motor_b;
}

void MotorController::Set_Speed(float speed) {
    global_speed = speed;
    motor_a->Set_Speed(speed);
    motor_b->Set_Speed(speed);
}

void MotorController::Set_PID_Tunings(float kp, float ki, float kd) {
    motor_a->pid->Set_Tunings(kp, ki, kd);
    motor_b->pid->Set_Tunings(kp, ki, kd);
}

void MotorController::Forward() {
    motor_a->Forward();
    motor_b->Forward();
}

void MotorController::Backward() {
    motor_a->Backward();
    motor_b->Backward();
}

void MotorController::Turn_Left() {
    motor_a->Backward();
    motor_b->Forward();
}

void MotorController::Turn_Right() {
    motor_a->Forward();
    motor_b->Backward();
}

void MotorController::Stop() {
    motor_a->Stop();
    motor_b->Stop();
}

void MotorController::Debug_Metrics() {
    static int16_t prev_count_a = 0, prev_count_b = 0;
    int16_t count_a = motor_a->Get_Encoder_Count();
    int16_t count_b = motor_b->Get_Encoder_Count();
    motor_a->Calculate_Metrics(count_a, prev_count_a, SAMPLE_RATE_MS / 1000.0, "Motor A");
    motor_b->Calculate_Metrics(count_b, prev_count_b, SAMPLE_RATE_MS / 1000.0, "Motor B");
    prev_count_a = count_a;
    prev_count_b = count_b;
}


void MotorController::Update_PID() {
    static int16_t prev_count_a = 0, prev_count_b = 0;
    int16_t count_a = motor_a->Get_Encoder_Count();
    int16_t count_b = motor_b->Get_Encoder_Count();
    motor_a->Update_PID(count_a, prev_count_a, SAMPLE_RATE_MS / 1000.0);
    motor_b->Update_PID(count_b, prev_count_b, SAMPLE_RATE_MS / 1000.0);
    prev_count_a = count_a;
    prev_count_b = count_b;
}