#ifndef AUDIO_LIB_H
#define AUDIO_LIB_H

#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include <esp_err.h>

//INMP引脚，根据自己连线修改
#define INMP_SD     GPIO_NUM_14
#define INMP_SCK    GPIO_NUM_13
#define INMP_WS     GPIO_NUM_12
 
//MAX98357A引脚，根据自己连线修改
#define MAX_DIN     GPIO_NUM_7
#define MAX_BCLK    GPIO_NUM_15
#define MAX_LRC     GPIO_NUM_16

#define SAMPLE_RATE 16000
#define BUF_SIZE    1024 * 4 // 示例缓冲区大小（可调整）

class AudioSystem {
public:
    AudioSystem();
    ~AudioSystem();

    void begin();
    void start();
    void stop();
    uint8_t read_buf[BUF_SIZE];
    uint8_t write_buf[BUF_SIZE];
    
private:
    static void i2s_read_task(void *param);
    static void i2s_write_task(void *param);

    void i2s_rx_init();
    void i2s_tx_init();

    i2s_chan_handle_t rx_handle;
    i2s_chan_handle_t tx_handle;


    size_t read_bytes;
    size_t write_bytes;
};

#endif // AUDIO_LIB_H