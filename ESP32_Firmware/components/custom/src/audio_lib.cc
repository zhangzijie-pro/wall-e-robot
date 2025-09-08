#include "audio_lib.hpp"


AudioSystem::AudioSystem()
    : rx_handle(nullptr), tx_handle(nullptr) {}

AudioSystem::~AudioSystem() {
    // 停止并释放资源
    stop();
}

void AudioSystem::begin() {
    i2s_rx_init();
    i2s_tx_init();
}

void AudioSystem::i2s_rx_init() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_frame_num = 1023;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = INMP_SCK,
            .ws = INMP_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = INMP_SD,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

void AudioSystem::i2s_tx_init() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_MASTER);
    chan_cfg.dma_frame_num = 1023;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &tx_handle, NULL));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = MAX_BCLK,
            .ws = MAX_LRC, 
            .dout = MAX_DIN,
            .din = I2S_GPIO_UNUSED,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(tx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(tx_handle));
}
void AudioSystem::start() {
    xTaskCreate(i2s_read_task, "i2s_read_task", 4096 * 2, this, tskIDLE_PRIORITY, NULL);
    xTaskCreate(i2s_write_task, "i2s_write_task", 4096 * 2, this, tskIDLE_PRIORITY, NULL);
}

void AudioSystem::stop() {
    if (rx_handle) {
        i2s_channel_disable(rx_handle);
        i2s_del_channel(rx_handle);
        rx_handle = nullptr;
    }
    if (tx_handle) {
        i2s_channel_disable(tx_handle);
        i2s_del_channel(tx_handle);
        tx_handle = nullptr;
    }
}

void AudioSystem::i2s_read_task(void *param) {
    AudioSystem *audio = static_cast<AudioSystem*>(param);
    while (1) {
        esp_err_t ret = i2s_channel_read(audio->rx_handle, audio->read_buf, BUF_SIZE, &audio->read_bytes, pdMS_TO_TICKS(1000));
        if (ret == ESP_OK) {
            // 可选：添加数据处理逻辑
        }
    }
    vTaskDelete(NULL);
}

void AudioSystem::i2s_write_task(void *param) {
    AudioSystem *audio = static_cast<AudioSystem*>(param);
    while (1) {
        if (audio->read_bytes > 0) {
            i2s_channel_write(audio->tx_handle, audio->read_buf, BUF_SIZE, &audio->read_bytes, pdMS_TO_TICKS(1000));
        }
    }
    vTaskDelete(NULL);
}
