#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_http_server.h"

#include "driver/i2c.h"
#include "driver/i2s_tdm.h"
#include "driver/i2s_std.h"  // 标准I2S模式（与ES7210兼容）
#include "es7210.h"  
#include "es7210_reg.h"

// ==================== 配置参数 ====================
#define TAG "ESP32-S3-AUDIO"  // 日志标签
#define WIFI_SSID "ESP"  // WiFi名称
#define WIFI_PASS "12345678"  // WiFi密码
#define HTTP_PORT 80          // HTTP服务端口
#define AUDIO_BUFFER_SIZE 1024  // 音频缓冲区大小（16bit采样点）

/* I2C port and GPIOs */
#define EXAMPLE_I2C_NUM            (0)
#define EXAMPLE_I2C_SDA_IO         (17)
#define EXAMPLE_I2C_SCL_IO         (18)

/* I2S port and GPIOs */
#define EXAMPLE_I2S_NUM            (0)
#define EXAMPLE_I2S_MCK_IO         (16)
#define EXAMPLE_I2S_BCK_IO         (9)
#define EXAMPLE_I2S_WS_IO          (45)
#define EXAMPLE_I2S_DI_IO          (10)

/* I2S configurations */
#define EXAMPLE_I2S_TDM_FORMAT     (ES7210_I2S_FMT_I2S)
#define EXAMPLE_I2S_CHAN_NUM       (2)
#define EXAMPLE_I2S_SAMPLE_RATE    (16000)
#define EXAMPLE_I2S_MCLK_MULTIPLE  (I2S_MCLK_MULTIPLE_256)
#define EXAMPLE_I2S_SAMPLE_BITS    (I2S_DATA_BIT_WIDTH_16BIT)
#define EXAMPLE_I2S_TDM_SLOT_MASK  (I2S_TDM_SLOT0 | I2S_TDM_SLOT1)

/* ES7210 configurations */
#define EXAMPLE_ES7210_I2C_ADDR    (0x40)
#define EXAMPLE_ES7210_I2C_CLK     (400000)
#define EXAMPLE_ES7210_MIC_GAIN    (ES7210_MIC_GAIN_30DB)
#define EXAMPLE_ES7210_MIC_BIAS    (ES7210_MIC_BIAS_2V87)
#define EXAMPLE_ES7210_ADC_VOLUME  (0)

// ==================== 全局变量 ====================
// 音频缓冲区（双缓冲机制）
#define AUDIO_BUFFER_SIZE_BYTES  1024    // 必须是 128 的整数倍
#define AUDIO_BUFFER_SAMPLES     (AUDIO_BUFFER_SIZE_BYTES / sizeof(int16_t))
#define AUDIO_TASK_STACK_SIZE    4096
#define AUDIO_TASK_PRIORITY      5
i2s_chan_handle_t i2s_rx_chan = NULL;

// ==================== 全局变量 ====================
static int16_t pcm_buffer[AUDIO_BUFFER_SAMPLES] __attribute__((aligned(4)));// PCM 数据缓冲区（双缓冲）

static i2s_chan_handle_t es7210_i2s_init(void)
{
    i2s_chan_handle_t i2s_rx_chan = NULL;  // 定义接收通道句柄
    ESP_LOGI(TAG, "Create I2S receive channel");
    i2s_chan_config_t i2s_rx_conf = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER); // 配置接收通道
    ESP_ERROR_CHECK(i2s_new_channel(&i2s_rx_conf, NULL, &i2s_rx_chan)); // 创建i2s通道

    ESP_LOGI(TAG, "Configure I2S receive channel to TDM mode");
    // 定义接收通道为I2S TDM模式 并配置
    i2s_tdm_config_t i2s_tdm_rx_conf = {  
        .slot_cfg = I2S_TDM_PHILIPS_SLOT_DEFAULT_CONFIG(EXAMPLE_I2S_SAMPLE_BITS, I2S_SLOT_MODE_STEREO, EXAMPLE_I2S_TDM_SLOT_MASK),
        .clk_cfg  = {
            .clk_src = I2S_CLK_SRC_DEFAULT,
            .sample_rate_hz = EXAMPLE_I2S_SAMPLE_RATE,
            .mclk_multiple = EXAMPLE_I2S_MCLK_MULTIPLE
        },
        .gpio_cfg = {
            .mclk = EXAMPLE_I2S_MCK_IO,
            .bclk = EXAMPLE_I2S_BCK_IO,
            .ws   = EXAMPLE_I2S_WS_IO,
            .dout = -1, // ES7210 only has ADC capability
            .din  = EXAMPLE_I2S_DI_IO
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_tdm_mode(i2s_rx_chan, &i2s_tdm_rx_conf)); // 初始化I2S通道为TDM模式
    return i2s_rx_chan;
}

static void es7210_codec_init(void)
{
    // 初始化I2C接口
    ESP_LOGI(TAG, "Init I2C used to configure ES7210");
    i2c_config_t i2c_conf = {
        .sda_io_num = EXAMPLE_I2C_SDA_IO,
        .scl_io_num = EXAMPLE_I2C_SCL_IO,
        .mode = I2C_MODE_MASTER,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = EXAMPLE_ES7210_I2C_CLK,
    };
    ESP_ERROR_CHECK(i2c_param_config(EXAMPLE_I2C_NUM, &i2c_conf));
    ESP_ERROR_CHECK(i2c_driver_install(EXAMPLE_I2C_NUM, i2c_conf.mode, 0, 0, 0));

    // 创建es7210器件句柄
    es7210_dev_handle_t es7210_handle = NULL;
    es7210_i2c_config_t es7210_i2c_conf = {
        .i2c_port = EXAMPLE_I2C_NUM,
        .i2c_addr = EXAMPLE_ES7210_I2C_ADDR
    };
    ESP_ERROR_CHECK(es7210_new_codec(&es7210_i2c_conf, &es7210_handle));

    // 初始化es7210芯片
    ESP_LOGI(TAG, "Configure ES7210 codec parameters");
    es7210_codec_config_t codec_conf = {
        .i2s_format = EXAMPLE_I2S_TDM_FORMAT,
        .mclk_ratio = EXAMPLE_I2S_MCLK_MULTIPLE,
        .sample_rate_hz = EXAMPLE_I2S_SAMPLE_RATE,
        .bit_width = (es7210_i2s_bits_t)EXAMPLE_I2S_SAMPLE_BITS,
        .mic_bias = EXAMPLE_ES7210_MIC_BIAS,
        .mic_gain = EXAMPLE_ES7210_MIC_GAIN,
        .flags.tdm_enable = true
    };
    ESP_ERROR_CHECK(es7210_config_codec(es7210_handle, &codec_conf));
    ESP_ERROR_CHECK(es7210_config_volume(es7210_handle, EXAMPLE_ES7210_ADC_VOLUME));
}

void audio_capture_task(void *pvParameters)
{
    size_t bytes_read;
    while (1) {
        esp_err_t ret = i2s_channel_read(i2s_rx_chan, pcm_buffer, sizeof(pcm_buffer), &bytes_read, pdMS_TO_TICKS(100));
        if (ret == ESP_OK) {
            // 打印采集到的数据量和前几个采样点
            // ESP_LOGI( TAG,"采集到音频数据: %d 字节", bytes_read);
            // for (int i = 0; i < 10 && i < bytes_read / sizeof(int16_t); i++) {
            //     ESP_LOGI(TAG, "pcm_buffer[%d] = %d\n", i, pcm_buffer[i]);
            // }
        } else {
            ESP_LOGE(TAG, "音频采集失败，错误代码: 0x%x", ret);
        }

        // 可选：延时一段时间再继续采集
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

void i2c_scan(void)
{
    if (!i2s_rx_chan) {
        ESP_LOGE(TAG, "I2S 初始化失败");
        return;
    }
    printf("Scanning I2C bus...\n");
    for (int addr = 0; addr < 128; addr++) {
        i2c_cmd_handle_t cmd = i2c_cmd_link_create();
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
        esp_err_t ret = i2c_master_cmd_begin(I2C_NUM_0, cmd, pdMS_TO_TICKS(100));
        i2c_cmd_link_delete(cmd);
        if (ret == ESP_OK) {
            printf("Found device at 0x%02X\n", addr);
        }
    }
}
// ==================== WIFI ====================

/**
 * @brief WiFi 事件处理回调
 */
static void wifi_event_handler(void* arg, esp_event_base_t event_base, 
                             int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_STA_START:
                ESP_LOGI(TAG, "WiFi STA 模式启动");
                esp_wifi_connect();
                break;
            case WIFI_EVENT_STA_CONNECTED:
                ESP_LOGI(TAG, "连接到 AP");
                break;
            case WIFI_EVENT_STA_DISCONNECTED:
                ESP_LOGI(TAG, "断开连接，尝试重新连接...");
                esp_wifi_connect();
                break;
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "获取到 IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

/**
 * @brief 初始化 NVS（存储 WiFi 配置）
 */
void wifi_init_nvs() {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || 
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
}
/**
 * @brief PCM 流处理函数
 * 提供裸 PCM 数据流的 HTTP 端点（无 WAV 头）
 */
esp_err_t pcm_handler(httpd_req_t *req) {
    // 设置响应类型为原始 PCM 数据流
    httpd_resp_set_type(req, "audio/x-wav");

    // 增加缓存控制头，防止浏览器缓存导致问题
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    httpd_resp_set_hdr(req, "Content-Type", "audio/L16; rate=16000");
    
    size_t bytes_read;
    esp_err_t ret;
    bool is_closed = false;

    while (!is_closed) {
        ret = i2s_channel_read(i2s_rx_chan, pcm_buffer, sizeof(pcm_buffer), &bytes_read, pdMS_TO_TICKS(100));
        
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "I2S 读取失败: %s", esp_err_to_name(ret));
            break;
        }

        if (bytes_read > 0) {
            // 检查数据发送是否成功
            esp_err_t send_ret = httpd_resp_send_chunk(req, (const char*)pcm_buffer, bytes_read);
            if (send_ret != ESP_OK) {
                ESP_LOGE(TAG, "发送失败，客户端可能已断开: %s", esp_err_to_name(send_ret));
                is_closed = true;
            }
        }
    }

    // 结束分块传输
    httpd_resp_send_chunk(req, NULL, 0);
    return ESP_OK;
}

// ==================== 网页端 HTML + JavaScript 修改版 ====================
/**
 * @brief 提供波形显示页面（动态IP和实时绘图）
 */
esp_err_t index_handler(httpd_req_t *req) {
    char* html_template = 
        "<!DOCTYPE html><html><head><meta http-equiv=\"Content-Type\" content=\"text/html;charset=utf-8\"/><style>"
        "canvas {"
        "    border: 2px solid #2c3e50;"
        "    border-radius: 8px;"
        "    margin: 20px auto;"
        "    background: #ecf0f1;"
        "    box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
        "}"
        "body {"
        "    text-align: center;"
        "    background: #bdc3c7;"
        "    font-family: Arial, sans-serif;"
        "}"
        "h2 {"
        "    color: #2c3e50;"
        "    margin: 20px 0;"
        "    font-size: 24px;"
        "}"
        "</style></head><body>"
        "<h2>实时音频波形显示</h2>"
        "<button onclick=\"startVisualization()\">开始接收数据</button>"
        "<canvas id=\"waveform\" width=\"800\" height=\"300\"></canvas>"
        "<script>"
        "let isStreaming = false;"
        "let canvas, ctx;"
        "const BUFFER_SIZE = 4096;"
        "const SAMPLE_RATE = 16000;"
        "let dataBuffer = new Float32Array(0);"
        
        "function initCanvas() {"
        "    canvas = document.getElementById('waveform');"
        "    ctx = canvas.getContext('2d');"
        "    ctx.clearRect(0, 0, canvas.width, canvas.height);"
        "}"
        
        "async function startVisualization() {"
        "    if (isStreaming) return;"
        "    isStreaming = true;"
        "    initCanvas();"
        "    "
        "    try {"
        "        const response = await fetch('http://192.168.231.173:80/pcm');"
        "        const reader = response.body.getReader();"
        "        "
        "        while (isStreaming) {"
        "            const { done, value } = await reader.read();"
        "            if (done) break;"
        "            "
        "            const int16Data = new Int16Array(value.buffer);"
        "            const newData = new Float32Array(int16Data.length);"
        "            for (let i = 0; i < int16Data.length; i++) {"
        "                newData[i] = int16Data[i] / 32768.0;"
        "            }"
        "            "
        "            dataBuffer = mergeBuffers(dataBuffer, newData);"
        "            dataBuffer = dataBuffer.slice(-BUFFER_SIZE);"
        "            "
        "            requestAnimationFrame(drawWaveform);"
        "        }"
        "    } catch (err) {"
        "        console.error('数据接收错误:', err);"
        "    } finally {"
        "        isStreaming = false;"
        "    }"
        "}"
        
        "function mergeBuffers(oldBuffer, newBuffer) {"
        "    const merged = new Float32Array(oldBuffer.length + newBuffer.length);"
        "    merged.set(oldBuffer, 0);"
        "    merged.set(newBuffer, oldBuffer.length);"
        "    return merged;"
        "}"
        
        "function drawWaveform() {"
        "    ctx.fillStyle = 'white';"
        "    ctx.fillRect(0, 0, canvas.width, canvas.height);"
        "    "
        "    if (dataBuffer.length === 0) return;"
        "    "
        "    ctx.beginPath();"
        "    ctx.strokeStyle = 'blue';"
        "    ctx.lineWidth = 1;"
        "    "
        "    const samplesPerPixel = Math.ceil(dataBuffer.length / canvas.width);"
        "    const centerY = canvas.height / 2;"
        "    const amplitude = canvas.height * 0.4;"
        "    "
        "    for (let x = 0; x < canvas.width; x++) {"
        "        const start = Math.floor(x * samplesPerPixel);"
        "        const end = Math.min(start + samplesPerPixel, dataBuffer.length);"
        "        let max = 0;"
        "        "
        "        for (let i = start; i < end; i++) {"
        "            const val = Math.abs(dataBuffer[i]);"
        "            if (val > max) max = val;"
        "        }"
        "        "
        "        const y = centerY - (max * amplitude);"
        "        ctx.lineTo(x, y);"
        "    }"
        "    "
        "    ctx.stroke();"
        "}"
        "    "
        "setInterval(function() {"
        "      startVisualization();"
        "}, 500);"
        "</script></body></html>";

    // 动态获取IP地址的正确实现
    char ip_str[16];
    esp_netif_ip_info_t ip_info;

    // 获取STA接口句柄
    esp_netif_t *sta_netif = esp_netif_get_handle_from_ifkey("WIFI_STA");
    if (sta_netif == NULL) {
        ESP_LOGE(TAG, "Failed to get STA interface");
    } else {
        esp_netif_get_ip_info(sta_netif, &ip_info);
        snprintf(ip_str, sizeof(ip_str), IPSTR, IP2STR(&ip_info.ip));
    }
    // // 动态获取IP地址（同之前代码）
    // char ip_str[16];
    // esp_netif_ip_info_t ip_info;
    // esp_netif_get_ip_info(esp_netif_get_handle_from_key("WIFI_STA_DEF"), &ip_info);
    // snprintf(ip_str, sizeof(ip_str), IPSTR, IP2STR(&ip_info.ip));
    
    char* html = (char*)malloc(strlen(html_template) + 20);
    sprintf(html, html_template, ip_str);

    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
    free(html);
    return ESP_OK;
}
/**
 * @brief 启动 HTTP 服务器
 * 兼容 ESP-IDF v4.3 及以上版本
 */
void start_webserver() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = HTTP_PORT;

    // ESP-IDF v4.3 及以下版本需通过 menuconfig 设置缓冲区大小
    // 移除不支持的字段
    // 根据需求调整以下字段
    // config.max_uri_handlers = 12;
    // config.uri_match_fn = httpd_uri_match_wildcard;
    // config.max_resp_headers = 16;
    // config.recv_wait_timeout = 10;  // seconds
    // config.send_wait_timeout = 10;  // seconds
    
    // // Increase these values:
    // config.stack_size = 8192;       // Larger stack size
    // 声明 server 局部变量
    httpd_handle_t server = NULL;

    // 注册 PCM 流接口
    static const httpd_uri_t pcm_uri = {
        .uri       = "/pcm",
        .method    = HTTP_GET,
        .handler   = pcm_handler,
        .user_ctx  = NULL
    };

    // 注册主页接口
    static const httpd_uri_t index_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = index_handler,  // 确保函数名正确
        .user_ctx  = NULL
    };

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_register_uri_handler(server, &pcm_uri);
        httpd_register_uri_handler(server, &index_uri);
        ESP_LOGI(TAG, "HTTP 服务器启动在端口 %d", HTTP_PORT);
    }
}


/**
 * @brief 初始化 WiFi（STA 模式）
 */
void wifi_init_sta(const char* ssid, const char* password) {
    wifi_init_nvs();

    // 网络接口初始化
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    // WiFi 初始化
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // 注册事件处理
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, 
                ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, 
                IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    // 配置 WiFi 连接参数
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = "",
            .password = "",
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = {
                .capable = true,
                .required = false
            },
        },
    };
    strncpy((char*)wifi_config.sta.ssid, ssid, sizeof(wifi_config.sta.ssid));
    strncpy((char*)wifi_config.sta.password, password, 
           sizeof(wifi_config.sta.password));

    // 启动 WiFi
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "WiFi 初始化完成，正在连接...");
}




// ==================== 主函数 ====================

void app_main() {
    // 初始化日志
    esp_log_level_set("*", ESP_LOG_INFO);  // 设置全局日志级别
   
    // 初始化音频硬件
    i2s_rx_chan = es7210_i2s_init();  

    es7210_codec_init();
    i2c_scan();  // 查看是否有 0x40 地址设备
    
    // 初始化 WiFi
    wifi_init_sta(WIFI_SSID, WIFI_PASS);
    // 启动 HTTP 服务器
    start_webserver();
    ESP_ERROR_CHECK(i2s_channel_enable(i2s_rx_chan));
    vTaskDelay(pdMS_TO_TICKS(100));
    ESP_LOGI(TAG, "系统初始化完成");
    // 创建音频采集任务
    xTaskCreate(audio_capture_task, "audio_task", 2048, NULL, 5, NULL);
        
}