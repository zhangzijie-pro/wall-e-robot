#pragma once

#include <string>
#include <array>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <functional>
#include <unordered_set>
#include <spdlog/spdlog.h>

struct VoskModel;
struct VoskRecognizer;

class VoskSpeechRecognizer {
public:
    struct Config {
        std::string model_path;
        int sample_rate;
        int buffer_block_size;
        std::string log_file;
    };

    explicit VoskSpeechRecognizer(const Config& config);
    ~VoskSpeechRecognizer();

    void push_audio(const std::vector<int16_t>& audio_block);
    bool start();
    bool stop();
    void set_result_callback(std::function<void(const std::string&)> callback);

private:
    void init_logger();
    void recognizer_thread();

    // 三缓冲结构
    static constexpr int BUFFER_COUNT = 3;
    std::array<std::vector<int16_t>, BUFFER_COUNT> buffers_;
    int active_index_;  // 正在写入的缓冲区下标
    std::unordered_set<int> filled_indices_;  // 标记已写满等待处理的缓冲区
    std::mutex buffer_mutex_;
    std::condition_variable buffer_cv_;

    Config config_;
    std::shared_ptr<spdlog::logger> logger_;
    VoskModel* model_;
    VoskRecognizer* recognizer_;

    std::function<void(const std::string&)> result_callback_;

    std::thread recognition_thread_;
    bool running_;
};
