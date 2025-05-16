#ifndef VOSK_SPEECH_RECOGNIZER_H
#define VOSK_SPEECH_RECOGNIZER_H

#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

struct sp_port;

class VoskSpeechRecognizer{
public:
    struct Config{
        std::string model_path;
        std::string serial_port;
        int baud = 115200;
        int sample_rate = 16000;
        size_t block_size = 1600;
        size_t queue_size = 50;
        std::string log_file = "./log_file";
    };

    explicit VoskSpeechRecognizer(const Config &config);
    ~VoskSpeechRecognizer();

    bool start();

    bool stop();

    void set_result_callback(std::function<void(const std::string&)> callback);

private:
    template <typename T>
    class ThreadSafeQueue{
    private:
        std::queue<T> queue_;
        mutable std::mutex mutex_;
        std::condition_variable cond_;
        size_t max_size_;

    public:
        ThreadSafeQueue(size_t max_size=100): max_size_(max_size) {}
        void push(T item);
        bool pop(T& item);
        size_t size() const;
    };

    void serial_read_thread();
    void recongnzier_thread();
    void init_logger();

    Config config_;
    struct VoskModel* model_;
    struct VoskRecognizer* recognizer_;

    struct sp_port *port_;
    ThreadSafeQueue<std::vector<int16_t>> audio_queue_;
    std::thread serial_thread_;
    std::thread recognition_thread_;
    std::atomic<bool> running_;
    std::function<void(const std::string&)> result_callback_;
    std::shared_ptr<spdlog::logger> logger_;
};


#endif