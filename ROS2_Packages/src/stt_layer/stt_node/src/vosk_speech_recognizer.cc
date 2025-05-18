#include "vosk_speech_recognizer.h"
#include <vosk_api.h>
#include <iostream>
#include <cstring>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/pattern_formatter.h>

class CustomFormatter : public spdlog::custom_flag_formatter {
public:
    void format(const spdlog::details::log_msg& msg, const std::tm& tm, spdlog::memory_buf_t& dest) override {
        std::string basename = msg.source.filename;
        auto last_slash = basename.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            basename = basename.substr(last_slash + 1);
        }
        dest.append(basename);
    }

    std::unique_ptr<custom_flag_formatter> clone() const override {
        return spdlog::details::make_unique<CustomFormatter>();
    }
};

void VoskSpeechRecognizer::init_logger() {
    try {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(config_.log_file, 1024 * 1024 * 5, 3);
        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};

        logger_ = std::make_shared<spdlog::logger>("speech_recognizer", sinks.begin(), sinks.end());
        logger_->set_level(spdlog::level::info);

        auto formatter = std::make_unique<spdlog::pattern_formatter>();
        formatter->add_flag<CustomFormatter>('F').set_pattern("%Y-%m-%d %H:%M:%S.%e - %F - %n - %l - %v");
        logger_->set_formatter(std::move(formatter));
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        throw;
    }
}

VoskSpeechRecognizer::VoskSpeechRecognizer(const Config& config)
    : config_(config), model_(nullptr), recognizer_(nullptr), running_(false),
      active_index_(0), filled_buffers_(0) {
    model_ = vosk_model_new(config.model_path.c_str());
    if (!model_) {
        throw std::runtime_error("Failed to load Vosk model");
    }

    recognizer_ = vosk_recognizer_new(model_, static_cast<float>(config.sample_rate));
    if (!recognizer_) {
        vosk_model_free(model_);
        throw std::runtime_error("Failed to create recognizer");
    }

    buffers_.resize(3); // triple buffer
    for (auto& buf : buffers_) buf.reserve(config.buffer_block_size);

    init_logger();
    logger_->info("Recognizer initialized with model: {}", config.model_path);
}

VoskSpeechRecognizer::~VoskSpeechRecognizer() {
    stop();
    vosk_recognizer_free(recognizer_);
    vosk_model_free(model_);
}

void VoskSpeechRecognizer::push_audio(const std::vector<int16_t>& audio_block) {
    std::unique_lock<std::mutex> lock(buffer_mutex_);
    if (buffers_[active_index_].size() + audio_block.size() > config_.buffer_block_size) {
        filled_indices_.push_back(active_index_);
        filled_buffers_++;
        active_index_ = (active_index_ + 1) % 3;
        buffers_[active_index_].clear();
        buffer_cond_.notify_one();
    }
    buffers_[active_index_].insert(buffers_[active_index_].end(), audio_block.begin(), audio_block.end());
}

bool VoskSpeechRecognizer::start() {
    if (running_) return false;
    running_ = true;
    recognition_thread_ = std::thread(&VoskSpeechRecognizer::recognizer_thread, this);
    logger_->info("Recognizer thread started.");
    return true;
}

bool VoskSpeechRecognizer::stop() {
    if (!running_) return false;
    running_ = false;
    buffer_cond_.notify_all();
    if (recognition_thread_.joinable()) recognition_thread_.join();
    logger_->info("Recognizer thread stopped.");
    return true;
}

void VoskSpeechRecognizer::set_result_callback(std::function<void(const std::string&)> callback) {
    result_callback_ = std::move(callback);
}

void VoskSpeechRecognizer::recognizer_thread() {
    while (running_ || filled_buffers_ > 0) {
        int index = -1;
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            buffer_cond_.wait(lock, [this]() { return !filled_indices_.empty() || !running_; });
            if (!filled_indices_.empty()) {
                index = filled_indices_.front();
                filled_indices_.pop_front();
                filled_buffers_--;
            }
        }

        if (index >= 0) {
            const std::vector<int16_t>& buffer = buffers_[index];
            if (vosk_recognizer_accept_waveform(recognizer_, (const char*)buffer.data(), buffer.size() * sizeof(int16_t))) {
                const char* result = vosk_recognizer_result(recognizer_);
                logger_->info("Final result: {}", result);
                if (result_callback_) result_callback_(result);
            } else {
                const char* partial = vosk_recognizer_partial_result(recognizer_);
                logger_->debug("Partial result: {}", partial);
                if (result_callback_) result_callback_(partial);
            }
            buffers_[index].clear();
        }
    }
}