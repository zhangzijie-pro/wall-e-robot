#include <packages/libserialport/include/libserialport.h>
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

template<typename T>
void VoskSpeechRecognizer::ThreadSafeQueue<T>::push(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this](){return queue_.size()<max_size_;});
    queue_.push(std::move(item));
    lock.unlock();
    cond_.notify_one();
}

template<typename T>
bool VoskSpeechRecognizer::ThreadSafeQueue<T>::pop(T &item){
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()){
        logger_->warn("Queue is Empty")
        return false;
    }
    item = std::move(queue_.front());
    queue_.pop();
    lock.unlock();
    cond_.notify_one();
    return true;
}

template<typename T>
size_t VoskSpeechRecognizer::ThreadSafeQueue<T>::size() const{
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
}

void VoskSpeechRecognizer::init_logger(){
    try{
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(config_.log_file, 1024 * 1024 * 5, 3);
        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};

        logger_ = std::make_shared<spdlog::logger>("speech_recognizer", sinks.begin(), sinks.end());
        logger_->set_level(spdlog::level::info);

        auto formatter = std::make_unique<spdlog::pattern_formatter>();
        formatter->add_flag<CustomFormatter>('F').set_pattern("%Y-%m-%d %H:%M:%S.%e - %F - %n - %l - %v");
        logger_->set_formatter(std::move(formatter));
    }catch(const spdlog::spdlog_ex& ex){
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        throw;
    }
}

VoskSpeechRecognizer::VoskSpeechRecognizer(const Config& config): 
    config_(config), model_(nullptr), recognizer_(nullptr), port_(nullptr),
    audio_queue_(config.queue_size),running_(false)
{
    model_ = vosk_model_new(config.model_path.c_str());
    if(!model_){throw std::runtime_error("Failed to load Vosk model");}

    recognizer_ = vosk_recognizer_new(model_, static_cast<float>(config.sample_rate));
    if(!recognizer_){
        vosk_model_free(model_);
        throw std::runtime_error("Failed to create recognizer");
    }
}

VoskSpeechRecognizer::~VoskSpeechRecognizer(){
    stop();
    vosk_recognizer_free(recognizer_);
    vosk_model_free(model_);
    if(port_){
        sp_close(port_);
        sp_free_port(port_);
    }
}

bool VoskSpeechRecognizer::start(){
    if(running_) return false;
    sp_return result = sp_get_port_by_name(config_.serial_port.c_str(), &port_);
    if(result!=SP_OK){
        logger_->error("Serial Search Error...");
        return false;
    }

    result = sp_open(port_, SP_MODE_READ);
    if(result!=SP_OK){
        logger_->error("Serial Open Error...");
        sp_free_port(port_);
        port_ = nullptr;
        return false;
    }

    sp_set_baudrate(port_, config_.baud);
    sp_set_bits(port_, 8);
    sp_set_parity(port_, SP_PARITY_NONE);
    sp_set_stopbits(port_, 1);

    running_=true;
    serial_thread_ = std::thread(&VoskSpeechRecognizer::serial_read_thread, this);
    recognition_thread_ = std::thread(&VoskSpeechRecognizer::recongnzier_thread, this);
    return true;
}

bool VoskSpeechRecognizer::stop(){
    if(!running_) return;
    running_=false;
    if(serial_thread_.joinable()) serial_thread_.join();
    if(recognition_thread_.joinable()) recognition_thread_.join();
}

void VoskSpeechRecognizer::set_result_callback(std::function<void(const std::string&)> callback){
    result_callback_ = std::move(callback);
}

void VoskSpeechRecognizer::serial_read_thread(){
    const size_t bytes_per_block = config_.block_size * sizeof(int16_t);
    std::vector<uint8_t> buffer(bytes_per_block);
    size_t bytes_read=0;

    while(running_){
        size_t bytes_to_read = bytes_per_block-bytes_read;
        int bytes_received = sp_blocking_read(port_, buffer.data()+bytes_read, bytes_to_read, 100);
        if(bytes_received < 0) { logger_->error("Bytes Received Error"); break;}
        bytes_read+=bytes_received;

        if (bytes_read == bytes_per_block) {
            std::vector<int16_t> audio_block(config_.block_size);
            std::memcpy(audio_block.data(), buffer.data(), bytes_per_block);
            audio_queue_.push(std::move(audio_block));
            bytes_read = 0;
        }
    }
}

void VoskSpeechRecognizer::recongnzier_thread(){
    while (running_) {
        std::vector<int16_t> buffer;
        if (audio_queue_.pop(buffer)) {
            if (vosk_recognizer_accept_waveform(recognizer_, (const char *)buffer.data(), buffer.size())) {
                const char* result = vosk_recognizer_result(recognizer_);
                if (result_callback_) {
                    result_callback_(result);
                }
            } else {
                const char* partial = vosk_recognizer_partial_result(recognizer_);
                if (result_callback_) {
                    result_callback_(partial);
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}