#include "vosk_speech_recognizer.h"
#include "thread_pool.h"
#include "config_loader.h"
#include <iostream>

int main() {
    try {
        // 加载 YAML 配置
        VoskSpeechRecognizer::Config config = ConfigLoader::load("config.yaml");

        ThreadPool pool(4);

        auto future = pool.submit([&config]() {
            VoskSpeechRecognizer recognizer(config);
            recognizer.set_result_callback([](const std::string& result) {
                std::cout << "Result: " << result << std::endl;
            });
            recognizer.start();
            std::this_thread::sleep_for(std::chrono::seconds(30));
            recognizer.stop();
        });

        future.get();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}