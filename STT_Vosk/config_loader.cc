#include "config_loader.h"
#include <stdexcept>

VoskSpeechRecognizer::Config ConfigLoader::load(const std::string& yaml_file) {
    VoskSpeechRecognizer::Config config;
    try {
        YAML::Node yaml = YAML::LoadFile(yaml_file);

        config.serial_port = yaml["serial"]["port"].as<std::string>();
        config.baud = yaml["serial"]["baud_rate"].as<int>();
        config.model_path = yaml["model"]["path"].as<std::string>();
        config.sample_rate = yaml["audio"]["sample_rate"].as<int>();
        config.block_size = yaml["audio"]["block_size"].as<size_t>();
        config.queue_size = yaml["audio"]["queue_size"].as<size_t>();
        config.log_file = yaml["logging"]["file"].as<std::string>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to parse YAML file: " + std::string(e.what()));
    }
    return config;
}