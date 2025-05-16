#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include "vosk_speech_recognizer.h"
#include <string>
#include <yaml-cpp/yaml.h>

class ConfigLoader {
public:
    static VoskSpeechRecognizer::Config load(const std::string& yaml_file);
};

#endif // CONFIG_LOADER_H