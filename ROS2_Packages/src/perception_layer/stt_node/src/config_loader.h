#pragma once

#include <yaml.h>
#include <string>
#include <map>
#include <stdexcept>
#include <unistd.h>
#include <limits.h>

namespace shared_utils {
    class ConfigLoader {
    public:
        static std::map<std::string, std::string> load(const std::string& config_path = "config.yaml") {
            char cwd[PATH_MAX];
            if (getcwd(cwd, sizeof(cwd)) == nullptr) {
                throw std::runtime_error("Failed to get current working directory");
            }
            std::string full_path = std::string(cwd) + "/config/" + config_path;

            std::map<std::string, std::string> config;

            yaml_parser_t parser;
            yaml_event_t event;
            FILE* file = fopen(full_path.c_str(), "rb");
            if (!file) {
                throw std::runtime_error("Failed to open config file: " + full_path);
            }

            if (!yaml_parser_initialize(&parser)) {
                fclose(file);
                throw std::runtime_error("Failed to initialize YAML parser");
            }

            yaml_parser_set_input_file(&parser, file);

            std::string key;
            bool in_map = false;
            bool in_mic_audio_node = false;

            while (1) {
                if (!yaml_parser_parse(&parser, &event)) {
                    yaml_parser_delete(&parser);
                    fclose(file);
                    throw std::runtime_error("YAML parsing error: " + std::string(parser.problem));
                }

                switch (event.type) {
                    case YAML_MAPPING_START_EVENT:
                        if (!in_mic_audio_node) {
                            in_mic_audio_node = true; // 进入顶级映射
                        }
                        break;
                    case YAML_SCALAR_EVENT:
                        if (in_mic_audio_node) {
                            if (!in_map && std::string(reinterpret_cast<char*>(event.data.scalar.value)) == "mic_audio_node") {
                                in_map = false; // 等待下一个标量作为键
                            } else if (in_mic_audio_node && !in_map) {
                                key = reinterpret_cast<char*>(event.data.scalar.value);
                                in_map = true;
                            } else if (in_mic_audio_node && in_map) {
                                config[key] = reinterpret_cast<char*>(event.data.scalar.value);
                                in_map = false;
                            }
                        }
                        break;
                    case YAML_MAPPING_END_EVENT:
                        if (in_mic_audio_node) {
                            in_mic_audio_node = false;
                        }
                        break;
                    case YAML_STREAM_END_EVENT:
                        yaml_event_delete(&event);
                        goto done;
                    default:
                        break;
                }

                yaml_event_delete(&event);
            }

        done:
            yaml_parser_delete(&parser);
            fclose(file);
            return config;
        }
    };
} // namespace shared_utils