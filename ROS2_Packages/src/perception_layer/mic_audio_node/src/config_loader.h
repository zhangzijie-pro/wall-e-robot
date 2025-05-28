#pragma once
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <string>


namespace shared_utils {
    class ConfigLoader {
    public:
        static YAML::Node load(const std::string &config_path = "config.yaml") {
            try {
                std::string base_path = ament_index_cpp::get_package_share_directory("ROS2_Package");
                std::string full_path = base_path + "/config/" + config_path;
                return YAML::LoadFile(full_path);
            }
        }
    };
}