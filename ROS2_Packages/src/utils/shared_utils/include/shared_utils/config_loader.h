#pragam once
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <string>


namespace shared_utils {
    class ConfigLoader {
    public:
        static YAML::Node load(const std::string &package_name, const std::string &file_name) {
            std::string path = ament_index_cpp::get_package_share_directory(package_name) + "/config/" + file_name;
            return YAML::LoadFile(path);
        }
    };
}