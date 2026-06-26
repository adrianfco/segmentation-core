#pragma once

#include <string>
#include <vector>
#include <stdexcept>

namespace seg {

struct Image {
    std::vector<unsigned char> data; // row-major, interleaved RGB
    int width    = 0;
    int height   = 0;
    int channels = 3;
};

// Throws std::runtime_error on failure.
Image load_image(const std::string& path);

// Throws std::runtime_error on failure.
void save_image(const std::string& path, const Image& img);

} // namespace seg
