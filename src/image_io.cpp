#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image_io.hpp"

#include <stb_image.h>
#include <stb_image_write.h>

#include <stdexcept>
#include <filesystem>

namespace seg {

Image load_image(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Input file does not exist: " + path);
    }

    int w, h, c;
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!raw) {
        throw std::runtime_error("Failed to decode image: " + path +
                                 " — " + stbi_failure_reason());
    }

    Image img;
    img.width    = w;
    img.height   = h;
    img.channels = 3;
    img.data.assign(raw, raw + static_cast<size_t>(w) * h * 3);
    stbi_image_free(raw);
    return img;
}

void save_image(const std::string& path, const Image& img) {
    if (img.data.empty() || img.width <= 0 || img.height <= 0) {
        throw std::runtime_error("Cannot save empty image to: " + path);
    }

    // Determine format from extension
    auto ext_pos = path.rfind('.');
    if (ext_pos == std::string::npos) {
        throw std::runtime_error("Output path has no extension: " + path);
    }

    std::string ext = path.substr(ext_pos + 1);
    for (auto& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    int ok = 0;
    if (ext == "png") {
        ok = stbi_write_png(path.c_str(), img.width, img.height, 3,
                            img.data.data(), img.width * 3);
    } else if (ext == "jpg" || ext == "jpeg") {
        ok = stbi_write_jpg(path.c_str(), img.width, img.height, 3,
                            img.data.data(), 90);
    } else if (ext == "bmp") {
        ok = stbi_write_bmp(path.c_str(), img.width, img.height, 3,
                            img.data.data());
    } else {
        throw std::runtime_error("Unsupported output format: " + ext);
    }

    if (!ok) {
        throw std::runtime_error("Failed to write image to: " + path);
    }
}

} // namespace seg
