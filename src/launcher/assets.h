#pragma once

// Files built into infernix-launcher (generated at build time by cmake/EmbedLauncherAssets.cmake):
// the web UI as one gzip-compressed HTML page and the download catalog.

#include <cstddef>

namespace infernix::launcher::assets {

extern const unsigned char kUiGzip[];
extern const std::size_t kUiGzipSize;
extern const unsigned char kCatalogJson[];
extern const std::size_t kCatalogJsonSize;

} // namespace infernix::launcher::assets
