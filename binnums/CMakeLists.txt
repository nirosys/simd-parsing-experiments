find_package(cxxopts CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)

add_executable(binnums main.cc)
target_link_libraries(binnums PRIVATE cxxopts::cxxopts fmt::fmt-header-only benchmark::benchmark)
