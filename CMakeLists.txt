cmake_minimum_required(VERSION 3.15...3.27)
project(bitquant_ext)

# Tìm kiếm Python và nanobind
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Định nghĩa module C++ có tên là _gemv
nanobind_add_module(_gemv
    bitquant/kernel/binder.cpp 
    bitquant/kernel/gemv.cpp
)

# Bật cờ AVX2 tùy theo hệ điều hành
if(MSVC)
    target_compile_options(_gemv PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(_gemv PRIVATE -O3 -mavx2)
endif()

# Yêu cầu CMake cài đặt file thư viện (.so hoặc .pyd) vào đúng thư mục python package
install(TARGETS _gemv DESTINATION bitquant/kernel)