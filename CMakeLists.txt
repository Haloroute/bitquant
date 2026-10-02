cmake_minimum_required(VERSION 3.15...3.27)
project(bitquant_ext)

# Search for Python and nanobind packages
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Define the Python modules and expose the functions
nanobind_add_module(_gemv
    bitquant/kernel/gemv/binder.cpp 
    bitquant/kernel/gemv/gemv.cpp
)
nanobind_add_module(_pack
    bitquant/kernel/pack/binder.cpp 
    bitquant/kernel/pack/pack.cpp
)

# Raise the optimization level to O3 and enable AVX2 instructions for both modules
if(MSVC)
    target_compile_options(_gemv PRIVATE /O2 /arch:AVX2)
    target_compile_options(_pack PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(_gemv PRIVATE -O3 -mavx2)
    target_compile_options(_pack PRIVATE -O3 -mavx2)
endif()

# Install the modules to the specified destination
install(TARGETS _gemv DESTINATION bitquant/kernel)
install(TARGETS _pack DESTINATION bitquant/kernel)