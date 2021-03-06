# 3.8 is specified since this is the first version with CUDA support
CMAKE_MINIMUM_REQUIRED (VERSION 3.8 FATAL_ERROR)

SET (CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Set target project name
SET (TRGT spatter)

# Set default values
SET (USE_CUDA 0 CACHE BOOL "If set, builds with CUDA support")
SET (USE_OPENCL 0 CACHE BOOL "If set, builds with OpenCL support")
SET (USE_OPENMP 0 CACHE BOOL "If set, builds with OpenMP support")
SET (USE_SERIAL 0 CACHE BOOL "If set, builds with Serial support")
SET (USE_PAPI 0 CACHE BOOL "If set, builds with PAPI support")

ADD_DEFINITIONS (-DSG_DATA_TYPE=double)

# Check for support for at least one backend
IF (NOT (USE_CUDA OR USE_OPENCL OR USE_OPENMP OR USE_SERIAL))
    MESSAGE (
        FATAL_ERROR
            "You must build with support for at least one backend. Pass at least one of -DUSE_CUDA=1 -DUSE_OPENCL=1 -DUSE_SERIAL=1 or -DUSE_OPENMP=1 to cmake."
    )
ENDIF ()

# Set CUDA compiler
IF (USE_CUDA)
    SET (CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    SET (CMAKE_CUDA_COMPILER_ENV_VAR "CUDAC")
ENDIF ()

# Declare project languages
IF (USE_CUDA)
    PROJECT (${TRGT} C CXX CUDA)
ELSE ()
    PROJECT (${TRGT} C CXX)
ENDIF ()

# Debug function to check all the variables in the CMakeFile
MACRO (print_all_variables)
    MESSAGE (
        STATUS "print_all_variables------------------------------------------{"
    )
    GET_CMAKE_PROPERTY (_variableNames VARIABLES)
    FOREACH (_variableName ${_variableNames})
        MESSAGE (STATUS "${_variableName}=${${_variableName}}")
    ENDFOREACH ()
    MESSAGE (
        STATUS "print_all_variables------------------------------------------}"
    )
ENDMACRO ()


# Check that appropriate compiler versions are used
IF (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        MESSAGE (FATAL_ERROR "GCC version must be at least 4.9.1 to run the OpenMP 4.0 backend")
ENDIF ()


#Add specific flags for the CodeXL compiler
IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "XL")
	SET(IBMXL_COMPILE_FLAGS "-qenablevmx -qtune=pwr9")
	SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${IBMXL_COMPILE_FLAGS}")
    SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=IBM")
ENDIF ()

IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "Cray") 
    SET (OPTIMIZATIONS "-O3 -h vector3 -h cache3 -h scalar3")
    #SET (OPTIMIZATIONS "-G0 -O0")
    #SET (ALLWARNINGS "-Wall -Wextra -ansi -pedantic")
    #SET (ALLWARNINGS "-h msglevel_2")
    SET (CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OPTIMIZATIONS} ${ALLWARNINGS} -hlist=m -hlist=d ")
    SET (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OPTIMIZATIONS} -hlist=m -D__CRAYC__")
    SET (CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=Cray -D__CRAYC__")
ENDIF ()

IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel") 
    SET (OpenMP_C_FLAGS "${OpenMP_C_FLAGS} -xHost")
    SET (OpenMP_C_FLAGS "${OpenMP_CXX_FLAGS} -xHost")
ENDIF ()

 
IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU") 
    SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=GNU -O3")
   #Use sanitize=address for debugging of access issues 
   #SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fsanitize=address")
ENDIF ()

IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang") 
    SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DSPAT_C_NAME=Clang")
ENDIF ()


#Add configure-specified flags
SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")

# Enable OpenCL
IF (USE_OPENCL)
    ADD_DEFINITIONS (-DUSE_OPENCL)
    # Point the compiler to the include and library directories
    INCLUDE_DIRECTORIES ($ENV{OCL_INCL} src/opencl)
    # include_directories(/usr/lib/gcc/x86_64-linux-gnu/5/include/)
    LINK_DIRECTORIES ($ENV{OCL_LIB})
    # Pull the OpenCL-specific files into the build
    FILE (GLOB BACKEND_C_FILES_CL src/opencl/*.c)
    FILE (GLOB BACKEND_H_FILES_CL src/opencl/*.h)

    # Copy over the OpenCL kernels used with the binary
    FILE (GLOB OCL_KERNELS src/opencl/*.cl)
    FILE (
        COPY
        ${OCL_KERNELS}
        include/sgtype.h
        DESTINATION
        ${CMAKE_CURRENT_BINARY_DIR}/kernels/
    )

ENDIF ()

# Enable Serial backend
IF (USE_SERIAL)
    ADD_DEFINITIONS (-DUSE_SERIAL)
    INCLUDE_DIRECTORIES (src/serial)
    # Pull the serial files and kernels into the build
    FILE (GLOB BACKEND_C_FILES_SERIAL src/serial/*.c)
    FILE (GLOB BACKEND_H_FILES_SERIAL src/serial/*.h)
    MESSAGE ("Using serial backend")
ENDIF ()

# Enable OpenMP
IF (USE_OPENMP)
    FIND_PACKAGE (OpenMP REQUIRED)
    IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "Cray") 
        SET (OpenMP_C_FLAGS "${OpenMP_C_FLAGS} -homp")
        SET (OpenMP_C_FLAGS "${OpenMP_CXX_FLAGS} -homp")
    ENDIF ()
    IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU") 
        SET (OpenMP_C_FLAGS "${OpenMP_C_FLAGS} -fopenmp")
        SET (OpenMP_C_FLAGS "${OpenMP_CXX_FLAGS} -fopenmp")
    ENDIF ()
    IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel") 
        SET (OpenMP_C_FLAGS "${OpenMP_C_FLAGS} -qopenmp")
        SET (OpenMP_C_FLAGS "${OpenMP_CXX_FLAGS} -qopenmp")
    ENDIF ()
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

    # enable pragma regions in the code for OpenMP
    ADD_DEFINITIONS (-DUSE_OPENMP)
    ADD_DEFINITIONS (-DUSE_OMP_SIMD)
    INCLUDE_DIRECTORIES (src/openmp)

    # Pull the OpenMP-specific files and kernels into the build
    FILE (GLOB BACKEND_C_FILES_OMP src/openmp/*.c)
    FILE (GLOB BACKEND_H_FILES_OMP src/openmp/*.h)
ENDIF ()

# Enable CUDA
IF (USE_CUDA)
    ADD_DEFINITIONS (-DUSE_CUDA)
    INCLUDE_DIRECTORIES (src/cuda)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I/usr/local/cuda/include")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/cuda/include")
    SET (
        CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    IF (USE_OPENMP)
        SET (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    ENDIF ()

    FILE (GLOB CUDA_CU_FILES src/cuda/*.cu)
    FILE (GLOB CUDA_C_FILES src/cuda/*.c)
    FILE (GLOB CUDA_H_FILES src/cuda/*.h)
ENDIF ()

IF (USE_PAPI) 
    SET (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
    INCLUDE_DIRECTORIES (src/papi)
    ADD_DEFINITIONS (-DUSE_PAPI)
    #Use the included FindPAPI cmake module
    INCLUDE (FindPAPI)
    FIND_PACKAGE (PAPI REQUIRED)
    #message("Found PAPI libraries at ${PAPI_INCLUDE_DIRS}")
    INCLUDE_DIRECTORIES(${PAPI_INCLUDE_DIRS})
    FILE (GLOB PAPI_C_FILES src/papi/*.c)
    FILE (GLOB PAPI_H_FILES src/papi/*.h)
ENDIF ()

# Include the location of stddef.h include_directories(/usr/include/linux/)

# Glob for the source files and headers in the current directory and kernels
FILE (GLOB C_FILES src/*.c)
FILE (GLOB H_FILES include/*.h)
SET (
    SOURCE_FILES
    ${C_FILES}
    ${H_FILES}
    ${BACKEND_C_FILES_CL}
    ${BACKEND_H_FILES_CL}
    ${BACKEND_C_FILES_OMP}
    ${BACKEND_H_FILES_OMP}
    ${BACKEND_C_FILES_SERIAL}
    ${BACKEND_H_FILES_SERIAL}
    ${CUDA_CU_FILES}
    ${CUDA_C_FILES}
    ${PAPI_C_FILES}
    ${PAPI_H_FILES}
)

# Specify the executable and source files
ADD_EXECUTABLE (${TRGT} ${SOURCE_FILES})
#Prefer C11 standard to allow for loop initialization and aligned_alloc
set_property(TARGET ${TRGT} PROPERTY C_STANDARD 11)

# Print out debug info print_all_variables()

# Link with the appropriate libraries
IF (USE_OPENCL)
    TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC OpenCL)
ENDIF ()

# Link math library for json
TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC m)

IF (USE_CUDA)
    FIND_LIBRARY (
        CUDART_LIBRARY cudart
        HINTS
            /usr/local/cuda/targets/x86_64-linux/lib/ /usr/local/cuda/lib64/
    )
    TARGET_LINK_LIBRARIES (${TRGT} LINK_PUBLIC ${CUDART_LIBRARY})
ENDIF ()

#Include PAPI libraries, if defined
IF (USE_PAPI)
    TARGET_LINK_LIBRARIES (spatter LINK_PUBLIC ${PAPI_LIBRARIES})
    # Explicitly link libdl because the Cray compiler refuses to link papi dynaimically...
    # (Static libraries carry no dependency info, and libpapi depends on libdl)
    TARGET_LINK_LIBRARIES (spatter LINK_PUBLIC dl)
ENDIF ()

# Copy over the test scripts
FILE (GLOB TEST_SCRIPTS tests/*.sh)
FILE (COPY ${TEST_SCRIPTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Add a command to put some important info in environment variables 
# to be output by spatter
message (STATUS "compiler version is ${CMAKE_C_COMPILER_VERSION}")
message (STATUS "compiler is ${CMAKE_C_COMPILER}")
SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSPAT_C=\"${CMAKE_C_COMPILER}\"")
SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSPAT_C_VER=\"${CMAKE_C_COMPILER_VERSION}\"")

# Note - This file can be reformatted with cmake-format CMakeLists.txt -i
# --command-case=upper --keyword-case=upper --dangle-parens --tab-size=4
# --separate-ctrl-name-with-space --separate-fn-name-with-space If you do not
# have cmake-format installed, install it with pip install --user cmake_format
