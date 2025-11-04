cmake_minimum_required(VERSION 3.24)

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found")

    if (NOT DEFINED AC4K_CUDA_ARCHITECTURES)
        set(AC4K_CUDA_ARCHITECTURES "120")
    else()
        if (NOT AC4K_CUDA_ARCHITECTURES STREQUAL "120")
            message(FATAL_ERROR "AC4K_CUDA_ARCHITECTURES only '120' is supported at the moment.")
        endif()
    endif()

    message(STATUS "AC4K using CUDA architectures: ${AC4K_CUDA_ARCHITECTURES}")
    message(STATUS "CMAKE using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    if (CUDAToolkit_VERSION VERSION_LESS "12.8")
        message(FATAL_ERROR "AC4K backend requires CUDA Toolkit 12.8 or higher, found ${CUDAToolkit_VERSION}")
    endif()

    enable_language(CUDA)

    file(GLOB   GGML_HEADERS_CUDA "*.cuh")
    list(APPEND GGML_HEADERS_CUDA "../../include/ggml-ac4k.h")

    file(GLOB   GGML_SOURCES_CUDA "*.cu")
    file(GLOB   SRCS "template-instances/fattn-tile*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/fattn-mma*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmq*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmf*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})

    if (GGML_AC4K_FA_ALL_QUANTS)
        file(GLOB   SRCS "template-instances/fattn-vec*.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        add_compile_definitions(GGML_AC4K_FA_ALL_QUANTS)
    else()
        file(GLOB   SRCS "template-instances/fattn-vec*q4_0-q4_0.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        file(GLOB   SRCS "template-instances/fattn-vec*q8_0-q8_0.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        file(GLOB   SRCS "template-instances/fattn-vec*f16-f16.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
    endif()

    ggml_add_backend_library(ggml-ac4k
                             ${GGML_HEADERS_CUDA}
                             ${GGML_SOURCES_CUDA}
                            )

    add_compile_definitions(GGML_AC4K_PEER_MAX_BATCH_SIZE=${GGML_AC4K_PEER_MAX_BATCH_SIZE})

    set_target_properties(ggml-ac4k PROPERTIES CUDA_ARCHITECTURES "${AC4K_CUDA_ARCHITECTURES}")
    # set_property(TARGET ggml-ac4k PROPERTY CUDA_ARCHITECTURES "")
    # set_property(TARGET ggml-ac4k APPEND PROPERTY CUDA_ARCHITECTURES 120)

    if (GGML_AC4K_GRAPHS)
        add_compile_definitions(GGML_AC4K_USE_GRAPHS)
    endif()

    if (GGML_AC4K_FORCE_MMQ)
        add_compile_definitions(GGML_AC4K_FORCE_MMQ)
    endif()

    if (GGML_AC4K_FORCE_CUBLAS)
        add_compile_definitions(GGML_AC4K_FORCE_CUBLAS)
    endif()

    if (GGML_AC4K_NO_VMM)
        add_compile_definitions(GGML_AC4K_NO_VMM)
    endif()

    if (NOT GGML_AC4K_FA)
        add_compile_definitions(GGML_AC4K_NO_FA)
    endif()

    if (GGML_AC4K_NO_PEER_COPY)
        add_compile_definitions(GGML_AC4K_NO_PEER_COPY)
    endif()

    if (GGML_STATIC)
        if (WIN32)
            # As of 12.3.1 CUDA Toolkit for Windows does not offer a static cublas library
            target_link_libraries(ggml-ac4k PRIVATE CUDA::cudart_static CUDA::cublas)
        else ()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.1")
                target_link_libraries(ggml-ac4k PRIVATE  CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            else()
                target_link_libraries(ggml-ac4k PRIVATE  CUDA::cudart_static CUDA::cublas_static)
            endif()
        endif()
    else()
        target_link_libraries(ggml-ac4k PRIVATE CUDA::cudart CUDA::cublas)
    endif()

    if (GGML_AC4K_NO_VMM)
        # No VMM requested, no need to link directly with the cuda driver lib (libcuda.so)
    else()
        target_link_libraries(ggml-ac4k PRIVATE CUDA::cuda_driver)
    endif()

    set(CUDA_CXX_FLAGS "")

    set(CUDA_FLAGS -use_fast_math -extended-lambda)

    if (GGML_AC4K_DEBUG)
        list(APPEND CUDA_FLAGS -lineinfo)
    endif()

    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        # Options are:
        # - none (not recommended)
        # - speed (nvcc's default)
        # - balance
        # - size
        list(APPEND CUDA_FLAGS -compress-mode=${GGML_AC4K_COMPRESSION_MODE})
    endif()

    if (GGML_FATAL_WARNINGS)
        list(APPEND CUDA_FLAGS -Werror all-warnings)
    endif()

    if (GGML_ALL_WARNINGS AND NOT MSVC)
        set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
        if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
            list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
        endif()

        execute_process(
            COMMAND ${NVCC_CMD} -Xcompiler --version
            OUTPUT_VARIABLE CUDA_CCFULLVER
            ERROR_QUIET
        )

        if (NOT CUDA_CCFULLVER MATCHES clang)
            set(CUDA_CCID "GNU")
            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                OUTPUT_VARIABLE CUDA_CCVER
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()

        message(STATUS "CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

        ggml_get_flags(${CUDA_CCID} ${CUDA_CCVER})
        list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # This is passed to -Xcompiler later
    endif()

    if (NOT MSVC)
        list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
    endif()

    list(JOIN   CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument

    if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
        list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
    endif()

    target_compile_options(ggml-ac4k PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
else()
    message(FATAL_ERROR "CUDA Toolkit not found")
endif()
