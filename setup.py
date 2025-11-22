import os
import torch
import glob
import subprocess
from os import path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import (
    CUDA_HOME,
    CUDNN_HOME,
    IS_HIP_EXTENSION,
    CppExtension,
    CUDAExtension,
    BuildExtension
)

ROOT_DIR = os.path.dirname(__file__)
NAME = "flash_ops"
ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CUR_DIR = path.dirname(path.abspath(__file__))

torch_include_dirs = torch.utils.cpp_extension.include_paths()
torch_library_dirs = torch.utils.cpp_extension.library_paths()

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def get_version() -> str:
    return "v0.0.1"

def build_for_cuda():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

    debug = os.getenv('FLASH_OPS_DEBUG', 0)
    max_jobs = os.getenv('MAX_JOBS', 8)

    # Compiler flags.
    NVCC_FLAGS = {
       'nvcc' : [
         '--threads={}'.format(max_jobs),
        #  '-gencode', 'arch=compute_89,code=sm_89',  # Specify compute capability
         '-gencode', 'arch=compute_80,code=sm_80',
         '--ptxas-options=-v',  # Verbose PTX assembly output
         '--use_fast_math',
         '-Xptxas',
        ],
       "cxx": [
            "-std=c++17",
            "-DENABLE_BF16",
            "-DBUILD_WITH_NVIDIA",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "-Wno-unused-value"
            # "-save-temps"
            # "-DCUTLASS_DEBUG_TRACE_LEVEL=1
       ],
    }
    NVCC_FLAGS["cxx"] += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

    extra_link_args=[
        '-Wl,-rpath,$ORIGIN',  # Linux 运行时库搜索路径
        '-Wl,-rpath,' + ':'.join(torch_library_dirs)  # 添加 Torch 库路径
    ],

    if debug == "1":
        NVCC_FLAGS["cxx"] +=["-g", "-O1"]
        NVCC_FLAGS["nvcc"] +=['-O1', '--maxrregcount=32', '-G', '-lineinfo', '--ptxas-options=-v',]
    else:
        NVCC_FLAGS["cxx"] += ["-O3"]
        NVCC_FLAGS["nvcc"] += ["-O3"]

    include_dirs.append(os.path.join("csrc"))
    sources = (
        [
            os.path.join("csrc", "py_init.cpp"),
            os.path.join("csrc", "norm", "norm.cpp"),
            os.path.join("csrc", "norm", "layernorm.cu"),
            os.path.join("csrc", "norm", "rmsnorm.cu"),
            os.path.join("csrc", "linear", "gemm_mma.cu"),
            os.path.join("csrc", "linear", "multi_stage_mma.cu"),
            os.path.join("csrc", "linear", "linear.cpp"),
        ]
    )

    if CUDNN_HOME is None:
        try:
            # Try to use the bundled version of CUDNN with PyTorch installation.
            # This is also used in CI.
            from nvidia import cudnn
        except ImportError:
            cudnn = None


        try:
            from nvidia import cublas
        except ImportError:
            cublas = None

        if cublas is not None:
            cublas_dir = os.path.dirname(cublas.__file__)
            print("Using CUBLAS from {}".format(cublas_dir))
            include_dirs.append(os.path.join(cublas_dir, "include"))

    define_macros=[
        ("WITH_CUDA", None)
    ]

    ext_modules.append(
        CUDAExtension(
            "flash_ops._C",
            sources=sources,
            include_dirs=[os.path.abspath(p) for p in include_dirs],
            define_macros=define_macros,
            extra_compile_args=NVCC_FLAGS,
            library_dirs=[os.path.abspath(p) for p in library_dirs],
            extra_objects=object_files,
            extra_link_args=["-Wl,-rpath,$ORIGIN", "-Wl,-rpath,${torch_library_dirs}"],
        )
    )

ext_modules = []
library_dirs = []
object_files = []
include_dirs = []

build_for_cuda()

setup(
    name="flash_ops",
    version=get_version(),
    license="Apache 2.0",
    description="Flash AI Ops Toy",
    url="https://github.com/duanyzhi/flash-ops.git",
    author="duanyzhi",
    packages=find_packages(exclude=("release")),
    install_requires=[
        #'torch==2.8.0',
    ],
    ext_modules = ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

