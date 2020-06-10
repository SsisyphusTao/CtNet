from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
    name='dcn_op_v2',
    ext_modules=[
        CUDAExtension(
        'dcn_op_v2',
        sources=['dcn_v2_cuda.cc','dcn_v2_im2col_cuda.cu'],
        # include_dirs=['/usr/local/cuda/include'],
        # library_dirs=['/usr/local/cuda/lib64']
        # extra_compile_args=extra_compile_args 
    )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)