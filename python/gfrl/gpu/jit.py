"""Small NVRTC/driver bridge; compilation and all caches remain local."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import torch


def _check(code, operation):
    if code:
        raise RuntimeError(f'{operation} failed (CUDA status {code})')


class Module:
    def __init__(self, source):
        self.driver = ct.CDLL('libcuda.so.1')
        nvrtc = ct.CDLL('libnvrtc.so')
        major, minor = torch.cuda.get_device_capability()
        nvcc = shutil.which('nvcc')
        if not nvcc:
            raise RuntimeError('nvcc must be on PATH so CUDA headers can be located')
        cuda = Path(nvcc).resolve().parents[1]
        fast_compile = os.environ.get('GFRL_GPU_FAST_COMPILE', 'off')
        options = ['--std=c++20', '--device-as-default-execution-space',
                   f'--gpu-architecture=compute_{major}{minor}', '--fmad=false',
                   *([f'--Ofast-compile={fast_compile}'] if fast_compile not in ('', 'off') else []),
                   f'--include-path={cuda / "include"}',
                   f'--include-path={cuda / "include/cccl"}', '--diag-suppress=177']
        key = hashlib.sha256((source + str(options) + torch.__version__).encode()).hexdigest()
        cache = Path(os.environ.get('GFRL_GPU_CACHE', Path(tempfile.gettempdir()) / 'gfrl-gpu-cache'))
        cache.mkdir(parents=True, exist_ok=True)
        ptx_path = cache / f'{key}.ptx'
        verbose = bool(os.environ.get('GFRL_GPU_VERBOSE'))
        since = perf_counter()
        if verbose:
            print(f'GPU compiler: {key[:12]} NVRTC cache={ptx_path.exists()}', flush=True)
        if not ptx_path.exists():
            program = ct.c_void_p()
            _check(nvrtc.nvrtcCreateProgram(ct.byref(program), source.encode(), b'gfrl_gpu.cu', 0, None, None), 'nvrtcCreateProgram')
            try:
                opt = (ct.c_char_p * len(options))(*(o.encode() for o in options))
                code = nvrtc.nvrtcCompileProgram(program, len(options), opt)
                size = ct.c_size_t()
                nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
                log = ct.create_string_buffer(size.value)
                nvrtc.nvrtcGetProgramLog(program, log)
                if code:
                    (cache / f'{key}.cu').write_text(source)
                    raise RuntimeError(log.value.decode())
                _check(nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), 'nvrtcGetPTXSize')
                ptx = ct.create_string_buffer(size.value)
                _check(nvrtc.nvrtcGetPTX(program, ptx), 'nvrtcGetPTX')
                temporary_path = ptx_path.with_suffix(f'.{os.getpid()}.tmp')
                temporary_path.write_bytes(ptx.raw)
                temporary_path.replace(ptx_path)
            finally:
                nvrtc.nvrtcDestroyProgram(ct.byref(program))
        if verbose:
            print(f'GPU compiler: NVRTC completed in {perf_counter()-since:.1f}s', flush=True)
        jit_level = int(os.environ.get('GFRL_GPU_JIT_OPT', '4'))
        if jit_level not in range(5):
            raise ValueError('GFRL_GPU_JIT_OPT must be in 0..4')
        driver_version = ct.c_int()
        _check(self.driver.cuDriverGetVersion(ct.byref(driver_version)), 'cuDriverGetVersion')
        max_registers = int(os.environ.get('GFRL_GPU_MAX_REGISTERS','0'))
        if not 0 <= max_registers <= 255:
            raise ValueError('GFRL_GPU_MAX_REGISTERS must be in 0..255')
        reg_suffix = f'.R{max_registers}' if max_registers else ''
        cubin_path = cache / f'{key}.driver{driver_version.value}.O{jit_level}{reg_suffix}.cubin' 
        if not cubin_path.exists():
            error_log = ct.create_string_buffer(8192)
            option_ids = [5,6,7]
            option_values = [ct.cast(error_log,ct.c_void_p),ct.c_void_p(len(error_log)),ct.c_void_p(jit_level)]
            if max_registers:
                option_ids.append(0)
                option_values.append(ct.c_void_p(max_registers))
            jit_options = (ct.c_int * len(option_ids))(*option_ids)
            jit_values = (ct.c_void_p * len(option_ids))(*option_values)
            link = ct.c_void_p()
            _check(self.driver.cuLinkCreate_v2(len(option_ids),jit_options,jit_values,ct.byref(link)), 'cuLinkCreate')
            try:
                ptx = ct.create_string_buffer(ptx_path.read_bytes())
                status = self.driver.cuLinkAddData_v2(link,1,ct.cast(ptx,ct.c_void_p),ct.c_size_t(len(ptx)),
                                                     b'gfrl_gpu.ptx',0,None,None)
                if status:
                    raise RuntimeError(f'cuLinkAddData failed ({status}): {error_log.value.decode()}')
                binary = ct.c_void_p(); size = ct.c_size_t()
                status = self.driver.cuLinkComplete(link,ct.byref(binary),ct.byref(size))
                if status:
                    raise RuntimeError(f'cuLinkComplete failed ({status}): {error_log.value.decode()}')
                temporary_path = cubin_path.with_suffix(f'.{os.getpid()}.tmp')
                temporary_path.write_bytes(ct.string_at(binary,size.value))
                temporary_path.replace(cubin_path)
            finally:
                self.driver.cuLinkDestroy(link)
        if verbose:
            print(f'GPU compiler: native binary ready in {perf_counter()-since:.1f}s',flush=True)
        self.module = ct.c_void_p()
        native_binary = ct.create_string_buffer(cubin_path.read_bytes())
        _check(self.driver.cuModuleLoadData(ct.byref(self.module),ct.cast(native_binary,ct.c_void_p)), 'cuModuleLoadData(cubin)')
        self.functions = {}

    def attributes(self,name):
        if name not in self.functions:
            function=ct.c_void_p()
            _check(self.driver.cuModuleGetFunction(ct.byref(function),self.module,name.encode()),'cuModuleGetFunction')
            self.functions[name]=function
        result={}
        for key,attribute in [('registers',4),('local_bytes',3),('shared_bytes',1),('max_threads',0)]:
            value=ct.c_int()
            _check(self.driver.cuFuncGetAttribute(ct.byref(value),attribute,self.functions[name]),'cuFuncGetAttribute')
            result[key]=value.value
        return result

    def launch(self, name, args=(), *, n=1, block=32):
        if name not in self.functions:
            function = ct.c_void_p()
            _check(self.driver.cuModuleGetFunction(ct.byref(function), self.module, name.encode()), 'cuModuleGetFunction')
            self.functions[name] = function
        parameters = (ct.c_void_p * len(args))(*(ct.cast(ct.byref(a), ct.c_void_p) for a in args))
        stream = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(self.driver.cuLaunchKernel(self.functions[name], (n+block-1)//block, 1, 1,
                                         block, 1, 1, block*ct.sizeof(ct.c_void_p), stream, parameters, None), name)


def pointer(tensor):
    return ct.c_void_p(tensor.data_ptr())
