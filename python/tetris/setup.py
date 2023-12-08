#!/usr/bin/env python3

from setuptools import Extension, setup
import numpy

sources = ['board.cpp', 'tetris.cpp', 'module.cpp']
args = ['-std=c++20', '-DLINE_CAP=430', '-DADJ_DELAY=18', '-DTAP_SPEED=Tap30Hz', '-mbmi2']
#args = ['-std=c++20', '-DLINE_CAP=430', '-DADJ_DELAY=18', '-DTAP_SPEED=Tap30Hz', '-mbmi2', '-O1']
#args = ['-std=c++20', '-DLINE_CAP=430', '-DADJ_DELAY=18', '-DTAP_SPEED=Tap30Hz', '-mbmi2', '-fsanitize=address', '-fsanitize=undefined', '-O1']

name = 'tetris'
module = Extension(
        name,
        sources = sources,
        include_dirs = [numpy.get_include()],
        extra_compile_args = args,
        extra_link_args = args)
setup(name = name, ext_modules = [module])
