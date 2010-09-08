#!/bin/csh -f

setenv LD_LIBRARY_PATH /opt/cuda/lib64

make emu=1 dbg=1 verbose=1 clean
if (-f plasticity.bin) then
    rm -vf plasticity.bin plasticity.gold
endif
echo ""
make emu=1 dbg=1 verbose=1

