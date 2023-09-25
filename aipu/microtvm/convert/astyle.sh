#!/bin/bash
# This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
# See the copyright file distributed with this work for additional information
# regarding copyright ownership.

for f in $(find ../ -name '*.c' -or -name '*.cpp' -or -name '*.h' -or -name '*.hpp' -type f)
do
    astyle --style=linux -p -s4 -n -U -H -c -S $f
done
