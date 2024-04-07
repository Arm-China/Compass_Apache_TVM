#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.

for f in $(find ../ -name '*.c' -or -name '*.cpp' -or -name '*.h' -or -name '*.hpp' -type f)
do
    astyle --style=linux -p -s4 -n -U -H -c -S $f
done
