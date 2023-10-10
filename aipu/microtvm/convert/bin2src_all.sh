#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 Arm Technology (China) Co. Ltd.

NETCASEPATH=./cases
BMGBINPATH=./output/gbin4bm
#BMGBINPATH=../aipu_demo/src/gbin4bm
GBIN4BM_HEADER=${BMGBINPATH}/gbin4bm.h
GBIN=aipu.bin
INPUT_0=input0.bin
INPUT_1=input1.bin
GT=output.bin

rm ./output -rf
mkdir -p ${BMGBINPATH}

echo "/*" > ${GBIN4BM_HEADER}
echo " * Auto-generated file. Do not Modify it directly!" >> ${GBIN4BM_HEADER}
echo " */" >> ${GBIN4BM_HEADER}
echo "#ifndef _GBIN_4_BM_H_" >> ${GBIN4BM_HEADER}
echo -e "#define _GBIN_4_BM_H_\n" >> ${GBIN4BM_HEADER}
echo -e "#include <stdint.h>\n" >> ${GBIN4BM_HEADER}

if [ -d ${NETCASEPATH} ]; then
    dir=$(ls -l ${NETCASEPATH} |awk '/^d/ {print $NF}')
    for case_name in $dir
    do
        rm -rf ${BMGBINPATH}/${case_name}
        mkdir -p ${BMGBINPATH}/${case_name}
        GBIN4BM_SRC=${BMGBINPATH}/${case_name}/gbin4bm.c

        INPUT_NUM=1
        # gbin_size=$((`ls -l ${NETCASEPATH}/${case_name}/${GBIN} | awk '{print $5}'`))
        input0_size=$((`ls -l ${NETCASEPATH}/${case_name}/${INPUT_0} | awk '{print $5}'`))
        gt_size=$((`ls -l ${NETCASEPATH}/${case_name}/${GT} | awk '{print $5}'`))

        # ./bin2arr.py -O ${BMGBINPATH}/${case_name}/${case_name}_bin.c -l 16 -a ${case_name}_bin ${NETCASEPATH}/${case_name}/${GBIN}
        ./bin2arr.py -O ${BMGBINPATH}/${case_name}/${case_name}_in0.c -l 16 -a ${case_name}_in0 ${NETCASEPATH}/${case_name}/${INPUT_0}
        ./bin2arr.py -O ${BMGBINPATH}/${case_name}/${case_name}_gt.c -l 16 -a ${case_name}_gt ${NETCASEPATH}/${case_name}/${GT}

        if [ -f ${NETCASEPATH}/${case_name}/${INPUT_1} ]; then
            echo "multiple inputs: case ${case_name}"
            INPUT_NUM=2
            input1_size=$((`ls -l ${NETCASEPATH}/${case_name}/${INPUT_1} | awk '{print $5}'`))
            ./bin2arr.py -O ${BMGBINPATH}/${case_name}/${case_name}_in1.c -l 16 -a ${case_name}_in1 ${NETCASEPATH}/${case_name}/${INPUT_1}
        fi

        # echo "extern uint8_t ${case_name}_bin[${gbin_size}];" >> ${GBIN4BM_HEADER}
        echo "extern uint8_t ${case_name}_in0[${input0_size}];" >> ${GBIN4BM_HEADER}
        if [ $INPUT_NUM -eq 2 ]; then
            echo "extern uint8_t ${case_name}_in1[${input1_size}];" >> ${GBIN4BM_HEADER}
        fi
        echo -e "extern uint8_t ${case_name}_gt[${gt_size}];\n" >> ${GBIN4BM_HEADER}

        echo "/*" > ${GBIN4BM_SRC}
        echo " * Auto-generated file. Do not Modify it directly!" >> ${GBIN4BM_SRC}
        echo -e " */\n" >> ${GBIN4BM_SRC}
        echo -e "#include <stdio.h>" >> ${GBIN4BM_SRC}
        echo -e "#include <stdint.h>" >> ${GBIN4BM_SRC}
        echo -e "#include \"../gbin4bm.h\"" >> ${GBIN4BM_SRC}

		echo "extern uint8_t* input0;" >> ${GBIN4BM_HEADER}
		echo "extern uint8_t* input1;" >> ${GBIN4BM_HEADER}
		echo "extern uint8_t* gt;" >> ${GBIN4BM_HEADER}

        echo "uint8_t* input0 = ${case_name}_in0;" >> ${GBIN4BM_SRC}
        if [ $INPUT_NUM -eq 2 ]; then
            echo "uint8_t* input1 = ${case_name}_in1;" >> ${GBIN4BM_SRC}
        else
            echo "uint8_t* input1 = NULL;" >> ${GBIN4BM_SRC}
        fi
        echo "uint8_t* gt = ${case_name}_gt;" >> ${GBIN4BM_SRC}
    done
fi

echo "#endif //_GBIN_4_BM_H_" >> ${GBIN4BM_HEADER}
