// This file is CONFIDENTIAL and created by Arm Technology (China) Co., Ltd.
// See the copyright file distributed with this work for additional information
// regarding copyright ownership.
#ifndef _AIPU_DRIVER_WRAPPER_H_
#define _AIPU_DRIVER_WRAPPER_H_

#define AIPU_BASE_ADDRESS 0x1280010000ul
#define AIPU_HOST_OFFSET 0

typedef enum { AIPU_RUN_ERROR = 0x0, AIPU_RUN_RESULT_FAIL, AIPU_RUN_RESULT_PASS } aipu_run_result_t;
typedef enum { NOT_BATCH_OUTPUT = 0x0, BATCH_OUTPUT } graph_output_type_t;

struct graph_run_info {
  char name[64];
  void* graph_addr;
  void* input0_addr;
  void* input1_addr;
  void* output_addr;
  int run_times;
  graph_output_type_t output_type;
};

struct check_tensor_desc {
  unsigned int id;
  unsigned int offset;
  unsigned int size;
};

aipu_run_result_t aipu_start_single_graph(struct graph_run_info* run_graph);
aipu_run_result_t aipu_start_multi_graph(struct graph_run_info* run_graph, int graph_num);
aipu_run_result_t aipu_start_pipeline(struct graph_run_info* run_graph);

#endif
