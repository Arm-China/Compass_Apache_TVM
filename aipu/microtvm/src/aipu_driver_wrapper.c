// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2023 Arm Technology (China) Co. Ltd.
#include "aipu_driver_wrapper.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "aipu_api.h"

static int load_inputs(aipu_buffer_desc_t* buffer_desc, void* input) {
  int ret = -1;

  if ((NULL == buffer_desc) || (NULL == input)) {
    return ret;
  }
  port_memcpy(buffer_desc->addr, input, buffer_desc->size);

  return 0;
}

static int get_output(aipu_graph_desc_t graph_desc, aipu_tensor_desc_t* output_tensor,
                      void* output) {
  unsigned int i = 0;
  int ret = 0;
  int compare_fail_flag = 0;
  unsigned int output_cnt = 0;
  unsigned int tensor_iter = 0;
  unsigned int bytes = 0;
  const unsigned int max_printf = 256;

  aipu_buffer_desc_t* tmp_tensor = NULL;

  for (output_cnt = 0; output_cnt < graph_desc.output_num; output_cnt++) {
    for (tensor_iter = 0; tensor_iter < graph_desc.output_num; tensor_iter++) {
      if (output_tensor->buffer_desc[tensor_iter].index == output_cnt) break;
    }
    if (tensor_iter == graph_desc.output_num) {
      ret = 1;
      TVMLogf("error: can't find the corresponding tensor\n");
      return ret;
    }

    tmp_tensor = &output_tensor->buffer_desc[tensor_iter];
    bytes = tmp_tensor->size > max_printf ? max_printf : tmp_tensor->size;

    TVMLogf("start copy, size is %d, output is %p, tensor is %p\n", tmp_tensor->size, output,
            tmp_tensor->addr);
    port_memcpy(output, tmp_tensor->addr, tmp_tensor->size);
    TVMLogf("finish output copy\n");
  }

  return 0;
}

aipu_run_result_t aipu_start_single_graph(struct graph_run_info* run_graph) {
  int start_run_flag = 0;
  int compare_fail_flag = 0;
  int iter = 0, i = 0;

  aipu_run_result_t aipu_run_result = AIPU_RUN_ERROR;
  aipu_task_status_t aipu_status = AIPU_TASK_STATUS_NO_STATUS;
  aipu_graph_desc_t graph_desc = {0};
  aipu_tensor_desc_t input_tensor_desc = {0};
  aipu_tensor_desc_t output_tensor_desc = {0};
  aipu_tensor_desc_t dump_tensor_desc = {0};
  aipu_tensor_desc_t profile_tensor_desc = {0};
  aipu_status_t ret = AIPU_STATUS_SUCCESS;

  aipu_config_address(AIPU_BASE_ADDRESS, AIPU_HOST_OFFSET);

  ret = aipu_init_ctx();
  if (AIPU_STATUS_SUCCESS != ret) {
    TVMLogf("aipu context init fail\n");
    goto error1;
  }

  ret = aipu_load_graph(run_graph->graph_addr, &graph_desc);
  if (AIPU_STATUS_SUCCESS != ret) {
    TVMLogf("aipu load graph fail\n");
    goto error2;
  }

  if (0 != graph_desc.input_num) {
    input_tensor_desc.buffer_desc =
        (aipu_buffer_desc_t*)malloc(graph_desc.input_num * sizeof(aipu_buffer_desc_t));
    if (NULL == input_tensor_desc.buffer_desc) {
      TVMLogf("usr malloc input tensor fail\n");
      goto error3;
    }
  }

  if (0 != graph_desc.output_num) {
    output_tensor_desc.buffer_desc =
        (aipu_buffer_desc_t*)malloc(graph_desc.output_num * sizeof(aipu_buffer_desc_t));
    if (NULL == output_tensor_desc.buffer_desc) {
      TVMLogf("usr malloc output tensor fail\n");
      goto error3;
    }
  }

  if (0 != graph_desc.dump_num) {
    dump_tensor_desc.buffer_desc =
        (aipu_buffer_desc_t*)malloc(graph_desc.dump_num * sizeof(aipu_buffer_desc_t));
    if (NULL == dump_tensor_desc.buffer_desc) {
      TVMLogf("usr malloc dump tensor fail\n");
      goto error3;
    }
  }

  if (0 != graph_desc.profile_num) {
    profile_tensor_desc.buffer_desc =
        (aipu_buffer_desc_t*)malloc(graph_desc.profile_num * sizeof(aipu_buffer_desc_t));
    if (NULL == profile_tensor_desc.buffer_desc) {
      TVMLogf("usr malloc profile tensor fail\n");
      goto error3;
    }
  }

  ret = aipu_alloc_tensor_buffer(graph_desc.graph_id, 1);
  if (AIPU_STATUS_SUCCESS != ret) {
    TVMLogf("aipu alloc tensor buffer fail\n");
    goto error3;
  }

  ret = aipu_get_tensor_desc(graph_desc.graph_id, 0, TENSOR_IO_TYPE_IN, &input_tensor_desc);
  if (AIPU_STATUS_SUCCESS != ret) {
    TVMLogf("aipu get tensor desc fail\n");
    goto error4;
  }

  ret = aipu_get_tensor_desc(graph_desc.graph_id, 0, TENSOR_IO_TYPE_OUT, &output_tensor_desc);
  if (AIPU_STATUS_SUCCESS != ret) {
    TVMLogf("aipu get tensor desc fail\n");
    goto error4;
  }

  if (0 != graph_desc.dump_num) {
    ret = aipu_get_tensor_desc(graph_desc.graph_id, 0, TENSOR_IO_TYPE_DUMP, &dump_tensor_desc);
    if (AIPU_STATUS_SUCCESS != ret) {
      TVMLogf("aipu get tensor desc fail\n");
      goto error4;
    }
  }

  if (0 != graph_desc.profile_num) {
    ret =
        aipu_get_tensor_desc(graph_desc.graph_id, 0, TENSOR_IO_TYPE_PROFILE, &profile_tensor_desc);
    if (AIPU_STATUS_SUCCESS != ret) {
      TVMLogf("aipu get tensor desc fail\n");
      goto error4;
    }
  }

  for (iter = 0; iter < run_graph->run_times; iter++) {
    if (1 == graph_desc.input_num) {
      TVMLogf("load input0\n");
      load_inputs(&input_tensor_desc.buffer_desc[0], run_graph->input0_addr);
    } else if (2 == graph_desc.input_num) {
      TVMLogf("load input0/1\n");
      load_inputs(&input_tensor_desc.buffer_desc[0], run_graph->input0_addr);
      load_inputs(&input_tensor_desc.buffer_desc[1], run_graph->input1_addr);
    } else {
      TVMLogf("error :too much inputs\n");
      goto error4;
    }

    ret = aipu_start(graph_desc.graph_id, 0);
    if (!((AIPU_STATUS_SUCCESS == ret) || (AIPU_STATUS_ERROR_RUN_SUCCESS_BUT_WARNING == ret))) {
      TVMLogf("aipu start fail\n");
      aipu_get_status(graph_desc.graph_id, &aipu_status);
      goto error4;
    }

    start_run_flag = 1;
    while (1) {
      aipu_get_status(graph_desc.graph_id, &aipu_status);
      if (AIPU_TASK_STATUS_RUNNING == aipu_status) {
        if (1 == start_run_flag) {
          start_run_flag = 0;
          TVMLogf(
              "\n-->graph [%s] is running on aipu, and its graph id is [%d], please "
              "waiting.......\n",
              run_graph->name, graph_desc.graph_id);
        }
      } else {
        break;
      }

      sleep(2);
    }

    if (AIPU_TASK_STATUS_DONE == aipu_status) {
      TVMLogf(
          "\n-->graph [%s] on aipu running done, graph id is [%d], and starting to check its "
          "result:\n",
          run_graph->name, graph_desc.graph_id);

      if (run_graph->output_type == NOT_BATCH_OUTPUT) {
        compare_fail_flag = get_output(graph_desc, &output_tensor_desc, run_graph->output_addr);
      }

      if (1 == compare_fail_flag) {
        aipu_run_result = AIPU_RUN_RESULT_FAIL;
        goto error4;
      } else {
        aipu_run_result = AIPU_RUN_RESULT_PASS;
      }

    } else if (AIPU_TASK_STATUS_EXCEPTION == aipu_status) {
      aipu_run_result = AIPU_RUN_ERROR;
      TVMLogf("aipu running exception\n");
      goto error4;
    } else {
      aipu_run_result = AIPU_RUN_ERROR;
      TVMLogf("graph id no exist or not starting to run\n");
      goto error4;
    }
  }

error4:
  aipu_free_tensor_buffer(graph_desc.graph_id);

error3:
  aipu_unload_graph(graph_desc.graph_id);
  if (NULL != input_tensor_desc.buffer_desc) {
    free(input_tensor_desc.buffer_desc);
    input_tensor_desc.buffer_desc = NULL;
  }
  if (NULL != output_tensor_desc.buffer_desc) {
    free(output_tensor_desc.buffer_desc);
    output_tensor_desc.buffer_desc = NULL;
  }
  if (NULL != dump_tensor_desc.buffer_desc) {
    free(dump_tensor_desc.buffer_desc);
    dump_tensor_desc.buffer_desc = NULL;
  }
  if (NULL != profile_tensor_desc.buffer_desc) {
    free(profile_tensor_desc.buffer_desc);
    profile_tensor_desc.buffer_desc = NULL;
  }

error2:
  aipu_deinit_ctx();

error1:
  if ((AIPU_STATUS_SUCCESS != ret) && (AIPU_STATUS_ERROR_RUN_SUCCESS_BUT_WARNING != ret)) {
    TVMLogf("TOP API error code:0x%x\n", ret);
  }

  return aipu_run_result;
}
