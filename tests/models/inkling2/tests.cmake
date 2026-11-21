# The Inkling 2 tokenizer against Hugging Face `tokenizers` ids on the real tokenizer.json: skips without
# INFERNIX_TEST_INKLING_TOKENIZER_DIR (generate_tokenizer_golden.py writes the directory from a checkpoint or artifact).
infernix_add_test(infernix_inkling2_tokenizer_golden_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tokenizer_golden_real.cpp"
  LIBRARIES infernix_text infernix::json
  NEEDS_SOURCE_DIR)
set_tests_properties(infernix_inkling2_tokenizer_golden_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The output session against the response_template's reference parser (transformers) on the real tokenizer (CPU only):
# skips without INFERNIX_TEST_INKLING_TOKENIZER_DIR.
infernix_add_test(infernix_inkling2_output_session_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_output_session_real.cpp"
  LIBRARIES infernix_model_runtime infernix_text infernix::json
  NEEDS_SOURCE_DIR)
set_tests_properties(infernix_inkling2_output_session_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The prompt's ids (template + literal spans) against Hugging Face's, and client text that spells markers (CPU only):
# skips without INFERNIX_TEST_INKLING_TOKENIZER_DIR.
infernix_add_test(infernix_inkling2_prompt_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prompt_real.cpp"
  LIBRARIES infernix_jinja infernix_text infernix::json
  NEEDS_SOURCE_DIR)
set_tests_properties(infernix_inkling2_prompt_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The frontend: requests against Hugging Face's ids, the echo of generated turns, refusals (CPU only): skips without
# INFERNIX_TEST_INKLING_TOKENIZER_DIR.
infernix_add_test(infernix_inkling2_frontend_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend_real.cpp"
  LIBRARIES infernix_model_runtime infernix_text infernix::json
  NEEDS_SOURCE_DIR)
set_tests_properties(infernix_inkling2_frontend_real_test PROPERTIES SKIP_RETURN_CODE 77)

# Through the public Engine (the Program's first slice), against an eager one-request Engine: the golden greedy
# continuation, graphs, request invariance at C = 4 and 8, prefill chunk width, and a scoring Engine's logits against
# the forward test's (INFERNIX_INKLING2_LOGITS_GUARD). Skips without INFERNIX_INKLING2_ARTIFACT and
# INFERNIX_INKLING2_REFERENCE.
infernix_add_test(infernix_inkling2_engine_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_real.cpp"
  LIBRARIES infernix_engine infernix::json)
set_tests_properties(infernix_inkling2_engine_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE
  LABELS "gpu;real" TIMEOUT 1800)

# The forward on a real artifact against the CPU reference's per-visit dump (teacher-forced stages, free-running
# argmax, decode = verify): skips without INFERNIX_INKLING2_ARTIFACT and INFERNIX_INKLING2_REFERENCE.
infernix_add_test(infernix_inkling2_forward_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_forward_real.cpp"
  LIBRARIES infernix_model_runtime infernix::json)
set_tests_properties(infernix_inkling2_forward_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE
  LABELS "gpu;real" TIMEOUT 1800)

# The official chat template through Infernix's Jinja engine against transformers' rendering (CPU only).
infernix_add_test(infernix_inkling2_chat_template_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_chat_template.cpp"
  LIBRARIES infernix_jinja infernix::json
  NEEDS_SOURCE_DIR)

# The loader's plan on a real artifact (CPU only): skips without INFERNIX_INKLING2_ARTIFACT.
infernix_add_test(infernix_inkling2_load_plan_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_load_plan_real.cpp"
  LIBRARIES infernix_model_loading)
set_tests_properties(infernix_inkling2_load_plan_real_test PROPERTIES SKIP_RETURN_CODE 77)
