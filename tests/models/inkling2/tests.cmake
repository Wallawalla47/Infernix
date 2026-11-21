# The Inkling 2 tokenizer against Hugging Face `tokenizers` ids on the real tokenizer.json: skips without
# INFERNIX_TEST_INKLING_TOKENIZER_DIR (generate_tokenizer_golden.py writes the directory from a checkpoint or artifact).
infernix_add_test(infernix_inkling2_tokenizer_golden_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tokenizer_golden_real.cpp"
  LIBRARIES infernix_text infernix::json
  NEEDS_SOURCE_DIR)
set_tests_properties(infernix_inkling2_tokenizer_golden_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The loader's plan on a real artifact (CPU only): skips without INFERNIX_INKLING2_ARTIFACT.
infernix_add_test(infernix_inkling2_load_plan_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_load_plan_real.cpp"
  LIBRARIES infernix_model_loading)
set_tests_properties(infernix_inkling2_load_plan_real_test PROPERTIES SKIP_RETURN_CODE 77)
