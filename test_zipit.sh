python mergekit/scripts/dump_out_features.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 -o ./dump_output  -d arcee-ai/pmc-test-perplexity  -s 6  -c text  -u test  --device cpu
python mergekit/scripts/dump_out_features.py TinyLlama/TinyLlama-1.1B-Chat-v0.6 -o ./dump_output  -d arcee-ai/pmc-test-perplexity  -s 6  -c text  -u test  --device cpu
python mergekit/scripts/dump_out_m_and_u.py ./dump_output/TinyLlama_TinyLlama-1.1B-Chat-v0.6_features.bin ./dump_output/TinyLlama_TinyLlama-1.1B-Chat-v1.0_features.bin --model_path TinyLlama/TinyLlama-1.1B-Chat-v0.6 --out_path ./m_v_out
python mergekit/scripts/zipit_fast_prototype.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 TinyLlama/TinyLlama-1.1B-Chat-v0.6  m_v_out -o new_model
python test_by_gen.py new_model
