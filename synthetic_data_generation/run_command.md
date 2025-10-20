# Generate the queries based on the documents from the datastore of BRIGHT

python -m doc_to_query --model_id $MODEL --queries_per_doc $queries_per_doc --num_docs $num_docs --subject $TASK  --output_dir $output_dir --filter fineweb --prompt_id $prompt_id

-> Use this

python -m doc_to_query --model_id gpt-4o --queries_per_doc 3 --num_docs 1 --subject "biology" --output_dir ./outputs/doc_to_query/ --filter fineweb --prompt_id "hq_gen"

python -m doc_to_query --model_id gpt-4o --queries_per_doc 3 --num_docs 1 --subject "biology" --output_dir ./outputs/doc_to_query/ --filter fineweb --prompt_id "baseline"

# Generate the rewritten queries with reasoning given the queries

python -m generate_reasoning --model_id $MODEL  --num_docs $num_docs   --subject $TASK  --base_dir $output_dir --prompt_id $prompt_id



# Batch version

python -m doc_to_query_batch --model_id $MODEL --queries_per_doc $queries_per_doc --num_docs $num_docs  --subject $TASK  --output_dir $output_dir --filter fineweb --prompt_id $prompt_id

python -m doc_to_query_batch --model_id $MODEL --queries_per_doc $queries_per_doc --num_docs $num_docs  --subject $TASK  --output_dir $output_dir --filter fineweb --prompt_id $prompt_id --gather_results

python -m generate_reasoning_batch --model_id $MODEL --num_docs $num_docs --subject $TASK --base_dir $output_dir --prompt_id $prompt_id

python -m generate_reasoning_batch --model_id $MODEL --num_docs $num_docs --subject $TASK --base_dir $output_dir --prompt_id $prompt_id --gather_results
