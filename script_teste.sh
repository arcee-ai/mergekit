
dir='./merged'

mergekit-evolve --batch-size 1 \
                --no-in-memory \
                --allow-crimes \
                --no-reshard \
                --strategy pool \
                --opt_method CMA-ES \
                --random-seed 0 \
                --storage-path $dir \
                --force-population-size 2\
                --max-fevals 5 \
                ./examples/evo_bert_large.yml

rm -rf $dir"/transformers_cache/"

