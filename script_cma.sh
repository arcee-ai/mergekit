
t='./experimentos_v1/cmaes_merged/merge'
for i in $(seq 1 5);
do
    dir=$t"_"$i
    mkdir $dir
    echo $dir

    mergekit-evolve --batch-size 1 \
                    --no-in-memory \
                    --allow-crimes \
                    --no-reshard \
                    --strategy pool \
                    --opt_method CMA-ES \
                    --random-seed $i \
                    --storage-path $dir \
                    --force-population-size 10\
                    --max-fevals 500 \
                    ./examples/evo_bert.yml

    rm -rf $dir"/transformers_cache/"
done
