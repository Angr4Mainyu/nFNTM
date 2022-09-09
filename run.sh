version='test'
dataset='reuters'
for model in 'propose'
do
    for dataset in 'reuters'
    do
        cd source
        echo "$model $dataset $version"

        
        mkdir -p ../log/${model}/${dataset}_${version}
        python $model.py $dataset Train $beta > ../log/${model}/${dataset}_${version}/Train.log
        echo "-Finished training!"
        python $model.py $dataset Test $beta > ../log/${model}/${dataset}_${version}/Test.log
        echo "-Finished testing!"
    done
done