#!/bin/bash

experiments={'eth','hotel','univ','zara1','zara2'}

for experiment in {'eth','hotel','univ','zara1','zara2'};
do
    for i in {1..5};
    do
        echo 'Training for' $experiment
        echo "Run number " $i
        python3 vlstm_train.py --experiment "../yaml_quan/"$experiment"_quan.yaml"
        python3 test.py --experiment "../yaml_quan/"$experiment"_quan.yaml"
    done
done