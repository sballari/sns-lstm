#!/bin/bash

experiments={'eth','hotel','univ','zara1','zara2'}

for experiment in {'eth',};
do
    for i in {1,2,3};
    do
        echo 'Training for' $experiment
        echo "Run number " $i
        python3 vlstm_train.py --experiment "../yaml_quan/"$experiment"_qP.yaml"
        python3 test.py --experiment "../yaml_quan/"$experiment"_qP.yaml"
    done
done
