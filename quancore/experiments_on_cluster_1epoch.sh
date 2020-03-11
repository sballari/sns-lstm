#!/bin/bash

declare -a experiments=('eth','hotel','univ','zara1','zara2')

for experiment in 'eth' 'hotel' 'univ' 'zara1' 'zara2';
do
    for i in '1' '2' '3' '4' '5';
    do
        echo 'Training for' $experiment
        echo 'experiment number: '$i
        python3 vlstm_train.py --experiment "../yaml_quan/"$experiment"_quan.yaml"
        python3 test.py --experiment "../yaml_quan/"$experiment"_quan.yaml"
        echo '==========================================='
    done
done
