#!/bin/bash

declare -a experiments=('eth','hotel','univ','zara1','zara2')

for experiment in {'eth',};
do
    for i in {'1',};
    do
        echo 'Training for' $experiment
        echo 'experiment number: '$i
        python3 train.py --experiment "../yaml_quan/"$experiment"_qP.yaml"
        python3 test.py --experiment "../yaml_quan/"$experiment"_qP.yaml"
        echo '==========================================='
    done
done
