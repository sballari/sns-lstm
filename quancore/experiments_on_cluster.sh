


#!/bin/bash

experiments={'eth','hotel','univ','zara1','zara2'}

for experiment in 'hotel' 'univ' 'zara1' 'zara2';
do
    for i in {'1','2',};
    do
        echo 'Training for' $experiment
        echo "Run number " $i
        python3 train.py --experiment "../yaml_quan/"$experiment"_qP.yaml"
        python3 test.py --experiment "../yaml_quan/"$experiment"_qP.yaml"
    done
done
