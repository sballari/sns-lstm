#!/bin/bash

experiments={'eth','hotel','univ','zara1','zara2'}

<<<<<<< HEAD
for experiment in {'eth','hotel','univ','zara1','zara2',};
=======
for experiment in {'hotel','univ','zara1','zara2',};
>>>>>>> fc1fde1edd5fe01d033bc9461d5e2853b07dafa4
do
    for i in {1,2,3};
    do
        echo 'Training for' $experiment
        echo "Run number " $i
<<<<<<< HEAD
        python3 vlstm_train.py --experiment "../yaml_quan/"$experiment"_quan.yaml" --num_epochs 1
        python3 test.py --experiment "../yaml_quan/"$experiment"_quan.yaml"
=======
        python3 vlstm_train.py --experiment "../yaml_quan/"$experiment"_qP.yaml"
        python3 test.py --experiment "../yaml_quan/"$experiment"_qP.yaml"
>>>>>>> fc1fde1edd5fe01d033bc9461d5e2853b07dafa4
    done
done
