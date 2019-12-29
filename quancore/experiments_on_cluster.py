import argparse 
import vlstm_train
import os




for ex in ['zara1','zara2','univ','eth','hotel']:
	for i in range(3):
		print ("experiment "+ex+" run "+str(i))
		#os.system("vlstm_train.py "+"--experiment ../yaml_quan/"+ex)
		parser = argparse.ArgumentParser()
		args = parser.parse_args(args=['--experiment','../yaml_quan/'+ex+'_quan.yaml'])
		vlstm_train.main(args)
print ("fine")
		
