from ensembleclassifier import *
import csv

base_acc = []
ense_acc = []

for i in range(8):
	esmb = EnsembleClassifier(8,(i+1))
	esmb.shuf_fit()
	base_acc.append(esmb.base_acc())
	ense_acc.append(esmb.ensm_acc())
	print("done with %1d out of 8 classifiers" %(i+1))

with open('./csvs/nonboosting_data8.csv', mode='w') as csv_file:
	fieldnames = ['ensemble size', 'depth', 'base accuracy', 'ensemble accuracy']
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

	writer.writeheader()
	for i in range(8):
		writer.writerow({'ensemble size': 8 , 'depth': i+1 , 'base accuracy':base_acc[i] , 'ensemble accuracy': ense_acc[i]})