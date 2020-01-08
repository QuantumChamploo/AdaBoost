from ensembleclassifier import *



ens1 = EnsembleClassifier(7,3)

ens1.shuf_fit()



print(ens1.ensm_acc())

print(ens1.base_acc())

print(ens1.models[0].score(X_test,Y_test))