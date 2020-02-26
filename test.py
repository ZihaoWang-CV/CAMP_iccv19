from vocab import Vocabulary
import evaluation
import pickle

evaluation.evalrank("",
                     data_path="./data", split="test", 
                     fold5=True)

"""print (rt,rti)
print(len(rt),len(rti))
dic_now = {}
dic_now["rt_ranks"]=rt[0]
dic_now["rt_top1"]=rt[1]
dic_now["rti_ranks"] =rti[0]
dic_now["rti_top1"]=rti[1]

with open('vsepp' + '.results.pickle', 'wb') as handle:
    pickle.dump(dic_now, handle, protocol=pickle.HIGHEST_PROTOCOL)"""
