'''
top_attention_suff_score = TOPk_scores.get('test_990').get('attention').get('sufficiency aopc').get('mean') # evinf @ 0.1


{'full text prediction': array([0.00194535, 0.99805462]), 
'true label': 1, 
'random': 
    {'sufficiency': 0.30214041956014076, 
    'comprehensiveness': 0.8373840546561564, 
    'masked R probs (comp)': array([0.73946786, 0.26053208]), 
    'only R probs (suff)': 0.38334789872169495, 
    'sufficiency aopc': 
        {'mean': array([0.]), 'per ratio': array([0.])}, 
    'comprehensiveness aopc': {
        'mean': array([0.]), 'per ratio': array([0.])}},

'deeplift': 
    {'sufficiency': 0.0, 
    'comprehensiveness': 1.0, 
    'masked R probs (comp)': array([0.93155044, 0.0684496 ]), 
    'only R probs (suff)': 0.09491266310214996, 
    'sufficiency aopc': {'mean': array([0.]), 'per ratio': array([0.])}, 
    'comprehensiveness aopc': {'mean': array([0.]), 'per ratio': array([0.])}}, 

'gradients': {'sufficiency': 0.0, 'comprehensiveness': 0.7674069269118021, 'masked R probs (comp)': array([0.67814285, 0.32185718]), 'only R probs (suff)': 0.017005568370223045, 'sufficiency aopc': {'mean': array([0.]), 'per ratio': array([0.])}, 'comprehensiveness aopc': {'mean': array([0.]), 'per ratio': array([0.])}}, 
'ig': {'sufficiency': 0.1652852871702561, 'comprehensiveness': 0.8873247520198834, 'masked R probs (comp)': array([0.78398544, 0.21601462]), 'only R probs (suff)': 0.2622987926006317, 'sufficiency aopc': {'mean': array([0.]), 'per ratio': array([0.])}, 'comprehensiveness aopc': {'mean': array([0.]), 'per ratio': array([0.])}}, 
'scaled attention': {'sufficiency': 0.0, 'comprehensiveness': 1.0, 'masked R probs (comp)': array([0.95105702, 0.04894301]), 'only R probs (suff)': 0.07845953106880188, 'sufficiency aopc': {'mean': array([0.]), 'per ratio': array([0.])}, 'comprehensiveness aopc': {'mean': array([0.]), 'per ratio': array([0.])}}, 
'attention': {'sufficiency': 0.07215457332409868, 'comprehensiveness': 1.0, 'masked R probs (comp)': array([0.93936759, 0.06063243]), 'only R probs (suff)': 0.17983797192573547, 'sufficiency aopc': {'mean': array([0.]), 'per ratio': array([0.])}, 'comprehensiveness aopc': {'mean': array([0.]), 'per ratio': array([0.])}}, 
'gradientshap': {'sufficiency': 0.0, 'comprehensiveness': 0.7062577987085175, 'masked R probs (comp)': array([0.62482655, 0.37517348]), 'only R probs (suff)': 0.035634610801935196, 'sufficiency aopc': {'mean': array([0.]), 'per ratio': array([0.])}, 'comprehensiveness aopc': {'mean': array([0.]), 'per ratio': array([0.])}}}
'''
    
from crypt import METHOD_SHA512
from telnetlib import PRAGMA_HEARTBEAT
import numpy as np
import pandas as pd
import logging
import os
import argparse
# libraries
import matplotlib.pyplot as plt
import numpy as np
import torch

# # create data
# values=np.cumsum(np.random.randn(1000,1))

# # use the plot function
# plt.plot(values)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type = str,
    help = "select dataset / task",
    default = "sst",
)


user_args = vars(parser.parse_args())
# user_args["importance_metric"] = None

dataset = str(user_args["dataset"])



# SET 0 = TOP1, TOP2, TOP3, TOP4 ---> original
# SET 1 = TOP1, TOP2, TOP3, Rand  
# SET 2 = TOP1, TOP2, Rand, Rand 
# SET 3 = TOP1, Rand, Rand, Rand 
# SET 4 = Rand, Rand, Rand, Rand


# importance_scores = np.load('./extracted_rationales/SST/importance_scores/test_importance_scores-10.npy', allow_pickle=True).item()
# print(importance_scores.get('test_358').get('attention')) # dict_keys(['random', 'attention', 'gradients', 'ig', 'scaled attention', 'deeplift', 'gradientshap'])

# x = np.argsort(importance_scores.get('test_358').get('attention'))[::-1][:4]
# print("Indices:",x)

df = pd.read_csv(f'datasets/{dataset}/data/test.csv')[100:].sample(5)
data_id_list = df['annotation_id']
print(df)
importance_scores = np.load(f'datasets/{dataset}/data/importance_scores/test_importance_scores_25.npy', allow_pickle=True).item()

text_S1 = []
text_S2 = []
text_S3 = []
text_S4 = []

for i, data_id in enumerate(data_id_list):
    print(df[i].text)
    attention_scores = importance_scores.get(data_id).get('attention')
    top4 = torch.topk(attention_scores, 4)[1] # 0 for value 1 for index
    print(top4)

    print(attention_scores)
    quit()
  


S0_suff = np.load('./posthoc_results/SST/ZEROOUT-faithfulness-scores-detailed.npy', 
                        allow_pickle=True).item()

S1_suff = np.load('./posthoc_results/SST/ZEROOUT-faithfulness-scores-detailed-S1.npy', 
                        allow_pickle=True).item()

ids = list(S1_suff.keys())

S2_importance_scores = np.load('./posthoc_results/SST/ZEROOUT-faithfulness-scores-detailed-S2.npy', 
                        allow_pickle=True).item()

S3_importance_scores = np.load('./posthoc_results/SST/ZEROOUT-faithfulness-scores-detailed-S3.npy', 
                        allow_pickle=True).item()

S4_importance_scores = np.load('./posthoc_results/SST/ZEROOUT-faithfulness-scores-detailed-S4.npy', 
                        allow_pickle=True).item()



def F_i(M_SO, M_S4, M_Si): # M is the metrics score 
    F_i = abs(M_SO-M_Si)/abs(M_SO-M_S4+0.00001)
    return F_i

S0_sufficiencies = []
S1_sufficiencies = []
S2_sufficiencies = []
S3_sufficiencies = []
S4_sufficiencies = []

for id in ids:
    M_S0 = S0_suff.get(id).get('attention').get('sufficiency')
    M_S1 = S1_suff.get(id).get('attention').get('sufficiency')
    M_S2 = S2_importance_scores.get(id).get('attention').get('sufficiency')
    M_S3 = S3_importance_scores.get(id).get('attention').get('sufficiency')
    M_S4 = S4_importance_scores.get(id).get('attention').get('sufficiency')
    
    S0 = F_i(M_S0, M_S4, M_S0)
    S1 = F_i(M_S0, M_S4, M_S1)
    S2 = F_i(M_S0, M_S4, M_S2)
    S3 = F_i(M_S0, M_S4, M_S3)
    S4 = F_i(M_S0, M_S4, M_S4)

    S0_sufficiencies.append(S0)
    S1_sufficiencies.append(S1)
    S2_sufficiencies.append(S2)
    S3_sufficiencies.append(S3)
    S4_sufficiencies.append(S4)


df = pd.DataFrame(list(zip(ids, S0_sufficiencies, S1_sufficiencies, S2_sufficiencies, S3_sufficiencies, S4_sufficiencies)),
               columns =['id', '0', '1', '2', '3', '4'])

df.to_csv('SST_attention_soft_sufficiency_interpolation.csv')




quit()
'''[      -inf 0.07614467 0.03361953 0.01381731 0.01694824 0.01160708
 0.01682331 0.00476329 0.00979104 0.00561564 0.00487841 0.00499885
 0.00257135 0.00282017 0.00283638 0.00786511 0.06252006 0.03839585
 0.01980747 0.01921392 0.03641286 0.02184515 0.02980015 0.02374282
 0.0199705  0.02761515 0.0133762  0.02211065 0.01256411 0.01168522
 0.01450356 0.36185393       -inf       -inf       -inf       -inf
       -inf       -inf       -inf       -inf       -inf       -inf
       -inf       -inf       -inf       -inf       -inf       -inf]'''




quit()


##########  GET M_SO, M_S1, M_S2, , M_S3, M_S4, for the 50 

############# save to a new diction 


# def presentation_of_SET_0(input_id, important_score): # i from 0 to 4 
#     presentation_at_i = input_id_changed

    

# def get_M_score_of_SETi(presentation):





pwd = os.getcwd()


NOISE_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'NOISE-faithfulness-scores-detailed-std_' + str(user_args["std"]) + '.npy') 
topk_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'topk-faithfulness-scores-detailed.npy')
ATTENTION_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ATTENTION-faithfulness-scores-detailed.npy')
ZEROOUT_scores_file = os.path.join(pwd, 'posthoc_results', str(dataset), 'ZEROOUT-faithfulness-scores-detailed.npy')


TOPk_scores = np.load(topk_scores_file, allow_pickle=True).item()
ZEROOUT_scores = np.load(ZEROOUT_scores_file, allow_pickle=True).item()
ATTENTION_scores = np.load(ATTENTION_scores_file, allow_pickle=True).item()
NOISE_scores = np.load(NOISE_scores_file, allow_pickle=True).item()
#NOISE05_scores = np.load(NOISE05_scores_file, allow_pickle=True).item()

data_id_list = TOPk_scores.keys()
fea_list = ['attention', "scaled attention", "gradients", "ig", "gradientshap", "deeplift"]
FA = 'attention'


D_TOP_Suff = []
D_ATTENTION_Suff = []
D_ZEROOUT_Suff = []
D_NOISE_Suff = []


for FA in fea_list:
    Diag_TOP_attention = 0
    Diag_ATTENTION_attention = 0
    Diag_ZEROOUT_attention = 0
    Diag_NOISE_attention = 0
    
    for i, data_id in enumerate(data_id_list):

        top_random_suff_score = TOPk_scores.get(data_id).get('random').get('sufficiency aopc').get('mean')
        NOISE_random_suff_score = NOISE_scores_file.get(data_id).get('random').get('sufficiency')#.get('mean')
        ZEROOUT_random_suff_score = ZEROOUT_scores.get(data_id).get('random').get('sufficiency')#.get('mean')
        ATTENTION_random_suff_score = ATTENTION_scores.get(data_id).get('random').get('sufficiency')#.get('mean')

        top_suff_score = TOPk_scores.get(data_id).get(FA).get('sufficiency aopc').get('mean') # evinf @ 0.1
        if top_suff_score >= top_random_suff_score: Diag_TOP_attention += 1
        else: pass

        NOISE_suff_score = NOISE_scores.get(data_id).get(FA).get('sufficiency')
        if NOISE_suff_score >= NOISE_random_suff_score: Diag_NOISE_attention += 1
        else: pass

        ZEROOUT_suff_score = ZEROOUT_scores.get(data_id).get(FA).get('sufficiency')
        if ZEROOUT_suff_score >= ZEROOUT_random_suff_score: Diag_ZEROOUT_attention += 1
        else: pass

        ATTENTION_suff_score = ATTENTION_scores.get(data_id).get(FA).get('sufficiency')
        if ATTENTION_suff_score >= ATTENTION_random_suff_score: Diag_ATTENTION_attention += 1
        else: pass

        
    D_TOP = Diag_TOP_attention/len(data_id_list)
    D_TOP_Suff.append(D_TOP)

    D_ATTENTION = Diag_ATTENTION_attention/len(data_id_list)
    D_ATTENTION_Suff.append(D_ATTENTION)

    D_ZEROOUT = Diag_ZEROOUT_attention/len(data_id_list)
    D_ZEROOUT_Suff.append(D_ZEROOUT)

    D_NOISE1= Diag_NOISE_attention/len(data_id_list)
    D_NOISE_Suff.append(D_NOISE1)


df = pd.DataFrame(list(zip(fea_list, D_TOP_Suff, D_ATTENTION_Suff, D_ZEROOUT_Suff, D_NOISE_Suff)),
               columns =['Feature', 'TopK', 'Soft(ATTENTION)', 'Soft(ZEROOUT)', 'Soft(NOISE1)'])

fname = os.path.join(pwd, 'Diagnosticity', str(dataset), 'Diagnosticity_Suff.csv')
os.makedirs(os.path.join(pwd, 'Diagnosticity', str(dataset)), exist_ok=True)
print(df)
df.to_csv(fname)





################ comp

D_TOP_Suff = []
D_ATTENTION_Suff = []
D_ZEROOUT_Suff = []
D_NOISE_Suff = []

for FA in fea_list:
    Diag_TOP_attention = 0
    Diag_ATTENTION_attention = 0
    Diag_ZEROOUT_attention = 0
    Diag_NOISE_attention = 0
    
    for i, data_id in enumerate(data_id_list):

        top_random_suff_score = TOPk_scores.get(data_id).get('random').get('comprehensiveness aopc').get('mean')
        NOISE_random_suff_score = NOISE_scores.get(data_id).get('random').get('comprehensiveness')#.get('mean')
        ZEROOUT_random_suff_score = ZEROOUT_scores.get(data_id).get('random').get('comprehensiveness')#.get('mean')
        ATTENTION_random_suff_score = ATTENTION_scores.get(data_id).get('random').get('comprehensiveness')#.get('mean')

        top_suff_score = TOPk_scores.get(data_id).get(FA).get('comprehensiveness aopc').get('mean') # evinf @ 0.1
        if top_suff_score >= top_random_suff_score: Diag_TOP_attention += 1
        else: pass

        NOISE_suff_score = NOISE_scores.get(data_id).get(FA).get('comprehensiveness')
        if NOISE_suff_score >= NOISE_random_suff_score: Diag_NOISE_attention += 1
        else: pass

        ZEROOUT_suff_score = ZEROOUT_scores.get(data_id).get(FA).get('comprehensiveness')
        if ZEROOUT_suff_score >= ZEROOUT_random_suff_score: Diag_ZEROOUT_attention += 1
        else: pass

        ATTENTION_suff_score = ATTENTION_scores.get(data_id).get(FA).get('comprehensiveness')
        if ATTENTION_suff_score >= ATTENTION_random_suff_score: Diag_ATTENTION_attention += 1
        else: pass

    D_TOP = Diag_TOP_attention/len(data_id_list)
    D_TOP_Suff.append(D_TOP)

    D_ATTENTION = Diag_ATTENTION_attention/len(data_id_list)
    D_ATTENTION_Suff.append(D_ATTENTION)

    D_ZEROOUT = Diag_ZEROOUT_attention/len(data_id_list)
    D_ZEROOUT_Suff.append(D_ZEROOUT)

    D_NOISE= Diag_NOISE_attention/len(data_id_list)
    D_NOISE_Suff.append(D_NOISE1)



df = pd.DataFrame(list(zip(fea_list, D_TOP_Suff, D_ATTENTION_Suff, D_ZEROOUT_Suff, D_NOISE_Suff, )),
               columns =['Feature', 'TopK', 'Soft(ATTENTION)', 'Soft(ZEROOUT)', 'Soft(NOISE1)'])

fname = os.path.join(pwd, 'Diagnosticity', str(dataset), 'Diagnosticity_Comp.csv')
os.makedirs(os.path.join(pwd, 'Diagnosticity', str(dataset)), exist_ok=True)
df.to_csv(fname)



