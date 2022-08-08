import numpy as np
import random


topk = np.load('./topk/dev-rationale_metadata.npy', allow_pickle=True).item(0)
conti = np.load('./contigious/dev-rationale_metadata.npy', allow_pickle=True).item(0)

id_list = list(topk.keys())
random_id = random.choice(id_list)

print('====================fixed rationale length==================')
print(topk[random_id]['var-len_var-feat_var-type']['fixed rationale length'])
print('------------')
print(conti[random_id]['var-len_var-feat_var-type']['fixed rationale length'])
print('======================================')

print('====================variable rationale length==================')
print(topk[random_id]['var-len_var-feat_var-type']['variable rationale length'])
print('------------')
print(conti[random_id]['var-len_var-feat_var-type']['variable rationale length'])
print('======================================')
# [random_id]
# 'variable rationale length': 22, 
# 'fixed rationale length': 32, 
# 'variable rationale ratio': 0.06962025316455696, 
# 'variable rationale mask': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
# 'fixed rationale mask': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
#    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#'fixed-length divergence': 8.181097655324265e-05, 
# 'variable-length divergence': 0.00031813146779313684, 
# 'running predictions': array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#   2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), 
# 'time elapsed': 1.593606948852539, 
# 'feature attribution name': 'scaled attention'


print('======================================')
print(topk[random_id]['var-len_var-feat_var-type'])
print('------------')
print(conti[random_id]['var-len_var-feat_var-type'])
print('======================================')