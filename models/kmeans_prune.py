import torch
import numpy as np
from sklearn.cluster import KMeans
from models.resnet import resnet20

def prune(model, clusters_num):
    
    state_dict = model.state_dict()

    for (name, module), num_cl in zip(model.named_modules(), clusters_num):
        if isinstance(module, torch.nn.Conv2d):
            prms = list(module.parameters())
            filters = prms[0].view(prms[0].shape[0], -1).cpu().detach().numpy()
            clustering = KMeans(num_cl)
            clusters = clustering.fit_predict(filters)
            new_weights = np.array([clustering.cluster_centers_[cl] for cl in clusters])
            new_weights = torch.Tensor(new_weights).view(-1,*prms[0].shape[1:])
            state_dict[name+'.weight'] = new_weights
            
    pruned_model = resnet20().load_state_dict(state_dict)
    return pruned_model