import torch
import numpy as np
from sklearn.cluster import KMeans
from models.resnet import resnet20
from models.pruned_resnet import pruned_resnet20

'''
Прунинг будем делать так:
  -- Получаем количество кластеров в на каждый сверточный слой
  -- Для каждого сверточного слоя меняем веса фильтра на центроиду к которой он относился
  -- Запоминаем в каждоом слое порядок в котором идут центроиды для последующей оптимизации сети
  

При оптимизации сети будем оставлять в слое количество выходных фильтров такое, как количесттво центроид в слое
После выпомнения такой свертки составляем входной тензор для следующего слоя, можем это сделать тк запомнили порядок центров кластеров

'''

def get_pruned_model(model, clusters_num):
    
    parameters = []
    i = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prms = list(module.parameters())

            filters = prms[0].view(prms[0].shape[0], -1).cpu().detach().numpy()
            clustering = KMeans(clusters_num[i])
            
            clusters = clustering.fit_predict(filters)
            
            centroids = torch.Tensor(clustering.cluster_centers_).view(-1,*prms[0].shape[1:])

            parameters.append((prms[0].shape[1], centroids, clusters))
            i+=1
            
    resnet20_parameters = []
    resnet20_parameters.append(parameters[0])
    for i in range(3):
        resnet20_parameters.append([])
        for j in range(3):
            resnet20_parameters[i+1].append([])
            for k in range(2):
                resnet20_parameters[i+1][j].append(parameters[i*6+j*2+1+k])
                
    
    pruned_model = pruned_resnet20(resnet20_parameters)        
    
    return pruned_model