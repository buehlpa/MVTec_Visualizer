import torch
import pandas as pd
import numpy as np


def sample_synthetic_norm(torch_array):
    
    #means=torch_array.mean(dim=0)
    
    means=torch.zeros(torch_array.size(1))
    stds=torch.sqrt(torch_array.var(dim=0))
    rows = []
    for _ in range(30):
        
        random_row = torch_array[torch.randint(0, torch_array.size(0), (1,)).item()]
        row = random_row+torch.normal(means, stds)
        
        
        rows.append(row)
        
    new_samples = torch.stack(rows)
    return new_samples



def create_synset_for_class(category:str,df,sampler=sample_synthetic_norm):

    # structure of combined_samples: ["good train and test!","anomaly","synthetic_anomaly","anomaly1","synthetic_anomaly1","anomaly2","synthetic_anomaly2".....]
    # zb category='bottle'


    anomaly_categories = {
        'bottle': ['broken_large', 'broken_small', 'contamination'],
        'cable': ['bent_wire', 'cable_swap', 'combined', 'cut_inner_insulation', 'cut_outer_insulation', 'missing_cable', 'missing_wire', 'poke_insulation'],
        'capsule': ['crack', 'faulty_imprint', 'poke', 'squeeze'],
        'carpet': ['color', 'cut', 'hole', 'metal_contamination', 'thread'],
        'grid': ['bent', 'broken', 'glue', 'metal_contamination'],
        'hazelnut': ['crack', 'cut', 'hole', 'print'],
        'leather': ['color', 'cut', 'fold', 'glue', 'poke'],
        'metal_nut': ['bent', 'color', 'flip', 'scratch'],
        'pill': ['color', 'contamination', 'crack', 'faulty_imprint', 'pill_type'],
        'screw': ['manipulated_front', 'scratch_head', 'thread_side', 'thread_top'],
        'tile': ['crack', 'glue_strip', 'gray_stroke', 'oil'],
        'toothbrush': ['defective'],
        'transistor': ['bent', 'cut', 'damaged_case', 'misplaced'],
        'wood': ['color', 'combined', 'hole', 'liquid', 'scratch'],
        'zipper': ['broken_teeth', 'fabric_border', 'split_teeth', 'squeezed_teeth']
    }



    df_category = df[df.index.str.contains(category)]

    all_data=[]
    all_data.append(torch.Tensor(df_category.to_numpy()))
    class_list=[category]

    for anocat in anomaly_categories[category]:
        df_subcategory = df_category[df_category.index.str.contains(anocat)]
        df_subcategory.head()
        torch_array=torch.Tensor(df_subcategory.to_numpy())
        
        all_data.append(torch_array)
        class_list.append(anocat)
        
        

        new_samples=sampler(torch_array)
        
        
        all_data.append(new_samples)
        class_list.append(anocat+'_synthetic')
        
    combined_samples = torch.cat(all_data).numpy()


    # Create labels for each class
    labels = []
    for idx, data in enumerate(all_data):
        labels.extend([idx] * data.size(0))  # Each class gets a unique integer label
    labels = np.array(labels)

    return combined_samples,labels,class_list