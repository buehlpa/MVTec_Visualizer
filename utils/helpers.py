import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# sample synthetic 
def sample_synthetic_add_gauss(torch_array,n_synthetic=30):
    # sampling from test set an add gaussian noise to it
    

    
    means=torch.zeros(torch_array.size(1))
    stds=torch.sqrt(torch_array.var(dim=0))
    
    
    # # test with shift
    # shift=torch.full((torch_array.size(1),), 0.1)
    # stds=torch_array.var(dim=0)**8

    rows = []
    for _ in range(n_synthetic):
        
        random_row = torch_array[torch.randint(0, torch_array.size(0), (1,)).item()]
        
        
        row = random_row+torch.normal(means, stds)
        
        # #test with shift
        # row=random_row+shift
        
        rows.append(row)
        
    new_samples = torch.stack(rows)
    return new_samples

def sample_synthetic_norm(torch_array,n_synthetic=30):
    # empirical variance and means -> sample from assummed multivariate normal distribution
    
    means=torch_array.mean(dim=0)
    stds=torch.sqrt(torch_array.var(dim=0))
    
    rows = []
    for _ in range(n_synthetic):
        row = torch.normal(means, stds)
        rows.append(row)
        
    new_samples = torch.stack(rows)
    return new_samples


def create_synset_for_class(category:str,df,n_synthetic=30,sampler="additive_0_mean_gauss"):

    # structure of combined_samples: ["good train and test!","anomaly","synthetic_anomaly","anomaly1","synthetic_anomaly1","anomaly2","synthetic_anomaly2".....]
    # zb category='bottle'

    if sampler == "additive_0_mean_gauss":
        sampler_function=sample_synthetic_add_gauss
    
    if sampler == "multivariate_gauss":
        sampler_function=sample_synthetic_norm

    anomaly_categories = {
        'bottle': ['broken_large', 'broken_small', 'contamination'],
        'cable': ['bent_wire', 'cable_swap', 'combined', 'cut_inner_insulation', 'cut_outer_insulation', 'missing_cable', 'missing_wire', 'poke_insulation'],
        'capsule': ['crack', 'faulty_imprint', 'poke', 'scratch','squeeze'],
        'carpet': ['color', 'cut', 'hole', 'metal_contamination', 'thread'],
        'grid': ['bent', 'broken', 'glue', 'metal_contamination', 'thread'],
        'hazelnut': ['crack', 'cut', 'hole', 'print'],
        'leather': ['color', 'cut', 'fold', 'glue', 'poke'],
        'metal_nut': ['bent', 'color', 'flip', 'scratch'],
        'pill': ['color', 'combined','contamination', 'crack', 'faulty_imprint', 'pill_type','scratch'],
        'screw': ['manipulated_front', 'scratch_head', 'scratch_neck','thread_side', 'thread_top'],
        'tile': ['crack', 'glue_strip', 'gray_stroke', 'oil','rough'],
        'toothbrush': ['defective'],
        'transistor': ['bent_lead', 'cut_lead', 'damaged_case', 'misplaced'],
        'wood': ['color', 'combined', 'hole', 'liquid', 'scratch'],
        'zipper': ['broken_teeth', 'combined','fabric_border', 'fabric_interior','split_teeth','rough', 'squeezed_teeth']}


    df_category = df[df.index.str.contains(category)]

    all_data=[]
    # tensor with all good samples from train and test
    all_data.append(torch.Tensor(df_category[df_category.index.str.contains('good')].to_numpy()))
    
    class_list=[category]
    # tensor with anomalies for the category , create anomalies for every subcategory
    for anocat in anomaly_categories[category]:
        df_subcategory = df_category[df_category.index.str.contains(anocat)]
        df_subcategory.head()
        torch_array=torch.Tensor(df_subcategory.to_numpy())
        
        all_data.append(torch_array)
        class_list.append(anocat)
        
        new_samples=sampler_function(torch_array,n_synthetic=n_synthetic)
        
        all_data.append(new_samples)
        class_list.append(anocat+'_synthetic')
        
    combined_samples = torch.cat(all_data).numpy()


    # Create labels for each class
    labels = []
    for idx, data in enumerate(all_data):
        labels.extend([idx] * data.size(0))  # Each class gets a unique integer label
    labels = np.array(labels)

    return combined_samples,labels,class_list





# plotting for the normal / anomaly / synthetic anomay numbers
def plot_data_distribution(normals, anomalies, synthetic_anomalies,info):
    # Categories and their corresponding values
    categories = ['Good Test+Train', 'Anomalies Test', 'Synthetic Anomalies Test+Gauss']
    values = [normals.shape[0], anomalies.shape[0], synthetic_anomalies.shape[0]]
    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=['blue', 'red', 'green'])

    # Adding value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')
        
    plt.title(f'N samples: {info["category"]} , {info["filename"]}')
    plt.ylabel('Count')

    plt.show()
    
    
    
    
    
#####  autencoder
class SimpleAutoencoder(nn.Module):
    def __init__(self,input_shape=200704):
        super(SimpleAutoencoder, self).__init__()
        # Encoder layers
        self.input_shape=input_shape
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_shape, 1000),  # Reducing dimension from 200704 to 1000
            nn.ReLU(),
            nn.Linear(1000, 500),     # Further reduction to 500
            nn.ReLU(),
            nn.Linear(500, 100)       # Code layer with 100 features
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(100, 500),      # Expanding from 100 to 500
            nn.ReLU(),
            nn.Linear(500, 1000),     # Expanding from 500 to 1000
            nn.ReLU(),
            nn.Linear(1000, self.input_shape)   # Reconstructing the original 200704 features
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




def train_autoencoder(input_data: torch.Tensor, num_epochs:50, info:dict, batch_size=10, learning_rate=0.01, patience=5):
    
    # Splitting data into training and validation sets
    train_data, val_data = train_test_split(input_data, test_size=0.2, random_state=42)

    # Creating TensorDatasets for training and validation sets
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    # Creating DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)



    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = SimpleAutoencoder(input_shape=input_data.shape[1]).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    best_val_loss = np.inf
    epochs_no_improve = 0
    train_loss_log = []
    val_loss_log = []

    for epoch in range(num_epochs):
        autoencoder.train()
        train_loss = 0.0

        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = loss_function(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_loss_log.append(train_loss)

        autoencoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                outputs = autoencoder(inputs)
                loss = loss_function(outputs, inputs)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_loss_log.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if epochs_no_improve == patience:
            print('Early stopping triggered!')
            break

    # Save the model
    #torch.save(autoencoder.state_dict(), 'autoencoder.pth')

    # Plotting the validation and training loss with logarithmic scale
    plt.plot(train_loss_log, label='Training Loss')
    plt.plot(val_loss_log, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Validation and Training Loss {info['filename']} {info['category']}")
    plt.legend()
    plt.yscale('log')  # Set y-axis scale to logarithmic
    plt.show()

    return autoencoder
