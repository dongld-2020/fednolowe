import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import random
import math
from statistics import median, mean
import matplotlib.pyplot as plt
import pandas as pd  # <--- Add this import
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.utils as nn_utils  # For gradient clipping
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Subset

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


       
# Define the VGG9 model 
class VGG9(nn.Module):
    def __init__(self):
        super(VGG9, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)  # 10 classes for CIFAR-10
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x

        
# Define the LeNet5 model 
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # First convolutional layer: 1 input channel (for grayscale), 32 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)  # Add batch normalization
        
        # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)  # Add batch normalization
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 256)  # Increased number of neurons
        self.fc2 = nn.Linear(256, 128)  # Add another fully connected layer
        self.fc3 = nn.Linear(128, 10)  # Output layer for 10 classes (digits 0-9)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply first convolutional layer + BatchNorm + ReLU + MaxPooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # Max pooling instead of average pooling

        # Apply second convolutional layer + BatchNorm + ReLU + MaxPooling
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 64 * 4 * 4)  # 64 feature maps of size 4x4
        
        # Fully connected layers with dropout and ReLU activations
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout for regularization
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation here for classification)
        
        return x




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Fix: Ensure FC1 is included
        self.fc2 = nn.Linear(512, 10)
        
        # Regularization
        self.dropout = nn.Dropout(0.3)        

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.dropout(F.relu(self.fc1(x)))  # Fix: Ensure FC1 is used
        x = self.fc2(x)  # No ReLU for final output
        
        return x


class VGG9_Ma(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG9_Ma, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 47, kernel_size=3, padding=1)  # Cin=3, Cout=47
        self.bn1_1 = nn.BatchNorm2d(47)
        self.conv1_2 = nn.Conv2d(47, 79, kernel_size=3, padding=1)  # Cin=47, Cout=79
        self.bn1_2 = nn.BatchNorm2d(79)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2_1 = nn.Conv2d(79, 143, kernel_size=3, padding=1)  # Cin=79, Cout=143
        self.bn2_1 = nn.BatchNorm2d(143)
        self.conv2_2 = nn.Conv2d(143, 143, kernel_size=3, padding=1)  # Cin=143, Cout=143
        self.bn2_2 = nn.BatchNorm2d(143)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3_1 = nn.Conv2d(143, 271, kernel_size=3, padding=1)  # Cin=143, Cout=271
        self.bn3_1 = nn.BatchNorm2d(271)
        self.conv3_2 = nn.Conv2d(271, 271, kernel_size=3, padding=1)  # Cin=271, Cout=271
        self.bn3_2 = nn.BatchNorm2d(271)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(271 * 4 * 4, 4336)  # Flattened input: 271 * 4 * 4
        self.bn_fc1 = nn.BatchNorm1d(4336)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4336, 527)
        self.bn_fc2 = nn.BatchNorm1d(527)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(527, 527)
        self.bn_fc3 = nn.BatchNorm1d(527)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(527, num_classes)  # Output layer

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)

        # Flatten and Fully Connected Layers
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                
        
def evaluate_global_model(model, test_loader, criterion, top_k=(1, 5)):
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)  # Scale by batch size

            # Top-1 prediction
            pred_top1 = output.argmax(dim=1, keepdim=True)
            correct_top1 += pred_top1.eq(target.view_as(pred_top1)).sum().item()

            # Top-5 predictions
            if 5 in top_k:
                _, pred_top5 = output.topk(5, dim=1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(target.view(-1, 1).expand_as(pred_top5)).sum().item()

            # Store predictions and targets for metrics
            all_preds.extend(pred_top1.squeeze().cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calculate metrics
    test_loss /= len(test_loader.dataset)  # Average loss per sample
    top1_accuracy = 100.0 * correct_top1 / len(test_loader.dataset)  # ADD THIS LINE
    top5_accuracy = 100.0 * correct_top5 / len(test_loader.dataset) if 5 in top_k else None

    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=1)
    cm = confusion_matrix(all_targets, all_preds)

    return test_loss, top1_accuracy, top5_accuracy, precision, recall, f1, cm

    

def non_iid_partition_dirichlet(dataset, arr_number_of_client, partition="hetero", alpha=0.1):
    """
    Partition dataset into IID or non-IID data for selected clients based on partition type.
    
    Args:
    - dataset: The dataset to partition (e.g., CIFAR-10).
    - arr_number_of_client: List of client indices selected for this round.
    - partition: Type of partition ("homo" for IID, "hetero-dir" for non-IID).
    - alpha: Dirichlet concentration parameter for "hetero-dir" partition.
    
    Returns:
    - clients_data: A dictionary where keys are client IDs (selected clients) and values are the data indices.
    - proportions: Proportions assigned to each client (for non-IID only).
    """
    num_clients = len(arr_number_of_client)
    y_train = np.array(dataset.targets)
    N = y_train.shape[0]

    # IID Partition
    if partition == "homo":
        # Shuffle indices and split equally among clients
        idxs = np.random.permutation(N)
        batch_idxs = np.array_split(idxs, num_clients)
        clients_data = {arr_number_of_client[i]: batch_idxs[i].tolist() for i in range(num_clients)}
        proportions = [1.0 / num_clients] * num_clients  # Equal proportions for IID partition

    # Non-IID Partition (Heterogeneous Dirichlet)
    elif partition == "hetero":
        K = len(np.unique(y_train))  # Number of classes
        min_size = 0

        # Repeat until each client has at least 10 samples
        while min_size < 10:
            idx_batch = [[] for _ in range(num_clients)]
            
            for k in range(K):  # For each class
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                
                # Dirichlet distribution for class allocation
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                
                # Adjust proportions to prevent clients from exceeding the average number of samples
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()  # Normalize
                
                # Convert proportions to cumulative counts to split idx_k
                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, split_points))]
            
            min_size = min([len(idx_j) for idx_j in idx_batch])

        # Map selected clients to their allocated data indices
        clients_data = {arr_number_of_client[i]: idx_batch[i] for i in range(num_clients)}
        proportions = [len(client_data) / N for client_data in idx_batch]

    return clients_data, proportions


    
# Load MNIST dataset
def get_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_data, test_data

# Load Fashio MNIST Fashion dataset
def get_fashion_mnist_data():
    # Define the transformation for the training dataset
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),        # Randomly flip images horizontally (Data Augmentation)
        transforms.ToTensor(),                    # Convert image to Tensor
        transforms.Normalize((0.2860,), (0.3530,))  # Normalize with Fashion MNIST mean and std
    ])
    
    # Define the transformation for the test dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),                    # Convert image to Tensor
        transforms.Normalize((0.2860,), (0.3530,))  # Normalize with Fashion MNIST mean and std
    ])
    
    # Download the Fashion MNIST dataset
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    
    return train_data, test_data
    
# Create dataloaders for each client
def get_dataloaders(train_data, clients_data, batch_size):
    dataloaders = {}
    for client_id, data_idxs in clients_data.items():
        client_dataset = torch.utils.data.Subset(train_data, data_idxs)
        dataloaders[client_id] = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
    return dataloaders

def get_cifar10_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly crop image to 32x32 with padding
        transforms.RandomHorizontalFlip(),     # Randomly flip images horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Change brightness/contrast/saturation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize with CIFAR-10 mean/std
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Same normalization for test set
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    return train_data, test_data


# Function to train a local model on a client's data
def train_local_model_avg(model, dataloader, criterion, optimizer, epochs=1):
    model.train()
    total_loss = 0.0  # Use float to accumulate correctly

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss (correctly scaled by batch size)
            epoch_loss += loss.item() * data.size(0)

        # Track loss across all epochs
        total_loss += epoch_loss

    # Average loss per sample over all epochs
    avg_loss = total_loss / (len(dataloader.dataset) * epochs)
    return model, avg_loss


# Function to train a local model on a client's data
def train_local_model_prox(global_model, model, dataloader, criterion, optimizer, epochs=1, mu=0.001):
    model.train()
    total_loss = 0.0  # Use float to accumulate correctly

    # Capture initial global parameters once at the start
    global_params = {name: param.detach().clone().to(device) for name, param in global_model.named_parameters()}
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Add FedProx proximal term
            prox_term = 0.0
            for name, param in model.named_parameters():
                prox_term += ((param - global_params[name]) ** 2).sum()
            loss += (mu / 2) * prox_term


            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss (correctly scaled by batch size)
            epoch_loss += loss.item() * data.size(0)

        # Track loss across all epochs
        total_loss += epoch_loss

    # Average loss per sample over all epochs
    avg_loss = total_loss / (len(dataloader.dataset) * epochs)
    return model, avg_loss

  
# Federated averaging (FedAvg) with weighted aggregation
def aggregate_avg(global_model, client_models, client_data_sizes):

    client_data_sizes = np.array(client_data_sizes)  # Ensure data size is a NumPy array
    client_data_sizes = client_data_sizes.astype(np.float64)  # Convert to float64 to avoid type casting issues
    total_client_data_sizes = client_data_sizes.sum()  # Normalize the data size
    
    weighted_sum = {key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float) for key in global_model.state_dict().keys()}
    
    # Loop over the parameters of the global model and perform the weighted sum
    for key in global_model.state_dict().keys():
        for i in range(len(client_models)):
            # Access the state_dict of each client model
            weighted_sum[key] += client_models[i].state_dict()[key].float() * client_data_sizes[i] / total_client_data_sizes 
        
    # Update the global model with the weighted sum
    global_model.load_state_dict(weighted_sum)

    return global_model

# Federated averaging (FedNolowe) with weighted aggregation

    

    
# Federated averaging (FedProx) with weighted aggregation
def aggregate_prox(global_model, client_models):
    
    weighted_sum = {key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float) for key in global_model.state_dict().keys()}
    
    # Loop over the parameters of the global model and perform the weighted sum
    for key in global_model.state_dict().keys():
        for i in range(len(client_models)):
            # Access the state_dict of each client model
            weighted_sum[key] += client_models[i].state_dict()[key].float()
            
        weighted_sum[key] = weighted_sum[key] / len(client_models)
        
    # Update the global model with the weighted sum
    global_model.load_state_dict(weighted_sum)

    return global_model
    
#FedMa aggregation (layers - wise matched average)
def aggregate_fedma(global_model, client_models):
    """
    A simplified layer-wise matched averaging.
    For each parameter tensor (if its dimension is >=2 and not an output layer), we use the first client as a reference.
    Then, for each other client we compute a cost (Euclidean distance) between each row (interpreted as a neuron/filter)
    and use the Hungarian algorithm to find a permutation that aligns the candidate to the reference.
    Finally, we average the aligned parameters.
    
    For output layers (e.g. keys containing 'fc3' or 'classifier') or 1D tensors, we use standard average.
    """
    global_state = global_model.state_dict()
    new_state = {}
    num_clients = len(client_models)
    
    for key in global_state.keys():
        # Get all clients' parameter tensors for this key.
        client_params = [client_models[i].state_dict()[key].float() for i in range(num_clients)]
        
        # If this is an output layer (we assume if key contains 'fc3' for LeNet5 or 'classifier' for others)
        # or if the parameter is 1D, then simply average.
        if ("fc3" in key or "classifier" in key) or (client_params[0].dim() < 2):
            avg = sum(client_params) / num_clients
            new_state[key] = avg
        else:
            # For weight parameters of conv or fc layers, we perform matching along the first dimension.
            # Use the first client's parameter as the reference.
            ref = client_params[0].clone().detach()
            n = ref.shape[0]
            ref_flat = ref.view(n, -1)  # shape: (n, d)
            aggregated = ref_flat.clone()
            
            count = 1
            # For each other client, compute a permutation and align
            for candidate_tensor in client_params[1:]:
                candidate_flat = candidate_tensor.view(n, -1)
                # Compute the cost matrix (Euclidean distances between rows)
                cost = torch.cdist(candidate_flat, ref_flat, p=2).cpu().numpy()
                # Solve the assignment problem
                row_ind, col_ind = linear_sum_assignment(cost)
                # Create a new tensor for the candidate with rows permuted to match ref.
                permuted = torch.zeros_like(candidate_flat)
                # For each index in the reference ordering (from 0 to n-1),
                # find the candidate row that was assigned to it.
                for j in range(n):
                    # Find index in candidate such that the assignment gives col_ind == j
                    idx = np.where(col_ind == j)[0]
                    if len(idx) > 0:
                        candidate_idx = row_ind[idx[0]]
                        permuted[j] = candidate_flat[candidate_idx]
                    else:
                        # if no matching found, keep candidate's original row j
                        permuted[j] = candidate_flat[j]
                aggregated += permuted
                count += 1
            aggregated = aggregated / count
            
            # Reshape aggregated tensor back to original shape.
            new_state[key] = aggregated.view(client_params[0].shape)
            
    global_model.load_state_dict(new_state)
    return global_model

def aggregate_asl(global_model, client_models, client_train_losses, alpha=0.5, beta=0.2):
    """
    FedASL aggregation: use only client training losses to compute aggregation weights.
    For each selected client:
      - Compute the median (med_loss) and standard deviation (sigma) of the losses.
      - For each client k, if its loss L_k is within [med_loss - alpha*sigma, med_loss + alpha*sigma],
        set d_k = beta*sigma; otherwise, set d_k = abs(med_loss - L_k).
      - Then, set the aggregation weight A_k = (1/d_k) / (sum_j (1/d_j)).
    Finally, update the global model as the weighted average of client models.
    """
    losses = np.array(client_train_losses, dtype=np.float64)
    med_loss = np.median(losses)
    sigma = np.std(losses)
    
    # In case sigma is zero (all losses equal), assign equal weights.
    if sigma == 0:
        weights = np.ones(len(losses)) / len(losses)
    else:
        d = []
        for L in losses:
            if (L >= med_loss - alpha * sigma) and (L <= med_loss + alpha * sigma):
                d.append(beta * sigma)
            else:
                d.append(abs(med_loss - L))
        print(d)
        d = np.array(d, dtype=np.float64)
        d = d / d.sum()
        inv_d = 1.0 / d
        weights = inv_d / inv_d.sum()
    print("FedASL weights:", weights)
    
    weighted_sum = {key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float)
                    for key in global_model.state_dict().keys()}
    for key in global_model.state_dict().keys():
        for i in range(len(client_models)):
            weighted_sum[key] += client_models[i].state_dict()[key].float() * weights[i]
            
    global_model.load_state_dict(weighted_sum)
    return global_model

def aggregate_nolowe(global_model, client_models, feedback_train_losses, clients_data_loaders, criterion):
    print("Clients train loss: ", feedback_train_losses)

    feedback_train_losses = np.array(feedback_train_losses, dtype=np.float64)

    if feedback_train_losses.sum() == 0:
        print("Warning: clients_train_loss sum to 0. Assigning equal weights.")
        feedback_train_losses = np.ones(len(feedback_train_losses)) / len(feedback_train_losses)
    else:
        feedback_train_losses /= feedback_train_losses.sum()

    feedback_train_losses = 1 - feedback_train_losses
    feedback_train_losses /= feedback_train_losses.sum()

    print("Normalized clients train loss: ", feedback_train_losses)

    # Step 1: Lấy gradient từng client trên dữ liệu của chính nó
    client_grad_vectors = []
    for model, dataloader in zip(client_models, clients_data_loaders):
        grad_vector = get_gradient_vector(model, dataloader, criterion)
        client_grad_vectors.append(grad_vector)

    # Step 2: Tính global gradient (tổng hợp từ gradient client, có trọng số)
    global_grad_vector = sum([client_grad_vectors[i] * feedback_train_losses[i] for i in range(len(client_models))])

    # Step 3: Cập nhật trọng số của mô hình toàn cục (vẫn theo tham số như cũ)
    weighted_sum = {key: torch.zeros_like(global_model.state_dict()[key], dtype=torch.float) 
                    for key in global_model.state_dict().keys()}
    
    for key in global_model.state_dict().keys():
        for i in range(len(client_models)):
            weighted_sum[key] += client_models[i].state_dict()[key].float() * feedback_train_losses[i]
    
    global_model.load_state_dict(weighted_sum)

    # Step 4: Tính cosine similarity giữa gradient từng client và global gradient
    cosine_similarities = []
    for grad_vector in client_grad_vectors:
        cos_sim = F.cosine_similarity(grad_vector.unsqueeze(0), global_grad_vector.unsqueeze(0), dim=1).item()
        cosine_similarities.append(cos_sim)

    return global_model, mean(cosine_similarities)

    
def get_gradient_vector(model, dataloader, criterion):
    model.zero_grad()
    model.train()

    # Lấy một batch đầu tiên (đủ để lấy gradient)
    data, target = next(iter(dataloader))
    data, target = data.to(device), target.to(device)

    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # Nối toàn bộ gradient thành một vector
    grad_vector = torch.cat([
        p.grad.flatten() for p in model.parameters() if p.grad is not None
    ]).detach().to(device)

    return grad_vector


def compute_cosine_similarity(model1, model2):
    """
    Compute cosine similarity between parameters of two models.
    Flatten all parameters into vectors and compute cosine similarity.
    
    Args:
        model1: First PyTorch model.
        model2: Second PyTorch model.
    
    Returns:
        float: Cosine similarity value between -1 and 1.
    """
    # Flatten parameters of model1
    params1 = torch.cat([p.flatten() for p in model1.parameters()]).detach().to(device)
    # Flatten parameters of model2
    params2 = torch.cat([p.flatten() for p in model2.parameters()]).detach().to(device)
    
    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0), dim=1)
    return cos_sim.item()
    
def split_test_dataset_evenly(dataset, num_clients):
    """
    Chia test dataset đồng đều cả về số lượng mẫu và phân phối lớp cho các clients.
    
    Args:
        dataset: Dataset cần chia (ví dụ: test_data).
        num_clients: Số lượng clients.
    
    Returns:
        clients_data: Danh sách các Subset cho từng client.
    """
    # Lấy labels từ dataset
    labels = np.array(dataset.targets)
    
    # Tạo danh sách các indices cho từng lớp
    class_indices = {}
    for class_label in np.unique(labels):
        class_indices[class_label] = np.where(labels == class_label)[0]
    
    # Chia đều mẫu trong từng lớp cho các clients
    clients_data = [[] for _ in range(num_clients)]
    for class_label, indices in class_indices.items():
        np.random.shuffle(indices)  # Xáo trộn indices để đảm bảo ngẫu nhiên
        split_indices = np.array_split(indices, num_clients)  # Chia đều indices
        for client_idx in range(num_clients):
            clients_data[client_idx].extend(split_indices[client_idx].tolist())
    
    # Tạo Subset cho từng client
    clients_subsets = [Subset(dataset, client_indices) for client_indices in clients_data]
    return clients_subsets    
  
# Main federated learning loop with loss tracking and plotting
# [Previous imports, model definitions, and functions like compute_client_gradients, compute_cosine_similarity, etc., remain unchanged]

# [Previous imports, model definitions, and functions like compute_client_gradients, compute_cosine_similarity, etc., remain unchanged]

def federated_learning(model_name='lenet5', algorithm='fedavg', num_clients=5, num_rounds=10, epochs=1, batch_size=32):
    set_seed(42)
    
    # Load data
    if model_name == 'vgg9':
        train_data, test_data = get_cifar10_data()
    elif model_name == 'cnn':
        train_data, test_data = get_fashion_mnist_data()
    elif model_name == 'lenet5':
        train_data, test_data = get_mnist_data()
    
    global_test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Initialize clients and data
    clients_data, proportions = non_iid_partition_dirichlet(train_data, list(range(num_clients)), alpha=0.1)
    print(f"Initial data proportions assigned to clients: {proportions}")
    
    clients_data_loaders = get_dataloaders(train_data, clients_data, batch_size)
    
    # Split test dataset
    clients_subsets = split_test_dataset_evenly(test_data, num_clients)
    clients_test_loaders = [DataLoader(subset, batch_size=32, shuffle=False) for subset in clients_subsets]
    
    # Initialize models
    if model_name == 'vgg9':
        global_model = VGG9().to(device)
        client_models = [VGG9().to(device) for _ in range(num_clients)]
    elif model_name == 'cnn':
        global_model = CNN().to(device)
        client_models = [CNN().to(device) for _ in range(num_clients)]
    elif model_name == 'lenet5':
        global_model = LeNet5().to(device)
        client_models = [LeNet5().to(device) for _ in range(num_clients)]
    
    client_optimizers = [optim.SGD(client_models[i].parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3) for i in range(num_clients)]
    criterion = nn.CrossEntropyLoss()
    
    client_data_sizes = [len(clients_data[i]) for i in range(num_clients)]
    feedback_train_loss = [0.0 for _ in range(num_clients)]
    
    # Lists to store metrics
    round_cosine_similarities = []  # For FedNoLoWe (stores cosine similarity per round)
    global_train_losses = []
    global_validation_losses = []
    global_top1_accuracies = []
    global_top5_accuracies = []
    global_precisions = []
    global_recalls = []
    global_f1_scores = []
    number_of_participants = []
    clients_id_per_round = []
    
    for round_num in range(num_rounds):
        print(f"===== Round {round_num + 1}/{num_rounds} of {algorithm} and {model_name} =====")
        
        selected_clients = random.sample(range(num_clients), 5)
        print(f"Selected clients for Round {round_num + 1}/{num_rounds}: {selected_clients}")
        number_of_participants.append(len(selected_clients))
        clients_id_per_round.append(selected_clients)
        
        # Distribute global model
        for client_idx in selected_clients:
            client_models[client_idx].load_state_dict(global_model.state_dict())
        
        total_train_loss = 0
        client_gradients = []
        
        for client_idx in selected_clients:
            client_optimizer = client_optimizers[client_idx]
            
            if algorithm in ['fedavg', 'fedma', 'fedasl']:
                client_models[client_idx], client_train_loss = train_local_model_avg(
                    client_models[client_idx], clients_data_loaders[client_idx], criterion, client_optimizer, epochs)
                if algorithm in ['fednolowe', 'fedasl']:
                    feedback_train_loss[client_idx] = client_train_loss
            elif algorithm == 'fednolowe':
                client_models[client_idx], client_train_loss = train_local_model_avg(
                    client_models[client_idx], clients_data_loaders[client_idx], criterion, client_optimizer, epochs)
                feedback_train_loss[client_idx] = client_train_loss
            
            print(f"Client {client_idx} Train Loss: {client_train_loss:.4f}")
            total_train_loss += client_train_loss
        
        avg_train_loss = total_train_loss / len(selected_clients)
        global_train_losses.append(avg_train_loss)
        print(f"Global Average Train Loss: {avg_train_loss:.4f}")
        
        # Aggregate models and compute cosine similarity for FedNoLoWe
        if algorithm == 'fedavg':
            global_model = aggregate_avg(global_model, [client_models[i] for i in selected_clients],
                                        [client_data_sizes[j] for j in selected_clients])
        elif algorithm == 'fednolowe':
            # Aggregate model
            global_model, cosine = aggregate_nolowe(
                global_model,
                [client_models[i] for i in selected_clients],
                [feedback_train_loss[j] for j in selected_clients],
                [clients_data_loaders[i] for i in selected_clients],
                criterion
            )                                      
                                           
            round_cosine_similarities.append(cosine)
            print(f"======>>>>>> Round cosine similarity between mean clients gradient and global gradient: {cosine:.4f}")
            
        elif algorithm == 'fedprox':
            global_model = aggregate_prox(global_model, [client_models[i] for i in selected_clients])
        elif algorithm == 'fedma':
            global_model = aggregate_fedma(global_model, [client_models[i] for i in selected_clients])
        elif algorithm == 'fedasl':
            global_model = aggregate_asl(global_model, [client_models[i] for i in selected_clients],
                                        [feedback_train_loss[j] for j in selected_clients])
        
        # Evaluate global model
        val_loss, top1_accuracy, top5_accuracy, precision, recall, f1, cm = evaluate_global_model(
            global_model, global_test_loader, criterion)
        
        global_validation_losses.append(val_loss)
        global_top1_accuracies.append(top1_accuracy)
        global_top5_accuracies.append(top5_accuracy)
        global_precisions.append(precision)
        global_recalls.append(recall)
        global_f1_scores.append(f1)
        
        print(f"Global Model Validation Loss: {val_loss:.4f}, Top1_Accuracy: {top1_accuracy:.2f}%, Top5_Accuracy: {top5_accuracy:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    # Compute mean and std of cosine similarities for FedNoLoWe
    if algorithm == 'fednolowe' and round_cosine_similarities:
        cos_values = round_cosine_similarities
        mean_cos = np.mean(cos_values) if cos_values else 0.0
        std_cos = np.std(cos_values) if cos_values else 0.0
        print(f"\nAverage Cosine Similarity: {mean_cos:.4f} ± {std_cos:.4f}")
    
    # Save metrics to CSV
    rounds = list(range(num_rounds))
    df_metrics = pd.DataFrame({
        'Round': rounds,
        'Global Train Loss': global_train_losses,
        'Global Validation Loss': global_validation_losses,
        'Global Top1 Accuracy (%)': global_top1_accuracies,
        'Global Top5 Accuracy (%)': global_top5_accuracies,
        'Precision': global_precisions,
        'Recall': global_recalls,
        'F1-Score': global_f1_scores,
        'Participants': number_of_participants,
        'Clients': clients_id_per_round,
    })
    
    csv_name = f"outcomes/metrics_{model_name}_{algorithm}_dong_clients{num_clients}_rounds{num_rounds}_clientsperround{len(selected_clients)}.csv"
    df_metrics.to_csv(csv_name, index=False)
    
    # Plot train vs validation loss
    plt.figure(figsize=(12, 8))
    plt.plot(rounds, global_train_losses, label='Client Average Train Loss', linewidth=3)
    plt.plot(rounds, global_validation_losses, label='Global Validation Loss', linewidth=3)
    plt.title(f"{model_name}_{algorithm}_Global Train Loss vs Validation Loss", fontsize=25)
    plt.xlabel("Rounds", fontsize=25)
    plt.ylabel("Loss", fontsize=25)
    plt.xticks(rounds[::10], fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.savefig(f"outcomes/Loss_{model_name}_{algorithm}_clients{num_clients}_rounds{num_rounds}_clientsperround{len(selected_clients)}.png")
    plt.show()
    
    # Plot accuracy
    plt.figure(figsize=(12, 8))
    plt.plot(rounds, global_top1_accuracies, label='Global Top1 Accuracy', linewidth=3)
    plt.plot(rounds, global_top5_accuracies, label='Global Top5 Accuracy', linewidth=3)
    plt.title(f"{model_name}_{algorithm}_Global Accuracy Comparation", fontsize=25)
    plt.xlabel("Rounds", fontsize=25)
    plt.ylabel("Accuracy (%)", fontsize=25)
    plt.xticks(rounds[::10], fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.savefig(f"outcomes/Accuracy_{model_name}_{algorithm}_clients{num_clients}_rounds{num_rounds}_clientsperround{len(selected_clients)}.png")
    plt.show()

# Run federated learning
model_name = input("Enter model name (lenet5, cnn, vgg9): ").strip().lower()
algorithm = input("Enter algorithm (fedavg, fedprox, fednolowe, fedma, fedasl): ").strip().lower()

if model_name not in ["cnn", "lenet5", "vgg9"]:
    raise ValueError("Invalid model name. Choose 'cnn', 'lenet5', 'vgg9'.")
if algorithm not in ["fedavg", "fedprox", "fednolowe", 'fedma', 'fedasl']:
    raise ValueError("Invalid algorithm. Choose 'fedavg', 'fedprox', 'fednolowe', 'fedma', 'fedasl'.")

federated_learning(model_name=model_name, algorithm=algorithm, num_clients=50, num_rounds=50, epochs=2, batch_size=32)