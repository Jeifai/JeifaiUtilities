import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as data_utils

dataset = pd.read_csv('ml_dataset.csv')

categorical_columns = ['seniority', 'company_code', 'country_code']
numerical_columns = ['day', 'month', 'days']
outputs = ['is_it_job']

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')

seniority = dataset['seniority'].cat.codes.values
company_code = dataset['company_code'].cat.codes.values
country_code = dataset['country_code'].cat.codes.values

categorical_data = np.stack([seniority, company_code, country_code], 1)

categorical_data = torch.tensor(categorical_data, dtype=torch.int64)

numerical_data = np.stack([dataset[col].values for col in numerical_columns], 1)
numerical_data = torch.tensor(numerical_data, dtype=torch.float)

outputs = torch.tensor(dataset[outputs].values).flatten()

# Define the embedding size (vector dimensions) for all the categorical columns
categorical_column_sizes = [len(dataset[column].cat.categories) for column in categorical_columns] # [4, 175, 123]
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes] # [(4, 2), (175, 50), (123, 50)]

total_records = len(dataset)
test_records = int(total_records * .2)

categorical_train_data = categorical_data[:total_records-test_records]
categorical_test_data = categorical_data[total_records-test_records:total_records]

numerical_train_data = numerical_data[:total_records-test_records]
numerical_test_data = numerical_data[total_records-test_records:total_records]

train_outputs = outputs[:total_records-test_records]
test_outputs = outputs[total_records-test_records:total_records]

class Model(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()

        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        '''
        ModuleList(
          (0): Embedding(4, 2)
          (1): Embedding(175, 50)
          (2): Embedding(123, 50)
        )
        '''

        self.embedding_dropout = nn.Dropout(p)
        '''
        Dropout(p=0.4, inplace=False)
        '''

        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols) # Applies Batch Normalization
        '''
        BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        '''

        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        '''
        102
        '''

        input_size = num_categorical_cols + num_numerical_cols
        '''
        105
        '''

        all_layers = []
        for i in layers:

            # Linear Layers
            all_layers.append(nn.Linear(input_size, i)) # Linear(in_features=105, out_features=200, bias=True)
            
            # Non-linear Activations
            all_layers.append(nn.ReLU(inplace=True)) # ReLU(inplace=True)
            
            # Normalization Layers
            all_layers.append(nn.BatchNorm1d(i)) # BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            
            # Dropout Layers
            all_layers.append(nn.Dropout(p)) # Dropout(p=0.4, inplace=False)
            
            input_size = i # The input size of next layer is the output size of the actual layer

        all_layers.append(nn.Linear(layers[-1], output_size))
        '''
        [
          Linear(in_features=105, out_features=200, bias=True),
          ReLU(inplace=True),
          BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          Dropout(p=0.4, inplace=False),
          Linear(in_features=200, out_features=100, bias=True),
          ReLU(inplace=True),
          BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          Dropout(p=0.4, inplace=False),
          Linear(in_features=100, out_features=50, bias=True),
          ReLU(inplace=True),
          BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          Dropout(p=0.4, inplace=False),
          Linear(in_features=50, out_features=2, bias=True)]
        '''

        self.layers = nn.Sequential(*all_layers) # Sequential container
        '''
        Sequential(
          (0): Linear(in_features=105, out_features=200, bias=True)
          (1): ReLU(inplace=True)
          (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): Dropout(p=0.4, inplace=False)
          (4): Linear(in_features=200, out_features=100, bias=True)
          (5): ReLU(inplace=True)
          (6): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (7): Dropout(p=0.4, inplace=False)
          (8): Linear(in_features=100, out_features=50, bias=True)
          (9): ReLU(inplace=True)
          (10): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (11): Dropout(p=0.4, inplace=False)
          (12): Linear(in_features=50, out_features=2, bias=True)
        )
        '''

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1) # Concatenates sequence of tensors
        ''''
        tensor(
          [[ 0.3887,  1.2546,  0.3901,  ...,  1.3139, -1.8194,  0.1141],
          [ 0.3887,  1.2546,  0.3901,  ..., -0.1104, -0.1457,  0.8401],
          [-1.5241,  2.0882,  0.3901,  ...,  0.6592, -0.7444,  0.7443],
          ...,
          [ 0.3887,  1.2546, -1.4994,  ..., -1.4686,  1.2227,  0.3452],
          [ 0.3887,  1.2546, -1.4994,  ..., -1.1852, -0.0527, -0.0741],
          [-1.5241,  2.0882, -1.4994,  ..., -1.6618,  1.1295,  0.4100]],
          grad_fn=<CatBackward>)
        '''
        x = self.embedding_dropout(x) # Set Dropout as done above
        x_numerical = self.batch_norm_num(x_numerical) # Applies Batch Normalization as done above
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x

# Define the loss function and the optimizer that will be used to train the model.

model = Model(
  categorical_embedding_sizes, # [(4, 2), (175, 50), (123, 50)]
  numerical_data.shape[1], # 3
  2, 
  [200,100,50], 
  p=0.4
)
'''
Model(
  (all_embeddings): ModuleList(
    (0): Embedding(4, 2)
    (1): Embedding(175, 50)
    (2): Embedding(123, 50)
  )
  (embedding_dropout): Dropout(p=0.4, inplace=False)
  (batch_norm_num): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layers): Sequential(
    (0): Linear(in_features=105, out_features=200, bias=True)
    (1): ReLU(inplace=True)
    (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.4, inplace=False)
    (4): Linear(in_features=200, out_features=100, bias=True)
    (5): ReLU(inplace=True)
    (6): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.4, inplace=False)
    (8): Linear(in_features=100, out_features=50, bias=True)
    (9): ReLU(inplace=True)
    (10): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): Dropout(p=0.4, inplace=False)
    (12): Linear(in_features=50, out_features=2, bias=True)
  )
)
'''

loss_function = nn.CrossEntropyLoss()
'''
CrossEntropyLoss()
'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
'''
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.001
    weight_decay: 0
)
'''

epochs = 300
aggregated_losses = []

for i in range(epochs):
    i += 1

    y_pred = model(categorical_train_data, numerical_train_data)
    '''
    tensor([[-4.9960e-01, -1.0694e-03],
            [ 1.0622e+00,  9.1217e-01],
            [-5.6726e-01,  1.5883e-03],
            ...,
            [-8.9797e-01, -1.3528e+00],
            [ 3.4734e-01,  7.2590e-01],
            [-8.4607e-01, -6.6772e-01]], grad_fn=<AddmmBackward>)
    '''

    single_loss = loss_function(y_pred, train_outputs)
    '''
    tensor(0.8073, grad_fn=<NllLossBackward>)
    '''

    aggregated_losses.append(single_loss)
    if i%25 == 1:
        print("epoch: " + str(i) + "\tloss: " + str(single_loss.item()))

    optimizer.zero_grad() # Set the gradients to zero before starting to do backpropragation
    single_loss.backward() # Computes the gradient of current tensor updating the weight
    optimizer.step() # Performs a single optimization step updating the gradient

print("epoch: " + str(i) + "\tloss: " + str(single_loss.item()))

plt.plot(range(epochs), aggregated_losses)
plt.ylabel('Loss')
plt.xlabel('epoch');

# Making Predictions
with torch.no_grad():
    y_val = model(categorical_test_data, numerical_test_data)
    loss = loss_function(y_val, test_outputs)
print("Loss: " + str(loss))
y_val = np.argmax(y_val, axis=1)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("confusion_matrix")
print(confusion_matrix(test_outputs, y_val))
print("\nclassification_report")
print(classification_report(test_outputs,y_val))
print("\naccuracy_score")
print(accuracy_score(test_outputs, y_val))