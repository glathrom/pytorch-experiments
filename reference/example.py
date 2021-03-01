import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader



# create the neural network model
class NN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2)
        # self.fc2 = nn.Linear(50, output_dim)


    def forward(self,x):
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        return x



# generate a model which has an input size of 28x28 = 784
# and an output size of 10
model = NN(2,2)

print('affine matrix')
print(model.fc1.weight)

print('\naffine translation vector')
print(model.fc1.bias)

# generate a random sample of 64 sample points from our 784 dimensional sample space
x = torch.randn(5, 2)
print('\nData Points')
print(x)
print('\n',model(x))



# worrying about the hardware to calculate these things
# no cuda driver is available to use a gpu so this is irrelevant
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setting hyperparameters
input_dim = 2
output_dim = 2
learning_rate = 0.01
batch_size = 10
num_epochs = 1

# load the data by filling in the following 
# pytorch datasets along with the training loaders

# train_dataset = 
# train_loader = 
# test_dataset = 
# test_loader = 


# initialize the model
model = NN(input_dim = input_dim, output_dim=output_dim)

# loss and optimzer function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# training the network
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device)

        # run the data through the model
        scores = model(data)

        # zero all the gradients
        optimizer.zero_grad()
        
        # calculate the gradient vector for the parameters
        loss.backward()

        # scale the gradient and update
        optimzer.step()


# checking accuracy 
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            # loading data as modified for gaussian model

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)}')
