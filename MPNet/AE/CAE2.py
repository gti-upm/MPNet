import os
import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np

# Parameters
path_obs_db = '/mnt/hgfs/Databases/dataset2/obs_cloud'  # Path to obstacle database
model_path = './output/models/'  # Path for saving trained models.
output_path = './output/'
num_epochs = 400
batch_size = 100
lam = 1e-3 # Contractive loss function
max_obs_points = 1400  # All environments have the same number of obstacles # TODO: change the format to allow a variable number

# Load environments with obstacles
def load_dataset(path_obs_db):
    # Get names of dat files with obstacles (every file contains the obstacles of a different environment).
    # An obstacle is defined by a (x,y) coordinates. Thus, a dat file contains a sequence of obstacles:
    # x1,y1,x2,y2,...,xn,yn.
    file_list = []
    for file in os.listdir(path_obs_db):
        if file.endswith(".dat"):
            file_list.append(file)
    num_envir = len(file_list)
    obstacles = []
    # Load obstacle data.
    for i in range(0, num_envir):
        print(os.path.join(path_obs_db, file_list[i]))
        obstacle_i = np.float32(np.fromfile(os.path.join(path_obs_db, file_list[i])))
        if len(obstacle_i) != max_obs_points:
            sys.exit("All environment should have {} obstacle points (and every point has two coordinates).".format(max_obs_points))
        obstacles.append(obstacle_i)
    return obstacles

# A list of vector arrays of obstacles coordinates: x1,y1,x2,y2,...,xn,yn. Every list element contains an
# array of obstacle coordinates of a different environment.
obs = load_dataset(path_obs_db)

# Database spliting
ind = int(np.round(0.8 * len(obs)))
train_data = obs[:ind]
validation_data = obs[ind+1:]

# Encoder and decoder models
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(max_obs_points, 512), nn.PReLU(), nn.Linear(512, 256), nn.PReLU(),
                                     nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, 28))

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(28, 128), nn.PReLU(), nn.Linear(128, 256), nn.PReLU(),
                                     nn.Linear(256, 512), nn.PReLU(), nn.Linear(512, max_obs_points))

    def forward(self, x):
        x = self.decoder(x)
        return x


encoder = Encoder()
decoder = Decoder()
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()

# Training paramaters
# Loss function
mse_loss = nn.MSELoss()
def loss_function(W, x, recons_x, h, lam):
    # W is shape of N_hidden x N. So, we do not need to transpose it as opposed to
    # http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
    mse = mse_loss(recons_x, x)
    dh = h*(1-h) # N_batch x N_hidden
    contractive_loss = torch.sum(Variable(W)**2, dim=1).sum().mul_(lam)
    return mse + contractive_loss

# Optimizer
params = list(encoder.parameters())+list(decoder.parameters())
optimizer = torch.optim.Adagrad(params)

# Training
if not os.path.exists(model_path):
    os.makedirs(model_path)
total_loss = []
for epoch in range(num_epochs):
    print("epoch" + str(epoch))

    # Training loop
    avg_loss = 0
    for i in range(0, len(train_data), batch_size):
        decoder.zero_grad()
        encoder.zero_grad()
        # Batch creation. TODO: random batch creation and data augmentation.
        if i + batch_size < len(train_data):
            inp = train_data[i:i + batch_size]
        else:
            inp = train_data[i:]
        inp = torch.from_numpy(np.asarray(inp))
        if torch.cuda.is_available():
            inp = Variable(inp).cuda()
        else:
            inp = Variable(inp)

        # Forward pass
        h = encoder(inp)
        output = decoder(h)
        keys = encoder.state_dict().keys()
        # Regularize or contracting last layer of encoder. Print keys to displace the layers name.
        W = encoder.state_dict()['encoder.6.weight']
        loss = loss_function(W, inp, output, h, lam)
        avg_loss = avg_loss + loss.data.item()
        # Backward pass
        loss.backward()
        optimizer.step()

    # Training metrics
    print("--average loss:")
    print(avg_loss / (len(train_data) / batch_size))
    total_loss.append(avg_loss / (len(train_data) / batch_size))

    # Validation loop
    avg_loss = 0
    for i in range(0, len(validation_data), batch_size):
        # Batch creation. TODO: data augmentation.
        if i + batch_size < len(validation_data):
            inp = validation_data[i:i + batch_size]
        else:
            inp = validation_data[i:]
        inp = torch.from_numpy(np.asarray(inp))
        if torch.cuda.is_available():
            inp = Variable(inp).cuda()
        else:
            inp = Variable(inp)
        # Forward pass
        output = encoder(inp)
        output = decoder(output)
        loss = mse_loss(output, inp)
        avg_loss = avg_loss + loss.data.item()

    # Validation metrics
    print("--Validation average loss:")
    print(avg_loss / (len(validation_data) / batch_size))

    # Save models TODO: Save only best models
    torch.save(encoder.state_dict(), os.path.join(model_path, 'cae_encoder.pkl'))
    torch.save(decoder.state_dict(), os.path.join(model_path, 'cae_decoder.pkl'))
    torch.save(total_loss, os.path.join(output_path, ' total_loss_AE.dat'))
