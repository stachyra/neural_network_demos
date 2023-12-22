import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy

# -----------------
# Part 1: Load Data
# -----------------

# Define lookup table between native integer labels vs. text string labels
classes = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

# The default sort order for the classes is kind of arbitrary, so for display
# purposes, put things in more logical order: larger / bulkier tops first
# (Dress, Coat, Pullover), then smaller / closer fitting tops next (Shirt, 
# T-shirt/top), then bottoms (Trouser), then footwear, from bulkiest to
# lightest (Ankle boot, Sneaker, Sandal), then accessories (Bag) last.
clssrt = ['Dress', 'Coat', 'Pullover', 'Shirt', 'T-shirt/top', 'Trouser',
    'Ankle boot', 'Sneaker', 'Sandal', 'Bag']

# Load data, using train / test split suggested by pytorch
data, X, y = dict(), dict(), dict()
for splitid, option in zip(['train', 'test'], [True, False]):
    # Input format typically preferred by pytorch
    data[splitid] = datasets.FashionMNIST(root="data", train=option,
        download=True, transform=ToTensor())
    # Input format typically preferred by scikit-learn
    X[splitid] = [np.array(item[0].flatten()) for item in data[splitid]]
    y[splitid] = [classes[item[1]] for item in data[splitid]]

# --------------------------------------------------
# Part 2: Display Sample Data and Class Distribution
# --------------------------------------------------

# For each of the classes, select the first 5 samples to display as images
nc, nr = 5, len(classes)
# Counts number of items encountered so far in each class
count = np.zeros(nr, dtype=int)
# Get image size
depth, hgt, wd = data['train'][0][0].shape
# Array of sample images that we wish to display
sample_images = np.zeros([nc*nr, hgt, wd])
# Array of sample class IDs
sample_classid = np.zeros(nc*nr, dtype=int)
# Indices (into full data) set of samples that we filtered out for displayed
sample_index = np.zeros(nc*nr, dtype=int)
# Loop through training data to extract a total nc X nr samples for display 
for ii in range(len(data['train'])):
    # Integer class ID
    classid = data['train'][ii][1]
    # Extract samples from each class until each class is filled with
    # nc samples.  Use clssrt to sort them vertically.
    if count[classid] < nc:
        idxdsp = clssrt.index(classes[classid])
        sample_images[idxdsp * nc + count[classid]] = data['train'][ii][0]
        sample_classid[idxdsp * nc + count[classid]] = data['train'][ii][1]
        sample_index[idxdsp * nc + count[classid]] = ii
        count[classid] += 1
    # Break out of loop when all classes have enough samples
    if np.all(count == nc):
        break

# Prepare individual titles for each image in an nc X nr sized array of images
sample_titles = list()
for ii in range(nc*nr):
    sample_titles.append('{0} [{1}]'.format(classes[sample_classid[ii]],
        sample_index[ii]))

# Plot an nc X nr array of sample images
fig, axes = plt.subplots(nr, nc, figsize=(6,12))
for img, ttl, ax in zip(sample_images, sample_titles, axes.flatten()):
    ax.imshow(img, cmap='Blues')
    ax.set_title(ttl, fontsize=10)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
fig.tight_layout()
fig.savefig('figures/sample_images.png', dpi=150)

# Prepare histograms showing numbers of members of each class across
# both the training as well as the test split
df = pd.DataFrame(data={'classname': y['train'] + y['test'],
    'subset': ['train'] * len(y['train']) + ['test'] * len(y['test'])})
df['classname'] = pd.Categorical(df['classname'], clssrt)
g = sns.FacetGrid(df, row='subset')
g.figure.set_size_inches(6,9)
g.map(sns.histplot, 'classname')
for ax in g.axes.flat:
    lbl = ax.get_xticklabels()
    ax.tick_params(axis='x', labelrotation=-90)
g.tight_layout()
g.savefig('figures/class_distribution.png', dpi=150)
#plt.show()

# -----------------------------------------------------------------------
# Part 3: Initial Benchmark Classification Performance Using scikit-learn
# -----------------------------------------------------------------------

# Optimize over hyperparameters of K Nearest Neighbors and Random Forest
classifier = {'knn': KNeighborsClassifier(), 'rfc': RandomForestClassifier()}
# 1-D grid of example hyperparameters to be tuned
param = {'knn': {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]},
    'rfc': {'max_depth': [4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35,
    40]}}
testscr = {'knn': [], 'rfc': []}
gscv = dict()
for ctyp in ['knn', 'rfc']:
    print(f'Now performing grid search cross validation for classifier: {ctyp}')
    # Perform 10-fold cross-validation in parallel across the entire grid,
    # for each classifer type
    gscv[ctyp] = GridSearchCV(estimator=classifier[ctyp],
        param_grid=param[ctyp], n_jobs=-1, cv=10, return_train_score=True)
    gscv[ctyp].fit(X['train'], y['train'])
    print(gscv[ctyp].cv_results_)
    # Perform final holdout testing, but as a learning / illustration
    # exercise, do it over _all_ grid locations instead of just testing
    # the optimal value from the cross-validation stage
    cls = classifier[ctyp]
    pmgrd = ParameterGrid(param[ctyp])
    print(f'Now performing final holdout test evaluation for classifier: {ctyp}')
    for pm in pmgrd:
        cls.set_params(**pm)
        cls.fit(X['train'], y['train'])
        testscr[ctyp].append(cls.score(X['test'], y['test']))

# ----------------------------------------------------------
# Part 4: Plot scikit-learn Benchmark Classification Results
# ----------------------------------------------------------

fig = plt.figure(figsize=(10,6))
# Plot accuracy vs. n_neighbors for K Nearest Neigbors, and accuracy vs.
# max_depth for Random Forest, over training, validation, and final holdout
# test data
axlst = []
lgd = {'train': 'Training', 'test': 'Validation'}
for plidx, c, hp, ttl in zip(range(2), ('knn', 'rfc'),
    ('n_neighbors', 'max_depth'), ('K Nearest Neighbors', 'Random Forest')):
    # For the left-most subplot, include a y-axis label, but for brevity
    # omit the y-axis label for plots to the right of that 
    if plidx == 0:
        axlst.append(fig.add_subplot(1, 2, plidx+1))
        axlst[0].set_ylabel('Accuracy')
    # For all plots after the first one, share the y-axis so that the
    # vertical scale will be the same
    else:
        axlst.append(fig.add_subplot(1, 2, plidx+1, sharey=axlst[0]))
    # Dictionary of handles so that we can sort the legend lables logically
    # (Training / Validation / Test) rather than default to alphabetically
    # (Test / Training / Validation)
    hdl = dict()
    # Labels here are a bit misleading; it's actually training vs. validation,
    # but scikit-learn sometimes refers to validation items as test, so
    # stick to their terminology to maintain compatibility
    for lbl in ('train', 'test'):
        mn = 'mean_' + lbl + '_score'
        st = 'std_' + lbl + '_score'
        hdl[lbl] = axlst[plidx].errorbar(x=param[c][hp],
            y=gscv[c].cv_results_[mn], yerr=gscv[c].cv_results_[st],
            capsize=2, marker='o', ms=5, mfc='none', label=lgd[lbl])
    # Plot final holdout test data as well
    hdl['holdout'] = axlst[plidx].errorbar(param[c][hp], testscr[c],
        marker='o', ms=3, label='Test')
    axlst[plidx].set_title(ttl)
    axlst[plidx].set_xlabel(hp)
# Include a legend, but for brevity only print it on the final subplot
axlst[1].legend(handles=[hdl['train'], hdl['test'], hdl['holdout']])
fig.savefig('figures/accuracy_vs_hyperparameter.png', dpi=150)

# Choose n_neighbors hyperparameter setting which lead to the highest
# accuracy results in cross-validation testing
maxidx = np.argmax(gscv['knn'].cv_results_['mean_test_score'])
maxval = param['knn']['n_neighbors'][maxidx]
# Train the K Nearest Neighbors classifier using all of the training data
knc = KNeighborsClassifier(n_neighbors=maxval)
knc.fit(X['train'], y['train'])
# Predict classes for final holdout test batch
ypred = knc.predict(X['test'])

def makeconfmat(ytru, yprd, lblsort, ttl):
    # Convenience function for preparing a confusion matrix plot
    #
    # Parameters:
    # 
    #     ytru:      List of truth labels
    #
    #     yprd:      List of predicted labels
    #
    #     lblsort:   Preferred sort order of labels (use this to group
    #                together classes where we might a priori expect to
    #                see a lot of confusion, because the images tend to
    #                look somewhat similar)
    #
    #     ttl:       Title string
    #
    # Returns:
    #
    #     disp:      A sckit-learn confusion matrix object

    # Calculate confusion matrix
    cm = confusion_matrix(ytru, yprd, labels=lblsort)
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(cm, display_labels=lblsort)
    # Configure to make it look a bit nicer, given the number of labels
    # that we have, and given the length of their description strings 
    disp.plot(xticks_rotation='vertical')
    disp.ax_.set_title(ttl)
    disp.figure_.set_size_inches(9,8)

    return(disp)

# Calculate confusion matrix for K Nearest Neighbors
disp = makeconfmat(y['test'], ypred, clssrt, 'K Nearest Neighbors')
disp.figure_.savefig('figures/knn_confusion.png', dpi=150)

# -------------------------------------------------------------------
# Part 5: Define a Neural Network Model and Training / Test Functions
# -------------------------------------------------------------------
    
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # Accept 28 X 28 input because that is the image size in pixels
            nn.Linear(28*28, 512),
            nn.ReLU(),
            # Simple hidden layer
            nn.Linear(512, 512),
            nn.ReLU(),
            # Deliver 10 channel output because that is the number of classes
            nn.Linear(512, 10)
        )

    # This is defined from basic tutorial examples that I've seen elsewhere
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(model, lrn_rate, device, loader):
    # Train the model through a single epoch (called again each new epoch)
    #
    # Parameters:
    #
    #     model:       A pytorch neural network model object
    #
    #     lrn_rate:    Learning rate; typically a floating point number
    #                  that's small compared to 1, e.g., like 1e-3
    #
    #     device:      String encoding a pytorch allowed device type; e.g.
    #                  like 'cpu' or 'cuda'
    #
    #     loader:      A pytorch data loader object (should be formulated 
    #                  specifically to load training data)
    #
    # Returns:
    #
    #     model:       A modified version of the input model object, with its
    #                  weights and biases adjusted to accurately predict
    #                  the labels of the input training data

    # Create a loss function object
    loss_fn = nn.CrossEntropyLoss()
    # Create an optimizer object
    optimizer = torch.optim.SGD(model.parameters(), lr=lrn_rate)

    # Put model in "training" state.  Based upon descriptions that I've seen
    # elsewhere, I think what's going on under the hood is that this allows
    # for gradients to be calculated and back-propagated through the layers
    # so that the weights and biases can be tuned.
    model.train()
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print evolving loss, so we can watch as it trends downward as the
        # training progresses
        if batch % 100 == 99:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>.6f}  [{current:>5d}/{len(loader.dataset):>5d}]")

    return(model)

def test(model, device, loader):
    # Test the model at the end of a training epoch
    #
    # Parameters: see train function above
    #
    # Returns:
    #
    #     avg_val_loss:     Cross entropy loss for whatever input data was
    #                       provided in the loader object
    #
    #     acc:              Prediction accuracy of the trained input model
    #                       object for whatever input data was provided in
    #                       the loader object
    #
    #     prdct:            List of predicted class values, with length
    #                       equal to the total number of input samples across
    #                       all sample batches within the loader object

    # Create a loss function object
    loss_fn = nn.CrossEntropyLoss()

    nbatch = len(loader)
    # Put model in "evaluation" state.  I think this disables gradient and
    # backpropagation calculations.
    model.eval()
    avg_val_loss, correct, prdct = 0, 0, list()
    # I'm not exactly sure why we need this, given the model has been placed
    # into the evaluation state, but I saw it done in some other tutorial
    # examples
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            # Two dimensional tensor; dimension 0: number of samples in this
            # loader batch; dimension 1: logit values for each class
            pred = model(X)
            # To compute average loss over entire data set, weight each
            # batch loss by the number of samples in the batch (the final
            # batch can sometimes be smaller than the others, if the number
            # of samples does not divide evenly by the batch size selected)
            avg_val_loss += loss_fn(pred, y).item() * len(y) / len(loader.dataset)
            # For each sample, this extracts the class index with the largest
            # logit value (which is effectively the predicted class, because
            # the largest logit will have the largest probability after
            # normalization via softmax)
            prmx = pred.argmax(dim=1)
            # Count total number of correct classifications
            correct += (prmx == y).type(torch.int).sum().item()
            # Keep track of predictions across all samples, which we need
            # later for calculating a confusion matrix
            for idx in range(len(prmx)):
                prdct.append(prmx[idx].item())
        acc = correct / len(loader.dataset)
        
        print(f"accuracy: {acc:>.4f}")

    return(avg_val_loss, acc, prdct)

# Training / validation function
def validate(config, device, nepoch, datadct):
    # Create a model and train / test it through multiple epochs
    #
    # Parameters:
    #
    #     config:     Dictionary of configuration parameters.  Expected
    #                 dictionary keys are 'lr' for learning rate, and
    #                 'batch_size' for data loader batch size
    #
    #     device:     String encoding for pytorch which device to use in 
    #                 performing calculations
    #
    #     nepoch:     Desired number of training epochs
    #
    #     datadct:    Dictionary with keys 'train' and 'test' defining
    #                 splits of training vs. test data
    #
    # Returns:
    #
    #     result:     Dictionary object containig loss vs. epoch,
    #                 accuracy vs. epoch, and final model class prediction
    #                 for all samples after the end of the last epoch,
    #                 for both training and test data 

    # Create a blank, new Neural Network model object to be trained
    model = NeuralNetwork().to(device)

    # Create data loaders for the training and test splits defined by the
    # input data dictionary object
    dtload, size = dict(), dict()
    for label in ['train', 'test']:
        dtload[label] = DataLoader(datadct[label], batch_size=config['batch_size'])

    # Create a blank results object
    blank = {'loss': list(), 'accuracy': list()}
    result = {'train': copy.deepcopy(blank), 'test': copy.deepcopy(blank)}
    for epoch in range(nepoch):
        # Train the model
        print(f"Begin epoch: {(epoch+1):>2d}")
        model = train(model, config['lr'], device, dtload['train'])
        print(f"End epoch: {(epoch+1):>2d}")
        # Test the model.  Test on both the training data as well as the
        # final holdout test data, so that we can try to interrogate whether
        # overfitting is taking place
        for splt in ['train', 'test']:
            avg_val_loss, acc, prdct = test(model, device, dtload[splt])
            # Record loss and accuracy at the end of each epoch
            result[splt]['loss'].append(avg_val_loss)
            result[splt]['accuracy'].append(acc)
            # Overwrite prediction at the end of each epoch, retaining only
            # the final value at the end of training
            result[splt]['predict'] = prdct

    # Final output status after all epochs are completed
    print(f"Finished Training  lr: {config['lr']:>.1e}  batch_size: {config['batch_size']:>3d}")
    print(f"loss: {avg_val_loss:>.6f}  accuracy: {acc:>.4f}")
    print(f"")

    return(result, model)

# -------------------------------------------------------------
# Part 6: Tune Neural Network Training Schedule Hyperparameters
# -------------------------------------------------------------

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Select a variety of different possible learning rates and batch sizes,
# and borrow functionality from scikit-learn to calculate a grid
prm = {'lr': [1e-4, 5e-3, 1e-3, 5e-2, 1e-2], 'batch_size': [10, 15, 20, 30, 60]}
pmg = ParameterGrid(prm)

# The Fashion_MNIST data set comes pre-divided into train / test splits, but
# for tuning hyperparameters, we want an intermediate validation split that
# allows us to do some testing while tuning, but without peeking at the final
# holdout test data yet.  So further subdivide the training data into a 
# smaller training split plus a validation split.
sss = StratifiedShuffleSplit(n_splits=1, test_size=1/6)
tridx, validx = next(sss.split(X['train'], y['train']))
valdata = {'train': Subset(data['train'], tridx),
    'test': Subset(data['train'], validx)}

# Accuracy results are returned vs. all training epochs, but for purposes
# of initial tuning, we only want the final accuracy at the end of all epochs
final_acc = []
for pm in pmg:
    # For each parameter combination in the grid, train for 3 epochs then stop
    result, dummy = validate(pm, device, 3, valdata)
    final_acc.append(result['test']['accuracy'][-1])

# Here we take a "greedy" approach to determining the optimal learning rate
# and batch size parameters: the combination which obtains the best accuracy
# after three epochs is judged "best", and that is what we'll use for a much
# longer training schedule of 50 epochs
idxmx = np.argmax(final_acc)
result, model = validate(pmg[idxmx], device, 50, data)

# --------------------------------------------
# Part 7: Plot Neural Network Training Results
# --------------------------------------------

# Plot cross entropy loss and accuracy for both the training and the test
# splits vs. epoch
fig = plt.figure(figsize=(10,6))
axlst = []
lgd = {'train': 'Training', 'test': 'Test'}
for plidx, scorekey, ttl in zip(range(2), ['loss', 'accuracy'], 
    ['Cross-Entropy Loss', 'Accuracy']):
    axlst.append(fig.add_subplot(1, 2, plidx+1))
    hdl = dict()
    for lbl in ('train', 'test'):
        ep = [ii+1 for ii in range(len(result[lbl][scorekey]))]
        hdl[lbl] = axlst[plidx].plot(ep, result[lbl][scorekey], label=lgd[lbl])
        axlst[plidx].set_title(ttl)
        axlst[plidx].set_xlabel('Epoch')
axlst[1].legend()
fig.savefig('figures/score_vs_epoch.png', dpi=150)

# Plot final confusion matrix after 50 epochs
ypred = [classes[ii] for ii in result['test']['predict']]
disp = makeconfmat(y['test'], ypred, clssrt,
    'Pytorch Single Hidden Layer Model')
disp.figure_.savefig('figures/pytorch_confusion.png', dpi=150)
