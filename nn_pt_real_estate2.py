import csv
import torch
from sklearn.metrics import r2_score
import pandas as pd

# please use set PYTHONHASHSEED=0 or export PYTHONHASHSEED=0
# to ensure determinist hash before running !

def hash_9999(str):
    return hash(str)%9999

def import_df(csvfilename):
    hash_col = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']
    hash_col = ['MSZoning','LotArea','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition']
    X = pd.read_csv(csvfilename)
    for to_hash in hash_col:
        X[to_hash] = X[to_hash].apply(hash_9999)
    X['LotFrontage'] = X['LotFrontage'].apply(lambda x : x if pd.notnull(x) else 70)
    Y = X[['SalePrice']]
    X = X.drop('SalePrice',1)
    return X,Y

def plot_data(X,Y):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import norm, skew #for some statistics
    from scipy import stats
    color = sns.color_palette()
    sns.set_style('darkgrid')

    sns.distplot(Y['SalePrice'] , fit=norm);

    # Get the fitted parameters used by the function
    mu, sigma = norm.fit(Y['SalePrice'])
    #print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(Y['SalePrice'], plot=plt)
    plt.show()

def gradient_explore(x,y,w1,w2,w3,learning_rate,loop):
    prev_loss = 9e+32
    for t in range(loop):
        # Forward pass: compute predicted y using operations on Tensors; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        y_pred = x.mm(w1).clamp(min=0).mm(w2).clamp(min=0).mm(w3)

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the a scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()

        if t%500==0:
            print(t, loss.item()/1e+9)
            if loss > prev_loss:
                learning_rate /= 1.1
                print(learning_rate)
            prev_loss = loss

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call w1.grad and w2.grad will be Tensors holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        # An alternative way is to operate on weight.data and weight.grad.data.
        # Recall that tensor.data gives a tensor that shares the storage with
        # tensor, but doesn't track history.
        # You can also use torch.optim.SGD to achieve this.
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w3 -= learning_rate * w3.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()
            w3.grad.zero_()

    return w1,w2,w3,learning_rate,y_pred


train,Y = import_df('data/train.csv')
train.insert(0, 'Beta0', 1)

plot_data(train,Y)

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.tensor(train.values,device=device,dtype=dtype)
y = torch.tensor(Y.values,device=device,dtype=dtype)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(x.shape[1], 60, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(60, 30, device=device, dtype=dtype, requires_grad=True)
w3 = torch.randn(30, 1, device=device, dtype=dtype, requires_grad=True)


learning_rate = 1e-16
loop = 5000

w1,w2,w3,learning_rate,y_pred = gradient_explore(x,y,w1,w2,w3,learning_rate,loop)

end = False
while end == False:
    value = input("Continue? Y/N ?")
    if value == 'N' or value == 'n':
        for el in y_pred:
            print(int(el))
        end = True
        torch.save(w1, 'w1.torch')
        torch.save(w2, 'w2.torch')
        torch.save(w3, 'w3.torch')
        print("w1 w2 w3 saved")
    else:
        lrate_needed = input("learning_rate "+str(learning_rate)+" ?")
        if lrate_needed != "":
            learning_rate = float(lrate_needed)
        loop_needed = input("loop "+str(loop)+" ?")
        if loop_needed != "":
            loop = int(loop_needed)
        w1,w2,w3,learning_rate,y_pred = gradient_explore(x,y,w1,w2,w3,learning_rate,loop)
