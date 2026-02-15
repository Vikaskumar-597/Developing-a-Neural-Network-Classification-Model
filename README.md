# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="654" height="631" alt="Screenshot 2026-02-10 113219" src="https://github.com/user-attachments/assets/cd020644-2a4e-4428-9f36-20b119b1ab49" />


## DESIGN STEPS

### STEP 1:
Data Collection and Understanding – Load the dataset, inspect features, and identify the target variable.
### STEP 2: 
Data Cleaning and Encoding – Handle missing values and convert categorical data and labels into numerical form.
### STEP 3: 
Feature Scaling and Data Splitting – Normalize features and split data into training and testing sets.
### STEP 4: 
Model Architecture Design – Define the neural network layers, neurons, and activation functions.
### STEP 5: 
Model Training and Optimization – Train the model using a loss function and optimizer through backpropagation.
### STEP 6: 
Model Evaluation and Prediction – Evaluate performance using metrics and make predictions on unseen data.



## PROGRAM

### Name: Vikaskumar M

### Register Number: 212224220122

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

        
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)


def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs , labels)
      loss.backward()
      optimizer.step()

  if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information
<img width="787" height="154" alt="image" src="https://github.com/user-attachments/assets/4908e422-f672-48ea-b997-fcd6efbe5c0e" />


### OUTPUT

## Confusion Matrix
<img width="539" height="455" alt="2" src="https://github.com/user-attachments/assets/c3154f43-1176-4a4c-8dcb-21559dd6b9ea" />.


<img width="201" height="114" alt="image" src="https://github.com/user-attachments/assets/fac23b82-0044-4b4a-b9a7-402287691652" />


## Classification Report
<img width="352" height="159" alt="image" src="https://github.com/user-attachments/assets/22ac0c2e-4d82-46c7-b846-cff1e9cba49a" />


### New Sample Data Prediction
<img width="230" height="61" alt="image" src="https://github.com/user-attachments/assets/3eda3c94-33dd-49a1-88f8-552ef2f165a2" />


## RESULT
Neural network classification model for the given dataset is successfully developed.
