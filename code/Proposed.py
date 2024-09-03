import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from torch.autograd import Variable
from function import *
from common import *
from sklearn.metrics import f1_score, precision_score, recall_score
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    args = args_parser()
    print(args)
    
    seq_length, delay_length = args.seq_length, 1
    
    cuda_dev = None
    print("---Has GPU: ", torch.cuda.is_available())
    if torch.cuda.is_available:
        cuda_dev = torch.device('cuda')
    
    
    x_train, y_train, x_test, y_test = {}, {}, {}, {}
    trainX, trainY, testX, testY = {}, {}, {}, {}
    
    # load train data
    for dataset in range(len(df)):
        x_train[dataset], y_train[dataset], x_test[dataset], y_test[dataset] = load_dataset_phi(datapath, dataset, dataset_test, seq_length, delay_length)
        trainX[dataset] = Variable(torch.Tensor(np.array(x_train[dataset])).cuda())
        trainY[dataset] = Variable(torch.Tensor(np.array(y_train[dataset])).cuda())
        testX[dataset] = Variable(torch.Tensor(np.array(x_test[dataset])).cuda())
        testY[dataset] = Variable(torch.Tensor(np.array(y_test[dataset])).cuda())
    
    # train settings
    num_epochs = args.num_epochs
    learning_rate = 0.001    
    input_size = 1
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_classes = 1

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(LSTM, self).__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            # Set initial hidden states (and cell states for LSTM)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(cuda_dev)   
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(cuda_dev)   
            
            out, _ = self.lstm(x, (h0,c0))  
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(cuda_dev)   
    model.train()
    criterion = nn.MSELoss().to(cuda_dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Lists to store evaluation metrics for each epoch
    epoch_f1_scores = []
    epoch_precision_scores = []
    epoch_recall_scores = []
    epoch_rmse_scores = []

    start = time.time()

    for run in range(args.num_run):
        for epoch in range(num_epochs):
            outputs = model(trainX[data])
            optimizer.zero_grad()
            loss = criterion(outputs, trainY[data])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                predicted_values = model(testX[data])
                model.train()

            predicted_values = predicted_values.cpu().numpy()
            true_values = testY[data].cpu().numpy()

            threshold = 0.5
            predicted_classes = (predicted_values > threshold).astype(int)
            true_classes = (true_values > threshold).astype(int)
            
            f1 = f1_score(true_classes, predicted_classes)
            precision = precision_score(true_classes, predicted_classes)
            recall = recall_score(true_classes, predicted_classes)
            rmse = np.sqrt(loss.item())

            # Append the metrics for this epoch
            epoch_f1_scores.append(f1)
            epoch_precision_scores.append(precision)
            epoch_recall_scores.append(recall)
            epoch_rmse_scores.append(rmse)

            # print(f"EPOCH [{epoch+1}/{num_epochs}] F1 score: {f1:.4f}  {precision:.4f} {recall:.4f} {rmse:.4f} LOSS: {loss.item():.4f} TIME: {time.time()-start}")

    # Calculate and plot CDF for F1-Score
    sorted_f1_scores = np.sort(epoch_f1_scores)
    cdf_f1 = np.arange(1, len(sorted_f1_scores) + 1) / len(sorted_f1_scores)

    # Calculate and plot CDF for Precision
    sorted_precision_scores = np.sort(epoch_precision_scores)
    cdf_precision = np.arange(1, len(sorted_precision_scores) + 1) / len(sorted_precision_scores)

    # Calculate and plot CDF for Recall
    sorted_recall_scores = np.sort(epoch_recall_scores)
    cdf_recall = np.arange(1, len(sorted_recall_scores) + 1) / len(sorted_recall_scores)

    # Calculate and plot CDF for RMSE
    sorted_rmse_scores = np.sort(epoch_rmse_scores)
    cdf_rmse = np.arange(1, len(sorted_rmse_scores) + 1) / len(sorted_rmse_scores)

    # Plot CDFs
    plt.plot(sorted_f1_scores, cdf_f1, marker='o', linestyle='-', color='b', label='F1-Score')
    plt.plot(sorted_precision_scores, cdf_precision, marker='o', linestyle='-', color='g', label='Precision')
    plt.plot(sorted_recall_scores, cdf_recall, marker='o', linestyle='-', color='r', label='Recall')
    plt.plot(sorted_rmse_scores, cdf_rmse, marker='o', linestyle='-', color='y', label='RMSE')

    plt.xlabel('Score')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function (CDF) of Evaluation Metrics')

    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
        'F1-Score': epoch_f1_scores,
        'Precision': epoch_precision_scores,
        'Recall': epoch_recall_scores,
        'RMSE': epoch_rmse_scores
    })

    # Save the DataFrame to an Excel file
    metrics_df.to_excel('123.xlsx', index=False)
    
    # Show the CDF plot
    plt.legend()
    plt.grid()
    plt.show()
print("done")