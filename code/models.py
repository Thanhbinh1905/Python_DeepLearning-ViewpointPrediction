import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.isCUDA = torch.cuda.is_available()
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        if self.isCUDA:
            h_0 = Variable(torch.zeros(
                self.num_layers, x.size(0), self.hidden_size).cuda())
            
            c_0 = Variable(torch.zeros(
                self.num_layers, x.size(0), self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(
                self.num_layers, x.size(0), self.hidden_size))
            
            c_0 = Variable(torch.zeros(
                self.num_layers, x.size(0), self.hidden_size))

        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out