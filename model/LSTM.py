import torch
import torch.nn as nn

class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5):

        super(LSTMTextClassifier, self).__init__()
        
        # 임베딩 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM 
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # 양방향 LSTM
        )
        
        # 드롭아웃 
        self.dropout = nn.Dropout(dropout)
        
        # 전결합 
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # bidirectional을 사용하므로 hidden_dim * 2

    def forward(self, x):

        # 입력 데이터 임베딩
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        dropped_out = self.dropout(lstm_out)
        output = self.fc(dropped_out)
        
        return output
