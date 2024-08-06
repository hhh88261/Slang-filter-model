import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from models.model_definition import SlangNLPModel  
from data.dataset import SlangDataset 
from config.config import Config  

def main():
    config = Config.load('config/config.yaml')

    train_data = MyDataset(config['data']['train_file'])
    val_data = MyDataset(config['data']['val_file'])

    # 데이터 로더 설정
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'], shuffle=False)

    model = SlangNLPModel(
        vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    ).to(config['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # 훈련 루프
    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{config["training"]["num_epochs"]}, Loss: {avg_train_loss}')

        # 검증
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(config['device']), targets.to(config['device'])

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            print(f'Validation Loss: {avg_val_loss}, Accuracy: {accuracy}%')

    # 모델 저장
    torch.save(model.state_dict(), 'models/saved_models/my_nlp_model.pth')
    print('Model saved to models/saved_models/my_nlp_model.pth')

if __name__ == '__main__':
    main()

