import torch

def evaluate_model(model, test_loader, criterion, device):
    '''
    Evaluate performance of model 
    
    param model: model to evaluate
    param test_loader: test data loader
    param criterion: loss function
    param device: device to use for computation
    
    return: dictionary containing test loss and test accuracy
    '''
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            h = model.module.init_hidden(inputs.size(0))
            h = tuple([e.data for e in h])
            
            output, h = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            test_loss += loss.item() * inputs.size(0)
            pred = torch.round(output.squeeze())
            total += labels.size(0)
            correct += pred.eq(labels.float().view_as(pred)).sum().item()
    
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = correct / total
    print(f'Test Loss: {test_loss:.6f}')
    print(f'Test Accuracy: {accuracy:.6f}')

    return {'Test Loss': test_loss, 'Test Accuracy': accuracy}