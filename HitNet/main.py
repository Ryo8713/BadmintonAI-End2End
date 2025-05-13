from data_preparation import process
from hitnet import HitNet
from train import train, validate
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MATCHES_FOR_TRAIN = 27
MATCHES_FOR_TEST = 4
MATCHES_DIR = 'D:/AmeHibiki/Desktop/monotrack/matches'

def make_data(matches):
    X_lst, y_lst = [], []
    for match in matches:
        basedir = f'{MATCHES_DIR}/{match}'
        for video in os.listdir(f'{basedir}/rally_video/'):
            rally = video.split('.')[0]
            data = process(basedir, rally)
            if data is None:
                continue
            x, y = data
            X_lst.append(x)
            y_lst.append(y)
    X = np.vstack(X_lst)
    y = np.hstack(y_lst)
    return X, y

def get_logits_labels(model, data_loader, device=None):
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            logits_list.append(outputs.cpu())
            labels_list.append(labels.cpu())

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    return all_logits, all_labels

def find_best_threshold(labels, probs):
    best_thresh = 0.5
    best_f1 = 0.0

    thresholds = [x * 0.01 for x in range(100)]  # 測試不同的閾值

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average='macro')

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


if __name__ == '__main__':

    matches = list('match' + str(i) for i in range(1, MATCHES_FOR_TRAIN + 1))
    test_matches = list('test_match' + str(i) for i in range(1, MATCHES_FOR_TEST + 1))

    # check if data already exists
    if os.path.exists('X_train.npy') and os.path.exists('y_train.npy'):
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
    else:
        X_train, y_train = make_data(matches)
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)

    if os.path.exists('X_test.npy') and os.path.exists('y_test.npy'):
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
    else:
        X_test, y_test = make_data(test_matches)
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)

    # Print size
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    # 資料轉換
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_test, dtype=torch.long)

    # Dataset & Dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 4096
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # train
    num_consec = 12 
    model = HitNet(X_train.shape[1], num_consec=num_consec)
    train(model, train_loader, val_loader, num_epochs=100, learning_rate=3e-4, use_amp=True)

    # load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    # find best threshold
    logits, labels = get_logits_labels(model, val_loader)
    probs = sigmoid(logits).squeeze().cpu().numpy()
    best_thresh, best_f1 = find_best_threshold(labels.numpy(), probs)

    print(f"Best threshold = {best_thresh:.2f}, F1 = {best_f1:.4f}")

    # evaluate 
    all_preds, _ = validate(model, val_loader, threshold=best_thresh)
    print(classification_report(y_test, all_preds))
    print(confusion_matrix(y_test, all_preds, normalize='true'))

    # save model
    torch.save(model.state_dict(), 'hitnet.pth')