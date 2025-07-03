import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import copy

def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')

def make_classification_data(n=200, n_features=2, n_classes=3):
        # Создание спирали
        X = torch.zeros(n * n_classes, n_features)
        y = torch.zeros(n * n_classes, dtype=torch.long)
        for i in range(n_classes):
            ix = range(n * i, n * (i + 1))
            r = torch.linspace(0.0, 1, n)
            t = torch.linspace(i * 4, (i + 1) * 4, n) + torch.randn(n) * 0.2
            x_coords = r * torch.sin(t)
            y_coords = r * torch.cos(t)
            X[ix] = torch.stack([x_coords, y_coords], dim=1)
            y[ix] = i
        return X, y

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()

def accuracy(y_pred, y_true):
    y_pred_bin = (y_pred > 0.5).float()
    return (y_pred_bin == y_true).float().mean().item()

def log_epoch(epoch, epochs, train_loss, val_loss=None, metrics=None):
    log_str = f'Epoch: {epoch}/{epochs} | Train Loss: {train_loss:.4f}'
    if val_loss:
        log_str += f' | Val Loss: {val_loss:.4f}'
    if metrics:
        for name, value in metrics.items():
            log_str += f' | {name}: {value:.4f}'
    print(log_str)

def cross_entropy_loss(y_pred, y_true, num_classes):
    n = y_pred.shape[0]
    y_one_hot = F.one_hot(y_true, num_classes=num_classes).float()
    log_probs = torch.log(y_pred + 1e-9)
    loss = -torch.sum(y_one_hot * log_probs) / n
    return loss

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return (self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LinearRegressionManual:
    def __init__(self, in_features):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)

    def __call__(self, X):
        return X @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        error = y_pred - y
        self.dw = (X.T @ error) / n
        self.db = error.mean(0)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save(self, path):
        torch.save({'w': self.w, 'b': self.b}, path)

    def load(self, path):
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']

### 1.1 Расширение линейной регрессии 
# Модифицируйте существующую линейную регрессию:
# - Добавьте L1 и L2 регуляризацию
class LinearRegressionManualWithReg:
    def __init__(self, in_features, L1 = 0.0, L2 = 0.0):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.l1 = L1
        self.l2 = L2

    def __call__(self, X):
        return X @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        error = y_pred - y

        mse_dw = (X.T @ error) / n
        mse_db = error.mean(0)
        
        l2_dw = 2 * self.l2 * self.w #Производная от L2, или же градиент
        l1_dw = self.l1 * torch.sign(self.w)  #Производная от L1

        self.dw = mse_dw + l2_dw + l1_dw
        self.db = mse_db

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save(self, path):
        torch.save({'w': self.w, 'b': self.b}, path)

    def load(self, path):
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']

# - Добавьте early stopping
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        """
        patience (int): Сколько эпох ждать улучшения, перед остановкой
        min_delta (float): Минимальное изменение, которое считается улучшением
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """
        val_loss (float): Ошибка за данную эпоху
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Сохраняем лучшую модель
            param = model.parameters()
            self.best_model_state = {
                'w': param[0].clone().detach(),
                'b': param[1].clone().detach()
            }
            # print(f"Наименьший loss - {self.best_loss:.4f}")
        else:
            self.counter += 1
            # print(f"До остановки - {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            return True
        return False
    
if __name__ == '__main__':
    # Генерируем данные
    X, y = make_regression_data(n=200)

    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)

    #Разделим датасет на тренировочный и валидационный
    val_split = 0.2  # 20% датасета будут валидационными
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset,[train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    print(f'train_size: {len(train_dataset)}| val_size: {len(val_dataset)}')
    print(f'train_batches: {len(train_dataloader)}| val_batches: {len(val_dataloader)}')
    
    # Обучаем модель
    model = LinearRegressionManualWithReg(in_features=1, L1=0.05, L2=0.05)
    lr = 0.1
    epochs = 100
    stopper = EarlyStopper(20)

    for epoch in range(1, epochs + 1):
        train_loss_history = []
        
        for i, (batch_X, batch_y) in enumerate(train_dataloader):
            y_pred = model(batch_X)
            loss = mse(y_pred, batch_y)
            train_loss_history.append(loss)
            
            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)
            model.step(lr)
        
        avg_train_loss = sum(train_loss_history) / len(train_loss_history)
        
        val_loss_history = []
        for batch_X_val, batch_y_val in val_dataloader:
            y_pred_val = model(batch_X_val)
            val_loss = mse(y_pred_val, batch_y_val)
            val_loss_history.append(val_loss)
        avg_val_loss = sum(val_loss_history) / len(val_loss_history)
        
        if epoch % 10 == 0:
            log_epoch(epoch,epochs, train_loss=avg_train_loss, val_loss=avg_val_loss)

        if stopper(avg_val_loss,model):
            break

    if stopper.best_model_state:
        model.w = stopper.best_model_state['w']
        model.b = stopper.best_model_state['b']

    print(f'Лучшие метрики для модели - {stopper.best_model_state} при потере - {stopper.best_loss}')

    model.save('linreg_manual.pth') 



def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class LogisticRegressionManual:
    def __init__(self, in_features):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)

    def __call__(self, X):
        return sigmoid(X @ self.w + self.b)

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        error = y_pred - y
        self.dw = (X.T @ error) / n
        self.db = error.mean(0)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save(self, path):
        torch.save({'w': self.w, 'b': self.b}, path)

    def load(self, path):
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']

### 1.2 Расширение логистической регрессии 
# Модифицируйте существующую логистическую регрессию:
# - Добавьте поддержку многоклассовой классификации
import torch.nn.functional as F

class SoftmaxRegressionManual:
    def __init__(self, in_features, num_classes):
        self.w = torch.randn(in_features, num_classes, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, num_classes, dtype=torch.float32, requires_grad=False)
        self.num_classes = num_classes

    def __call__(self, X):
        logits = X @ self.w + self.b
        return softmax(logits)

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).to(torch.float32)
        error = y_pred - y_one_hot
        self.dw = (X.T @ error) / n
        self.db = error.mean(0)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save(self, path):
        torch.save({'w': self.w, 'b': self.b}, path)

    def load(self, path):
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']

def softmax(x):  # Нормализуем тензор чтобы в сумме каждая строчка давала 1
    exps = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
    return exps / torch.sum(exps, dim=1, keepdim=True)

# - Реализуйте метрики: precision, recall, F1-score, ROC-AUC
def precision_recall_f1(y_true, y_pred, num_classes):

    precisions = []
    recalls = []
    f1s = []
    epsilon = 1e-7 

    for i in range(num_classes):
        #  i - "позитивный"
        tp = ((y_pred == i) & (y_true == i)).sum().item()
        fp = ((y_pred == i) & (y_true != i)).sum().item()
        fn = ((y_pred != i) & (y_true == i)).sum().item()

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    precision = sum(precisions) / num_classes
    recall = sum(recalls) / num_classes
    f1 = sum(f1s) / num_classes
    return precision, recall, f1

def roc_auc_score_manual(y_true, y_scores, num_classes):
    aucs = []
    for i in range(num_classes):
        y_true_binary = (y_true == i).float()
        # Берем вероятности для текущего класса
        y_scores_binary = y_scores[:, i]
        
        # Отделяем оценки для позитивных и негативных классов
        pos_mask = y_true_binary == 1
        neg_mask = y_true_binary == 0
        
        pos_scores = y_scores_binary[pos_mask]
        neg_scores = y_scores_binary[neg_mask]
        #Если нет данных класса, то переходим к следующему классу
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue

        num_correct_pairs = (pos_scores.unsqueeze(1) > neg_scores.unsqueeze(0)).sum()
        num_tied_pairs = (pos_scores.unsqueeze(1) == neg_scores.unsqueeze(0)).sum()
        total_pairs = pos_scores.numel() * neg_scores.numel()
        
        #Считаем
        auc = (num_correct_pairs + 0.5 * num_tied_pairs) / total_pairs
        aucs.append(auc)
    if not aucs:
        return 0.0
    macro_auc = torch.stack(aucs).mean()
    
    return macro_auc.item()

# - Добавьте визуализацию confusion matrix

def plot_confusion_matrix_text(y_true, y_pred, num_classes):

    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1
    
    print("\n" + "="*40)
    print("Confusion Matrix")
    print("="*40)

    header = f"{'True // Pred':<10}|"
    for i in range(num_classes):
        header += f"{'Класс ' + str(i):^10}|"
    print(header)
    print("-" * len(header))
    
    for i in range(num_classes):
        row_str = f"{'Класс ' + str(i):<10}|"
        for j in range(num_classes):
            count = conf_matrix[i, j].item()
            row_str += f"{str(count):^10}|"
        print(row_str)
    print("="*40 + "\n")



if __name__ == '__main__':

    NUM_FEATURES = 2
    NUM_CLASSES = 3
    X, y = make_classification_data(n=400, n_features=NUM_FEATURES, n_classes=NUM_CLASSES)

    dataset = ClassificationDataset(X, y)
    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False) 

    print(f'train_size: {len(train_dataset)} | val_size: {len(val_dataset)}')
    print(f'train_batches: {len(train_dataloader)} | val_batches: {len(val_dataloader)}')
    
    model = SoftmaxRegressionManual(in_features=NUM_FEATURES, num_classes=NUM_CLASSES)
    lr = 0.1
    epochs = 1000
    stopper = EarlyStopper(patience=50)

    for epoch in range(1, epochs + 1):
        train_loss_history = []
        for batch_X, batch_y in train_dataloader:
            y_pred = model(batch_X)
            loss = cross_entropy_loss(y_pred, batch_y, NUM_CLASSES)
            train_loss_history.append(loss.item())
            
            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)
            model.step(lr)
        
        avg_train_loss = sum(train_loss_history) / len(train_loss_history)
        
        val_loss_history = []
        all_y_true = []
        all_y_pred_probs = []

        for batch_X_val, batch_y_val in val_dataloader:
            y_pred_probs_val = model(batch_X_val)
            val_loss = cross_entropy_loss(y_pred_probs_val, batch_y_val, NUM_CLASSES)
            val_loss_history.append(val_loss.item())
            
            all_y_true.append(batch_y_val)
            all_y_pred_probs.append(y_pred_probs_val)

        avg_val_loss = sum(val_loss_history) / len(val_loss_history)
        
        all_y_true = torch.cat(all_y_true)
        all_y_pred_probs = torch.cat(all_y_pred_probs)
        all_y_pred_labels = torch.argmax(all_y_pred_probs, dim=1)
        
        # Считаем метрики
        precision, recall, f1 = precision_recall_f1(all_y_true, all_y_pred_labels, NUM_CLASSES)
        roc_auc = roc_auc_score_manual(all_y_true, all_y_pred_probs, NUM_CLASSES)
        
        metrics = {'Precision': precision, 'Recall': recall, 'F1-score': f1, 'ROC-AUC': roc_auc}
        
        if epoch % 10 == 0 or epoch == 1:
            log_epoch(epoch, epochs=epochs, train_loss=avg_train_loss, val_loss=avg_val_loss, metrics=metrics)

        if stopper(avg_val_loss, model):
            break

    # Восстанавливаем лучшую модель
    if stopper.best_model_state:
        model.w = stopper.best_model_state['w']
        model.b = stopper.best_model_state['b']
        print(f'Лучший Val Loss: {stopper.best_loss:.4f}')

    model.save('softmax_reg_manual.pth') 
    
    # Пересчитываем предсказания с лучшей моделью
    final_y_true = []
    final_y_pred_probs = []
    for batch_X_val, batch_y_val in val_dataloader:
        final_y_pred_probs.append(model(batch_X_val))
        final_y_true.append(batch_y_val)

    final_y_true = torch.cat(final_y_true)
    final_y_pred_probs = torch.cat(final_y_pred_probs)
    final_y_pred_labels = torch.argmax(final_y_pred_probs, dim=1)

    # Выводим
    precision, recall, f1 = precision_recall_f1(final_y_true, final_y_pred_labels, NUM_CLASSES)
    roc_auc = roc_auc_score_manual(final_y_true, final_y_pred_probs, NUM_CLASSES)
    
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    plot_confusion_matrix_text(final_y_true, final_y_pred_labels, num_classes=NUM_CLASSES)