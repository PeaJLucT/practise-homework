# Домашнее задание к уроку 2: Линейная и логистическая регрессия

## Задание 1: Модификация существующих моделей 
### 1.1 Расширение линейной регрессии 
```python
# Модифицируйте существующую линейную регрессию:
# - Добавьте L1 и L2 регуляризацию
class LinearRegressionManualWithReg:
    def __init__(self, in_features, L1 = 0.0, L2 = 0.0):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.l1 = L1
        self.l2 = L2
    def backward(self, X, y, y_pred):
            n = X.shape[0]
            error = y_pred - y

            mse_dw = (X.T @ error) / n
            mse_db = error.mean(0)
            
            l2_dw = 2 * self.l2 * self.w    #Производная от L2, или же градиент
            l1_dw = self.l1 * torch.sign(self.w)  #Производная от L1

            self.dw = mse_dw + l2_dw + l1_dw
            self.db = mse_db
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
            self.best_model_state = model.parameters()
            print(f"Наименьший loss - {self.best_loss:.4f}")
        else:
            self.counter += 1
            print(f"До остановки - {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            return True
        return False
```

### 1.2 Расширение логистической регрессии 
```python
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

```
#### После выполнения проверки на рандомно созданном датасете в виде спирали получились такие данные:
![alt text](image.png)

## Задание 2: Работа с датасетами

### 2.1 Кастомный Dataset класс
```python
# Создайте кастомный класс датасета для работы с CSV файлами:
# - Загрузка данных из файла
# - Предобработка (нормализация, кодирование категорий)
# - Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.)
```

### 2.2 Эксперименты с различными датасетами
```python
# Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию
```

## Задание 3: Эксперименты и анализ 

### 3.1 Исследование гиперпараметров 
```python
# Проведите эксперименты с различными:
# - Скоростями обучения (learning rate)
# - Размерами батчей
# - Оптимизаторами (SGD, Adam, RMSprop)
# Визуализируйте результаты в виде графиков или таблиц
```

### 3.2 Feature Engineering
```python
# Создайте новые признаки для улучшения модели:
# - Полиномиальные признаки
# - Взаимодействия между признаками
# - Статистические признаки (среднее, дисперсия)
# Сравните качество с базовой моделью
```

## Дополнительные требования

1. **Код должен быть модульным** - разделите на функции и классы
2. **Документация** - добавьте подробные комментарии и docstring
3. **Визуализация** - создайте графики для анализа результатов
4. **Тестирование** - добавьте unit-тесты для критических функций
5. **Логирование** - используйте logging для отслеживания процесса обучения