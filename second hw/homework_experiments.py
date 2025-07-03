from homework_model_modification import LinearRegressionManualWithReg, mse
from homework_model_modification import LinearRegressionManual
from homework_model_modification import accuracy, random_split,DataLoader,SoftmaxRegressionManual,EarlyStopper, cross_entropy_loss,precision_recall_f1,plot_confusion_matrix_text,roc_auc_score_manual,log_epoch
from homework_datasets import CustomCsvDataset
import pandas as pd
import matplotlib.pyplot as plt

### 3.1 Исследование гиперпараметров 
# Проведите эксперименты с различными:
# - Скоростями обучения (learning rate)
# - Размерами батчей
# - Оптимизаторами (SGD, Adam, RMSprop)
# Визуализируйте результаты в виде графиков или таблиц
# Настройка для вашего CustomCsvDataset

def train_and_evaluate(lr, batch_size, train_dataset, val_dataset, num_features, epochs=1000):
    """
    Функция для обучения и оценки модели с заданными гиперпараметрами
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LinearRegressionManual(in_features=num_features)
    stopper = EarlyStopper(patience=50) 

    for epoch in range(1, epochs + 1):
        train_loss_history = []

        for batch_X, batch_y in train_dataloader:
            y_pred = model(batch_X)
            loss = mse(y_pred, batch_y)
            train_loss_history.append(loss)
            
            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)
            model.step(lr)
                
        val_loss_history = []
        for batch_X_val, batch_y_val in val_dataloader:
            y_pred_val = model(batch_X_val)
            val_loss = mse(y_pred_val, batch_y_val)
            val_loss_history.append(val_loss)
        avg_val_loss = sum(val_loss_history) / len(val_loss_history)
        
        if stopper(avg_val_loss, model):
            break
            
    print(f"LR={lr}, Batch Size={batch_size}. Лучший Val MSE: {stopper.best_loss:.4f}")
    return stopper.best_loss

if __name__ == '__main__':
    csv_path = 'data/insurance.csv'
    target_col = 'charges'

    df = pd.read_csv(csv_path)
    num_columns = ['age', 'bmi', 'children']

    dataset = CustomCsvDataset(csv_path, target_col, num_columns, head=0)
    
    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print("\n--- LEARNING RATE ---")
    learning_rates_to_test = [0.1, 0.01, 0.001, 0.0001]
    lr_results = {}
    fixed_batch_size = 32 

    for lr in learning_rates_to_test:
        best_mse = train_and_evaluate(
            lr=lr, 
            batch_size=fixed_batch_size, 
            train_dataset=train_dataset, 
            val_dataset=val_dataset,
            num_features=len(num_columns)
        )
        lr_results[lr] = best_mse

    # Выводим результаты lr
    print("\n--- Результаты исследования Learning Rate ---")
    results_df_lr = pd.DataFrame(list(lr_results.items()), columns=['Learning Rate', 'Best Validation MSE'])
    print(results_df_lr)
    
    # Находим лучший LR
    best_lr = min(lr_results, key=lr_results.get)
    print(f"\nОптимальный Learning Rate: {best_lr}")

    # Исследование Batch Size
    print("\n--- BATCH SIZE ---")
    batch_sizes_to_test = [16, 32, 64, 128]
    bs_results = {}
    
    for bs in batch_sizes_to_test:
        best_mse = train_and_evaluate(
            lr=best_lr, # Используем лучший LR из предыдущего шага
            batch_size=bs,
            train_dataset=train_dataset, 
            val_dataset=val_dataset,
            num_features=len(num_columns)
        )
        bs_results[bs] = best_mse

    print("\n--- Результаты исследования Batch Size ---")
    results_df_bs = pd.DataFrame(list(bs_results.items()), columns=['Batch Size', 'Best Validation MSE'])
    print(results_df_bs)
    best_bs = min(bs_results, key=bs_results.get)
    print(f"\nОптимальный Batch Size: {best_bs}")

    plt.figure(figsize=(12, 5))

    # График для Learning Rate
    plt.subplot(1, 2, 1)
    plt.bar(results_df_lr['Learning Rate'].astype(str), results_df_lr['Best Validation MSE'])
    plt.title('Влияние Learning Rate на ошибку (MSE)')
    plt.xlabel('Learning Rate')
    plt.ylabel('Лучший MSE на валидации')
    plt.yscale('log') 

    # График для Batch Size
    plt.subplot(1, 2, 2)
    plt.bar(results_df_bs['Batch Size'].astype(str), results_df_bs['Best Validation MSE'])
    plt.title('Влияние Batch Size на ошибку (MSE)')
    plt.xlabel('Batch Size')
    plt.ylabel('Лучший MSE на валидации')

    plt.tight_layout()
    plt.show()


### 3.2 Feature Engineering
# Создайте новые признаки для улучшения модели:
# - Полиномиальные признаки
# - Взаимодействия между признаками
# - Статистические признаки (среднее, дисперсия)
# Сравните качество с базовой моделью