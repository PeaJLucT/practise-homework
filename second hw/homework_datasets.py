### 2.1 Кастомный Dataset класс
# Создайте кастомный класс датасета для работы с CSV файлами:
# - Предобработка (нормализация, кодирование категорий)
# - Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.)
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class CustomCsvDataset(Dataset):
    """
    Args:
        csv_path (str): Путь к CSV файлу
        target_col (str): Название колонки с целевой переменной(y)
        numerical_cols (list): Список названий числовых колонок
        categorical_cols (list): Список названий категориальных колонок
    """
    def __init__(self, csv_path, target_col, numerical_cols, categorical_cols = None, head = None):
# - Загрузка данных из файла
        df = pd.read_csv(csv_path, header=head)
        if not df.empty:
            print('Файл прочитан успешно')
        else:
            print('Файл не прочитан')
        if numerical_cols:    
            for col in numerical_cols:
                df[col] = df[col].fillna(df[col].mean())
        if categorical_cols:
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Разделение на признаки (X) и цель (y)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        processed_features = []
        if numerical_cols:
            scaler = StandardScaler()
            # Нормализуем и добавляем в список обработанных признаков
            num_features = pd.DataFrame(scaler.fit_transform(X[numerical_cols]), columns=numerical_cols)
            processed_features.append(num_features)
            
        # Обработка категориальных признаков 
        if categorical_cols:
            cat_features = pd.get_dummies(X[categorical_cols], drop_first=True)
            processed_features.append(cat_features)

        # Объединяем обработанные числовые и категориальные признаки
        if processed_features:
            X_processed = pd.concat(processed_features, axis=1)
        else:
            print('Error')

        self.X = torch.tensor(X_processed.values, dtype=torch.float32)
        self.y = torch.tensor((y.values), dtype=torch.float32).unsqueeze(1)
        # для классификации
        # self.y = torch.tensor((y.values - 1), dtype=torch.long)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


### 2.2 Эксперименты с различными датасетами
# Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию
from homework_model_modification import accuracy, random_split,DataLoader,SoftmaxRegressionManual,EarlyStopper, cross_entropy_loss,precision_recall_f1,plot_confusion_matrix_text,roc_auc_score_manual,log_epoch


# Обучение многоклассовой классификации на датаесете вин
'''
if __name__ == '__main__':
    dataset_cst = CustomCsvDataset("data/wine.csv",0,[1,2,3,4,5,6,7,8,9,10,11,12,13])
    NUM_FEATURES = 13
    NUM_CLASSES = 3

    val_split = 0.2
    val_size = int(len(dataset_cst) * val_split)
    train_size = len(dataset_cst) - val_size
    train_dataset, val_dataset = random_split(dataset_cst, [train_size, val_size])

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
    print(f'accuracy = {accuracy(final_y_pred_probs, final_y_true)}')
'''


# Обучение линейной регрессии на датасете цен американских домов
from homework_model_modification import LinearRegressionManualWithReg, mse
from homework_model_modification import LinearRegressionManual


if __name__ == '__main__':
    """ Очистили датасет от нечисленных метрик
    def clean_housing_data(csv_path):
        df = pd.read_csv(csv_path)
        
        # Удаляем ненужные колонки
        cols_to_drop = ['Address', 'MLS ID', 'Listing Agent', 'Status', 'Listing URL', 'Zipcode']
        df.drop(columns=cols_to_drop, inplace=True)
        
        # Очищаем 
        # Цена (целевая переменная)
        df['Price'] = df['Price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
        # Спальни
        df['Bedrooms'] = df['Bedrooms'].str.replace(' bds', '').astype(int)
        # Ванные
        df['Bathrooms'] = df['Bathrooms'].str.replace(' ba', '').astype(int)
        # Площадь дома
        df['Area (Sqft)'] = df['Area (Sqft)'].str.replace(' sqft', '').astype(int)
        # Площадь участка
        df['Lot Size'] = df['Lot Size'].str.replace(' sqft', '').astype(int)
        # Переименуем колонку для удобства
        df.rename(columns={'Area (Sqft)': 'Area_Sqft'}, inplace=True)

        print("Датасет после очистки:")
        print(df.head())
        print("\nТипы данных после очистки:")
        print(df.info())
        
        return df
    HOUSING_CSV_PATH = 'data/us_house_Sales_data.csv'
    cleaned_df = clean_housing_data(HOUSING_CSV_PATH)
    CLEANED_HOUSING_PATH = 'data/us_house_sales_cleaned.csv'
    cleaned_df.to_csv(CLEANED_HOUSING_PATH, index=False)
"""
    
    num_columns = [
        'Bedrooms', 
        'Bathrooms', 
        'Area_Sqft', 
        'Lot Size', 
        'Year Built', 
        'Days on Market'
    ]
    dataset_cst = CustomCsvDataset("data/us_house_sales_cleaned.csv",'Price',num_columns, head=0)
    
    val_split = 0.2
    val_size = int(len(dataset_cst) * val_split)
    train_size = len(dataset_cst) - val_size
    train_dataset, val_dataset = random_split(dataset_cst, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False) 

    print(f'train_size: {len(train_dataset)} | val_size: {len(val_dataset)}')
    print(f'train_batches: {len(train_dataloader)} | val_batches: {len(val_dataloader)}')
    
    # Обучаем модель
    
    model = LinearRegressionManual(in_features=len(num_columns))
    lr = 0.1
    epochs = 5000
    stopper = EarlyStopper(10000)

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

    if stopper.best_model_state:
        model.w = stopper.best_model_state['w']
        model.b = stopper.best_model_state['b']
        print(f'Лучший Val Loss: {stopper.best_loss:.4f}')