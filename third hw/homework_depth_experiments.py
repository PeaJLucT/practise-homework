# Создайте и обучите модели с различным количеством слоев:
# - 1 слой (линейный классификатор)
# - 2 слоя (1 скрытый)
# - 3 слоя (2 скрытых)
# - 5 слоев (4 скрытых)
# - 7 слоев (6 скрытых)
# 
# Для каждого варианта:
# - Сравните точность на train и test
# - Визуализируйте кривые обучения
# - Проанализируйте время обучения
from time import time
from utils.model_utils import FullyConnectedModel, train_model, count_parameters
from utils.visualization_utils import plot_training_history, get_mnist_loaders, get_cifar_loaders
import torch


def create_layer_config(num_hidden_layers: int, start_size: int = 512, Dropout = False, Batchnorm = False):
    """
    Генерирует конфигурацию для скрытых слоев модели
    """
    config = []
    if num_hidden_layers == 0:
        return []

    current_size = start_size
    for i in range(num_hidden_layers):
        config.append({"type": "linear", "size": current_size}) 
        if Batchnorm:
            config.append({"type": "batch_norm"})
        config.append({"type": "relu"})
        if Dropout:
            config.append({"type": "dropout"})
        current_size //= 2 
    return config

if __name__ == '__main__':
    config_info = [{
        'num_layers':1,
        'Dropout':True,
        'Batch_norm':False},
        
        {
        'num_layers':2,
        'Dropout':True,
        'Batch_norm':False},
        {
        'num_layers':3,
        'Dropout':True,
        'Batch_norm':False},
        {
        'num_layers':4,
        'Dropout':True,
        'Batch_norm':False},
        {
        'num_layers':5,
        'Dropout':True,
        'Batch_norm':False},
        {
        'num_layers':6,
        'Dropout':True,
        'Batch_norm':False},
    ]
    # layers = [1,2,3,4,5,6]
    times = {}
    history_config = []
    i = 1
    for info in config_info:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader, test_loader = get_cifar_loaders(batch_size=32)
        
        name = f'{info["num_layers"]} layers with {"Dropout" if info["Dropout"] else "Batch_norm"}'
        config = create_layer_config(info['num_layers'],Dropout=info['Dropout'], Batchnorm=info["Batch_norm"])
        model = FullyConnectedModel(
            input_size=3072,
            num_classes=10,
            layers=config
        ).to(device)

        print(f"Model parameters: {count_parameters(model)}")

        t1 = time()
        history = train_model(model, train_loader, test_loader, epochs=9, device=str(device),name=name)
        plot_training_history(history,name) 
        t2 = time()
        times[i] = [info['num_layers'],int(t2-t1)]
        print(f'Потрачено времени на {i} модель с {info["num_layers"]} слоями - {(t2-t1)} с')
        history_config.append(config)
        i+=1

    print('----   ПОДВОДИМ ИТОГИ   ----')
    time_sum = 0
    for key, info in times.items():
        print(f'{key} модель с {info[0]} слоями обучалась - {info[1]} секунд')
        print(f'Конфиги этой модели:{history_config[key-1]}')
        time_sum += info[1]
    print(f'Всего времени затрачено - {time_sum}')