
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
    config_info = [
        # {
        # 'num_layers':3,
        # 'Dropout':True,
        # 'Batch_norm':False,
        # 'size':64},
        {
        'num_layers':3,
        'Dropout':True,
        'Batch_norm':False,
        'size':256},
        # {
        # 'num_layers':3,
        # 'Dropout':True,
        # 'Batch_norm':False,
        # 'size':1024},
        # {
        # 'num_layers':3,
        # 'Dropout':True,
        # 'Batch_norm':False,
        # 'size':2048},
    ]
    times = {}
    history_config = []
    i = 1
    for info in config_info:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader, test_loader = get_cifar_loaders(batch_size=64)
        
        name = f'{info["num_layers"]} l, {info["size"]} sizes, {"Dropout ," if info["Dropout"] else ""} {"Batch_norm" if info["Batch_norm"] else ""}'
        config = create_layer_config(info['num_layers'],start_size=info['size'], Dropout=info['Dropout'], Batchnorm=info["Batch_norm"])
        model = FullyConnectedModel(
            input_size=3072,
            num_classes=10,
            layers=config
        ).to(device)

        print(f"Model parameters: {count_parameters(model)}")

        t1 = time()
        history = train_model(model, train_loader, test_loader, epochs=30, device=str(device),name=name)
        plot_training_history(history,name) 
        t2 = time()
        times[i] = [info['num_layers'], info["size"], int(t2-t1)]
        print(f'Потрачено времени на {i} модель с {info["num_layers"]} слоями шириной [{info["size"]},..] - {(t2-t1)} с')
        history_config.append(config)
        i+=1

    print('----   ПОДВОДИМ ИТОГИ   ----')
    time_sum = 0
    for key, info in times.items():
        print(f'{key} модель с {info[0]} слоями шириной {info[1]} обучалась - {info[2]} секунд')
        print(f'Конфиги этой модели:{history_config[key-1]}')
        time_sum += info[2]
    print(f'Всего времени затрачено - {time_sum}')