import torch
import time
import tracemalloc
tracemalloc.start()

'''
# Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
A = torch.randint(1,100,[64,1024,1024])
print(f'A.size - {A.shape}')
# print(f'{A[:10,:10,:10]}')

# - 128 x 512 x 512
B = torch.randint(1,100,[128,512,512])
print(f'B.size - {B.shape}')

# - 256 x 256 x 256
C = torch.randint(1,100,[256,256,256])
print(f'C.size - {C.shape}')
# Заполните их случайными числами
# '''



# Создайте функцию для измерения времени выполнения операций
# Используйте torch.cuda.Event() для точного измерения на GPU
def time_GPU(func, *args):
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize() #Ожидаем что все остальные процессы окончены для точности подсчетов
        
        start_event.record()
        func(*args)
        end_event.record()
        torch.cuda.synchronize() #Ожидаем пока завершится наш процесс

        time_ms = start_event.elapsed_time(end_event)
        return time_ms / 1000.0
# if torch.cuda.is_available():
#     print('GPU в норме, имеет доступ')
#     A_gpu = torch.randn(512, 512, device='cuda')
#     B_gpu = torch.randn(512, 512, device='cuda')
#     gpu_time = time_GPU(torch.matmul, A_gpu, B_gpu)
#     print(f'gpu_time - {gpu_time}')

# Используйте time.time() для измерения на CPU
def time_CPU(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    return end - start
# протестим
# def sum(x,n):
#     summa = 1
#     for i in range(n):
#         summa += x    
#     return summa
# print(f'CPU time - {time_CPU(sum,123, 1000000)}')



# Сравните время выполнения следующих операций на CPU и CUDA:
A = torch.randint(1,100, [8,1024, 1024],dtype=torch.float16) # float16 для уменьшения используемой памяти
B = torch.randint(1,100, [8,1024, 1024],dtype=torch.float16)
print(f'sizeA - {A.shape}; sizeB - {B.shape}')
gpu_times = []
cpu_times = []
if torch.cuda.is_available():
    print('GPU в норме, имеет доступ')
    A_gpu = A.to('cuda')
    B_gpu = B.to('cuda') 

    # - Матричное умножение (torch.matmul)
    gpu_matmul = time_GPU(torch.matmul, A_gpu, B_gpu)
    gpu_times.append(gpu_matmul)
    cpu_matmul = time_CPU(torch.matmul, A, B)
    cpu_times.append(cpu_matmul)
    print(f'1_gpu - {gpu_matmul}; 1_cpu - {cpu_matmul}')

    # - Поэлементное сложение
    gpu_add = time_GPU(torch.add, A_gpu, B_gpu)
    gpu_times.append(gpu_add)
    cpu_add = time_CPU(torch.add, A, B)
    cpu_times.append(cpu_add)
    print(f'2_gpu - {gpu_add}; 2_cpu - {cpu_add}')  

    # - Поэлементное умножение
    gpu_mul = time_GPU(torch.multiply, A_gpu, B_gpu)
    gpu_times.append(gpu_mul)
    cpu_mul = time_CPU(torch.mul, A, B)
    cpu_times.append(cpu_mul)
    print(f'3_gpu - {gpu_mul}; 3_cpu - {cpu_mul}')

    # - Транспонирование
    gpu_tran_1 = time_GPU(torch.transpose, A_gpu, 0, 1)
    gpu_tran_2 = time_GPU(torch.transpose, B_gpu, 0, 1)
    gpu_tran = (gpu_tran_1 + gpu_tran_2) # Тк транспонировать можно 
    # по одиночке, то мы складываем время транспонирования 1 матрицы и второй по отдельности
    gpu_times.append(gpu_tran)
    cpu_tran1 = time_CPU(torch.transpose, A, 0, 1)
    cpu_tran2 = time_CPU(torch.transpose, B, 0, 1)
    cpu_tran = cpu_tran1 + cpu_tran2
    cpu_times.append(cpu_tran)
    print(f'4_gpu - {gpu_tran}; 4_cpu - {cpu_tran}')

    # - Вычисление суммы всех элементов
    gpu_sum1 = time_GPU(torch.sum, B_gpu)
    gpu_sum2 = time_GPU(torch.sum, A_gpu)
    gpu_sum = gpu_sum1 + gpu_sum2
    gpu_times.append(gpu_sum)
    cpu_sum1 = time_CPU(torch.sum, A)
    cpu_sum2 = time_CPU(torch.sum, B)
    cpu_sum = cpu_sum1 + cpu_sum2
    cpu_times.append(cpu_sum)
    print(f'5_gpu - {gpu_sum}; 5_cpu - {cpu_sum}')


print(f'Операция                       | CPU(мс) | GPU(мс) | Ускорение') 
operations = ['Матричное умножение',
              'Поэлементное сложение',
              'Поэлементное умножение',
              'Транспонирование',
              'Вычисление суммы всех элементов']

for i in range(len(operations)):
    cpu_t = cpu_times[i]
    gpu_t = gpu_times[i]
    op_name = operations[i]
    acceleration = cpu_t / gpu_t
    print(f"{op_name:<30} | {cpu_t:>7.3f} | {gpu_t:>7.3f} | {acceleration:>6.1f}x")

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде

# У меня получилось несколько раз при разных данных:
# 1_gpu - 0.10270953369140624; 1_cpu - 16.84285283088684
# 2_gpu - 0.030246912002563478; 2_cpu - 0.0020024776458740234
# 3_gpu - 0.00891596794128418; 3_cpu - 0.0030014514923095703
# 4_gpu - 0.00037376000732183454; 4_cpu - 0.0
# 5_gpu - 0.09682738971710204; 5_cpu - 0.0020079612731933594
# Операция                       | CPU(мс) | GPU(мс) | Ускорение
# Матричное умножение            |  16.843 |   0.103 |  164.0x
# Поэлементное сложение          |   0.002 |   0.030 |    0.1x
# Поэлементное умножение         |   0.003 |   0.009 |    0.3x
# Транспонирование               |   0.000 |   0.000 |    0.0x
# Вычисление суммы всех элементов |   0.002 |   0.097 |    0.0x

# 1_gpu - 0.10707100677490235; 1_cpu - 8.304519891738892
# 2_gpu - 0.02082713508605957; 2_cpu - 0.0020055770874023438
# 3_gpu - 0.006554624080657959; 3_cpu - 0.0011508464813232422
# 4_gpu - 9.216000000014901e-05; 4_cpu - 0.0
# 5_gpu - 0.01033318442106247; 5_cpu - 0.0010035037994384766
# Операция                       | CPU(мс) | GPU(мс) | Ускорение
# Матричное умножение            |   8.305 |   0.107 |   77.6x
# Поэлементное сложение          |   0.002 |   0.021 |    0.1x
# Поэлементное умножение         |   0.001 |   0.007 |    0.2x
# Транспонирование               |   0.000 |   0.000 |    0.0x
# Вычисление суммы всех элементов |   0.001 |   0.010 |    0.1x

current, peak = tracemalloc.get_traced_memory()
print(f"Используемая память: {current / 10**6} МБ; Пик использования: {peak / 10**6} МБ")