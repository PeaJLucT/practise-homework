# Домашнее задание к уроку 1: Основы PyTorch

## Задание 1

Создайте файл `homework_tensors.py` и выполните следующие задачи:

### 1.1 Создание тензоров
```python
# Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
torch.rand(3,4)
# - Тензор размером 2x3x4, заполненный нулями
torch.zeros(2,3,4)
# - Тензор размером 5x5, заполненный единицами
torch.ones(5,5)
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
torch.arange(0,16)
torch.reshape(4,4)
```

### 1.2 Операции с тензорами
```python
# Дано: тензор A размером 3x4 и тензор B размером 4x3
A = torch.tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]])
B = torch.arange(1,13)
B = B.reshape(4,3)
# Выполните:
# - Транспонирование тензора A
A.transpose(1,0)
# - Матричное умножение A и B
C = A @ B
# - Поэлементное умножение A и транспонированного B
B.transpose_(0,1)
C = A * B
# - Вычислите сумму всех элементов тензора A
A_sum = A.sum()
```

### 1.3 Индексация и срезы
```python
# Создайте тензор размером 5x5x5
example = torch.arange(0,125).reshape(5,5,5)
print(f'A = {example}')
# Извлеките:
# - Первую строку
print(f'A[0]: \n{example[0]}')
# - Последний столбец
print(f'A[-1]: \n{example[-1]}')
# - Подматрицу размером 2x2 из центра тензора
print(f'Подматрица размером 2x2 из центра тензора: \n{example[2,2:4,2:4]')
# - Все элементы с четными индексами
print(f'Все элементы с четными индексами: \n{example[:,:,::2]}')
```

### 1.4 Работа с формами
```python
# Создайте тензор размером 24 элемента
example = torch.arange(1,25)
# Преобразуйте его в формы:
# - 2x12
example = example.reshape(2,12)
# - 3x8
example = example.reshape(3,8)
# - 4x6
example = example.reshape(4,6)
# - 2x3x4
example = example.reshape(2,3,4)
# - 2x2x2x3
example = example.reshape(2,2,2,3)
```

## Задание 2

Создайте файл `homework_autograd.py`:

### 2.1 Простые вычисления с градиентами
```python
# Создайте тензоры x, y, z с requires_grad=True
x = torch.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
y = torch.tensor([5.0,6.0,7.0,8.0],requires_grad=True)
z = torch.tensor([9.0,10.0,11.0,12.0],requires_grad=True)

# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f_x_y_z = x**2 + y**2 + z**2 + 2*x*y*z
f = f_x_y_z.sum()

# Найдите градиенты по всем переменным
f.backward()
x.grad
y.grad
z.grad

# Проверьте результат аналитически
Чтобы найти градиент какого либо тензора, надйем его частную производную относительно написанной функции
∂f / ∂x = 2*x + 2*y*z
∂f / ∂y = 2*y + 2*x*z
∂f / ∂z = 2*z + 2*x*y
Тогда для тензора X:
i=0: 2*1.0 + 2*5.0*9.0 = 2 + 90 = 92.0
i=1: 2*2.0 + 2*6.0*10.0 = 4 + 120 = 124.0
i=2: 2*3.0 + 2*7.0*11.0 = 6 + 154 = 160.0
i=3: 2*4.0 + 2*8.0*12.0 = 8 + 192 = 200.0
И получится x.grad = [ 92, 124, 160, 200], как и в случае выше
```

### 2.2 Градиент функции потерь
```python
# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x + b (линейная функция)
def mse(y_pred, y_true):
    """
    Mean Squared Error

    y_true: тензор с истинными значениями
    y_pred: тензор с предсказанными значениями
    """
    sqr_error = (y_pred - y_true)**2
    MSE = torch.mean(sqr_error)
    return MSE

# Найдите градиенты по w и b
w = torch.tensor(2.0, requires_grad=True) 
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([3.5, 5.0, 7.5])
y_pred = w * x + b

loss = mse(y_pred, y_true)
print(f'mse = {loss}')

loss.backward()
print(f'w.grad = {w.grad}')
print(f'b.grad = {b.grad}')
```

### 2.3 Цепное правило
```python
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
def func(x):
    sqr = x**2 + 1
    answer = torch.sin(sqr)
    return answer

x = torch.tensor([3.0, 10.0, 0.0], requires_grad=True)
answer = func(x)
print(f'func(x) = {answer}')

# Найдите градиент df/dx
answer = answer.sum()
answer.backward()
x_gr = x.grad   
print(f'x.grad = {x_gr}')

# Проверьте результат с помощью torch.autograd.grad
check_x_gr = torch.autograd.grad(answer, x)
print(f'autograd.grad(x) = {check_x[0]}')
# В первом случае получается 
[-5.0344, 17.8401,  0.0000]
и во втором
[-5.0344, 17.8401,  0.0000], поэтому найден градиент верно
```

## Задание 3
Создайте файл `homework_performance.py`:

### 3.1 Подготовка данных
```python
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
```

### 3.2 Функция измерения времени
```python
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
# Используйте time.time() для измерения на CPU
def time_CPU(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    return end - start
```

### 3.3 Сравнение операций
```python
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

# У меня получилось несколько результатов при разных данных:
1_gpu - 0.10270953369140624; 1_cpu - 16.84285283088684
2_gpu - 0.030246912002563478; 2_cpu - 0.0020024776458740234
3_gpu - 0.00891596794128418; 3_cpu - 0.0030014514923095703
4_gpu - 0.00037376000732183454; 4_cpu - 0.0
5_gpu - 0.09682738971710204; 5_cpu - 0.0020079612731933594
Операция                       | CPU(мс) | GPU(мс) | Ускорение
Матричное умножение            |  16.843 |   0.103 |  164.0x
Поэлементное сложение          |   0.002 |   0.030 |    0.1x
Поэлементное умножение         |   0.003 |   0.009 |    0.3x
Транспонирование               |   0.000 |   0.000 |    0.0x
Вычисление суммы всех элементов |   0.002 |   0.097 |    0.0x



1_gpu - 0.10707100677490235; 1_cpu - 8.304519891738892
2_gpu - 0.02082713508605957; 2_cpu - 0.0020055770874023438
3_gpu - 0.006554624080657959; 3_cpu - 0.0011508464813232422
4_gpu - 9.216000000014901e-05; 4_cpu - 0.0
5_gpu - 0.01033318442106247; 5_cpu - 0.0010035037994384766
Операция                       | CPU(мс) | GPU(мс) | Ускорение
Матричное умножение            |   8.305 |   0.107 |   77.6x
Поэлементное сложение          |   0.002 |   0.021 |    0.1x
Поэлементное умножение         |   0.001 |   0.007 |    0.2x
Транспонирование               |   0.000 |   0.000 |    0.0x
Вычисление суммы всех элементов |   0.001 |   0.010 |    0.1x
```

### 3.4 Анализ результатов
```python
# Проанализируйте результаты:
# - Какие операции получают наибольшее ускорение на GPU?
В которых важно запоминать множество операций/переменных для вычисления
# - Почему некоторые операции могут быть медленнее на GPU?
Потому что GPU тратит больше времени/памяти/ресурсов для своей инициализации и последующего использования, то есть время больше может занимать именно подготовка к вычислению а не само вычисление
# - Как размер матриц влияет на ускорение?
Чем больше матрица тем быстрее GPU тк ресурсы для инициализации одни и те же, а вычисление быстрее, поэтому итоговое время увеличивается не линейно а скорее логарифмически
# - Что происходит при передаче данных между CPU и GPU?
Данные выгружаются из RAM в VRAM которая находится в видеокарте для последующей работы
```   
