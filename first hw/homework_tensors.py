import torch

""" Задание 1
# Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
first_tensor = torch.rand(3,4)
print(first_tensor)
# - Тензор размером 2x3x4, заполненный нулями
second_tensor = torch.zeros(2,3,4)
print(second_tensor)
# - Тензор размером 5x5, заполненный единицами
third_tensor = torch.ones(5,5)
print(third_tensor)
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
four_tensor = torch.arange(0,16)
four_tensor = four_tensor.reshape(4,4)
print(four_tensor)
"""



"""
# Дано: тензор A размером 3x4 и тензор B размером 4x3
A = torch.tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]])
B = torch.arange(1,13)
B = B.reshape(4,3)
print(f'A:{A}')
print(f'B:{B}')
# Выполните:
# - Транспонирование тензора A
print(f'A.transpose:{A.transpose(1,0)}')
# - Матричное умножение A и B
C = A @ B
print(f'A @ B: {C}')
# - Поэлементное умножение A и транспонированного B
B.transpose_(0,1)
print(f'B.transpose = {B}')
C = A * B
print(f'A * B^T = {C}')
# - Вычислите сумму всех элементов тензора A
A_sum = A.sum()
print(f'A_sum = {A_sum}')
"""

"""
# Создайте тензор размером 5x5x5
example = torch.arange(0,125).reshape(5,5,5)
print(f'A = {example}')
# Извлеките:
# - Первую строку
print(f'A[0]: \n{example[0]}')
# - Последний столбец
print(f'A[-1]: \n{example[-1]}')
# - Подматрицу размером 2x2 из центра тензора
print(f'Подматрица размером 2x2 из центра тензора: \n{example[2,2:4,2:4]}')
# - Все элементы с четными индексами
print(f'Все элементы с четными индексами: \n{example[:,:,::2]}')
"""

"""
# Создайте тензор размером 24 элемента
example = torch.arange(1,25)
# Преобразуйте его в формы:
# - 2x12
example = example.reshape(2,12)
print(f'2x12: \n{example}')
# - 3x8
example = example.reshape(3,8)
print(f'3x8: \n{example}')
# - 4x6
example = example.reshape(4,6)
print(f'4x6: \n{example}')
# - 2x3x4
example = example.reshape(2,3,4)
print(f'2x3x4: \n{example}')
# - 2x2x2x3
example = example.reshape(2,2,2,3)
print(f'2x2x2x3: \n{example}')
"""
