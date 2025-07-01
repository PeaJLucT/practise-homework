import torch


"""
# Создайте тензоры x, y, z с requires_grad=True
x = torch.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
y = torch.tensor([5.0,6.0,7.0,8.0],requires_grad=True)
z = torch.tensor([9.0,10.0,11.0,12.0],requires_grad=True)
print(f'x.requires_grad = {x.requires_grad}')
# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f_x_y_z = x**2 + y**2 + z**2 + 2*x*y*z
f = f_x_y_z.sum()
print(f'f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z =\n{f}')
# Найдите градиенты по всем переменным
f.backward()
print(f'x.grad = {x.grad}')
print(f'y.grad = {y.grad}')
print(f'z.grad = {z.grad}')
# Проверьте результат аналитически
# Чтобы найти градиент какого либо тензора, надйем его частную производную относительно написанной функции
# ∂f / ∂x = 2*x + 2*y*z
# ∂f / ∂y = 2*y + 2*x*z
# ∂f / ∂z = 2*z + 2*x*y
# Тогда для тензора X:
# i=0: 2*1.0 + 2*5.0*9.0 = 2 + 90 = 92.0
# i=1: 2*2.0 + 2*6.0*10.0 = 4 + 120 = 124.0
# i=2: 2*3.0 + 2*7.0*11.0 = 6 + 154 = 160.0
# i=3: 2*4.0 + 2*8.0*12.0 = 8 + 192 = 200.0
# И получится x.grad = [ 92, 124, 160, 200], как и в случае выше
# """


'''
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
# Проверка
y_true = torch.tensor([3.5, 5.0, 7.5])
y_pred = torch.tensor([3.0, 5.0, 7.0])
print(mse(y_pred,y_true))

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
'''



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
print(f'autograd.grad(x) = {check_x_gr[0]}')
# В первом случае получается 
# [-5.0344, 17.8401,  0.0000]
# и во втором
# [-5.0344, 17.8401,  0.0000], поэтому найден градиент верно