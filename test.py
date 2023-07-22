# -*- coding: utf-8 -*-
#@Time    : 2023/07/18 22:25:55
#@Author  : Tang
#@File    : test.py
#@Software: VScode


# 判断数字是否为质数
def is_prime(n):
    """
    判断数字是否为质数
    :param n: 数字
    :return: True or False
    """
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5)+1):
        if n % i == 0:
            return False
    return True

# 输入数字
num = int(input("请输入一个数字："))

# 调用函数
if is_prime(num):
    print("{}是质数".format(num))

else:
    print("{}不是质数".format(num))












