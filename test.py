# -*- coding: utf-8 -*-
# @Time    : 2023/7/12 21:36
# @Author  : Tang
# @File    : test.py
# @Software: PyCharm
def is_prime(number):
    if number < 2:
        return False

    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False

    return True

# 示例用法
num = int(input("请输入一个正整数: "))
if is_prime(num):
    print(num, "是质数")
else:
    print(num, "不是质数")
