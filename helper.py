import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List

def plot_uniform_distribution(data: List, x_min = None, x_max = None):
    if x_min is None:
        x_min = min(data)
    if x_max is None:
        x_max = max(data)
    
    # make two plots next to each other
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # first plot
    
    x = np.arange(x_min, x_max + 1)
    y = [data.count(i) / len(data) for i in x]
    plt.ylim(0, 1)
    plt.plot(x, y, 'o')
    #plot line of uniform distribution
    plt.plot([x_min, x_max], [1 / (x_max - x_min + 1)] *
             2, color='red', linestyle
             ='dashed')
    plt.xlabel('Wartość')
    plt.ylabel('Prawdopodobieństwo')
    plt.title('Rozkład danych')
    
    # second plot
    plt.subplot(1, 2, 2)    
    plt.hist(data, bins = range(x_min, x_max + 2), rwidth=0.1, align='left')
    #plot line of uniform distribution
    plt.plot([x_min, x_max], [len(data) / (x_max - x_min + 1)] *
             2, color='red', linestyle
             ='dashed')
    
    plt.xlabel('Wartość')
    plt.ylabel('Wystąpienia')
    plt.title('Zliczenia danych')
    plt.show()
    
def plot_circle(data: List, radius = 1):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y)
    circle = plt.Circle((0, 0), radius, color='red', fill=False)
    plt.gca().add_artist(circle)
    plt.xlim(-radius*1.1, radius*1.1)
    plt.ylim(-radius*1.1, radius*1.1)
    plt.show()
    
def plot_function(data: List, function, x_min=-1, x_max=1):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    plt.scatter(x, y)
    X = np.linspace(x_min, x_max, 10000)
    Y = [function(i) for i in X]
    plt.plot(X,Y, color='red')
    plt.show()
    
def random_0_1():
    return np.random.rand()

def linear_congruential_generator(a, c, m, x0, n):
    x = x0
    for i in range(n):
        x = (a * x + c) % m
        yield x / m