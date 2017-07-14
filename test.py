import numpy as np
import rules
import model
import play
import train
import random

m = model.model()

for i in range(100000):
    play.play(m, 'orz.ch', 'orzzz')
    train.train(m, 'orz.ch', 'orzzz', back=1 + int(random.random() * i / 2500))
    print(i)
