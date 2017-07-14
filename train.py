import tensorflow as tf
import numpy as np
import random
import rules

def train(model=None, chess_name=None, model_name=None, times=5, back=1):
  assert model != None
  assert chess_name != None
  assert model_name != None

  model.sess.run(model.initer)

  latest_checkpoint = tf.train.latest_checkpoint(model_name)
  if latest_checkpoint:
    model.saver.restore(model.sess, latest_checkpoint)

  chess = rules.chess_load(chess_name)
  solve = rules.chess_solve(chess)

  for i in range(times):
    k = 0
    for j in range(0, len(solve), 2):
      ijchess, ijchoise = solve[j]
      model.opt.run(feed_dict={model.chess:ijchess, model.tar_choise:ijchoise})
      if random.randint(0, 1) == 0:
        k += 1
      if k == back:
        break


  model.saver.save(model.sess, model_name + '/' + model_name)
