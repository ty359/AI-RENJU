import tensorflow as tf
import numpy as np
import rules

foo = np.zeros([rules.N, rules.N])
foo[0][0] = 1
foo[0][1] = 3
foo[0][2] = 5
foo[0][3] = 7
foo[0][4] = 9
foo[1][0] = -2
foo[1][1] = -4
foo[1][2] = -6
foo[1][3] = -8



def play(model=None, chess_name=None, model_name=None):
  assert model != None
  assert chess_name != None
  assert model_name != None

  model.sess.run(model.initer)

  latest_checkpoint = tf.train.latest_checkpoint(model_name)
  if latest_checkpoint:
    model.saver.restore(model.sess, latest_checkpoint)

  chess = np.zeros([rules.N, rules.N], dtype=np.int16)
  for i in range(1, rules.N * rules.N + 1):
    choise = model.choise.eval(feed_dict={model.chess:chess})
    mask = np.equal(chess, 0)
    pos = np.argmax((choise + 1) * mask)
    assert chess[pos // rules.N][pos % rules.N] == 0
    chess[pos // rules.N][pos % rules.N] = i
    if rules.chess_check(chess)[0] >= 5:
      break
    if i == rules.N * rules.N:
      chess = foo
      break
    chess = -chess
  rules.chess_show(chess)
  rules.chess_save(chess_name, chess)
