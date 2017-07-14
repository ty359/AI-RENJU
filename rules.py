import numpy as np
import cv2

N = 15
M = 5

def chess_check(chess):
  assert chess.shape == (N, N)
  assert chess.dtype == np.int16
  x = np.zeros([N, N], dtype=np.int16)
  y = np.zeros([N, N], dtype=np.int16)
  xy = np.zeros([N, N], dtype=np.int16)
  yx = np.zeros([N, N], dtype=np.int16)
  for i in range(0, N):
    for j in range(0, N):
      if chess[i][j] > 0:
        if x[i][j] > 0:
          x[i][j] += 1
        else:
          x[i][j] = 1
        if y[i][j] > 0:
          y[i][j] += 1
        else:
          y[i][j] = 1
        if xy[i][j] > 0:
          xy[i][j] += 1
        else:
          xy[i][j] = 1
        if yx[i][j] > 0:
          yx[i][j] += 1
        else:
          yx[i][j] = 1

      elif chess[i][j] < 0:
        if x[i][j] < 0:
          x[i][j] -= 1
        else:
          x[i][j] = -1
        if y[i][j] < 0:
          y[i][j] -= 1
        else:
          y[i][j] = -1
        if xy[i][j] < 0:
          xy[i][j] -= 1
        else:
          xy[i][j] = -1
        if yx[i][j] < 0:
          yx[i][j] -= 1
        else:
          yx[i][j] = -1

      else:
        x[i][j] = y[i][j] = xy[i][j] = yx[i][j] = 0

      try:
        x[i][j + 1] = x[i][j]
      except IndexError:
        pass

      try:
        y[i + 1][j] = y[i][j]
      except IndexError:
        pass

      try:
        xy[i + 1][j + 1] = xy[i][j]
      except IndexError:
        pass

      try:
        yx[i + 1][j - 1] = yx[i][j]
      except IndexError:
        pass

  pos = max(np.max(x), np.max(y), np.max(xy), np.max(yx))
  neg = min(np.min(x), np.min(y), np.min(xy), np.min(yx))

  return (pos, -neg)



def chess_load(chess_name=None):
  assert chess_name != None
  with open(chess_name, 'rb') as f:
    return np.reshape(np.frombuffer(f.read(), dtype=np.int16), (N, N))


def chess_save(chess_name=None, chess=None):
  assert chess_name != None
  assert chess.dtype == np.int16
  assert chess.shape == (N, N)
  with open(chess_name, 'wb') as f:
    f.write(chess.tobytes())




def chess_show(chess=None):
  assert chess.shape == (N, N)

  chess = chess.copy()
  if np.count_nonzero(chess == 1) == 0:
    chess = -chess

  im = np.zeros([1024, 1024, 3])
  for i in range(N):
    for j in range(N):
      if chess[i][j] > 0:
        cv2.circle(im, (i * 50 + 100, j * 50 + 100), 20, (1, 0.2, 0.2))
      elif chess[i][j] < 0:
        cv2.circle(im, (i * 50 + 100, j * 50 + 100), 20, (0, 0.8, 0.8))
  cv2.imshow('foo', im)
  cv2.waitKey(10)



def chess_solve(chess=None):
  assert chess.shape == (N, N)
  assert chess.dtype == np.int16

  print(chess)

  if np.max(chess) < -np.min(chess):
    chess = -chess

  res = []

  n = np.max(chess)
  for i in range(n, 0, -1):
    assert i == np.max(chess)
    pos = np.argmax(chess)
    chess = chess.copy()
    chess[pos // N][pos % N] = 0
    choise = np.zeros((N, N), dtype=np.int16)
    choise[pos // N][pos % N] = 1
    res.append((chess, choise))
    chess = -chess

  return res

