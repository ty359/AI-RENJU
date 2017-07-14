import rules
import os

chess = rules.chess_load('orz.ch')
#solve = rules.chess_solve(chess)
rules.chess_show(chess)

os.system('pause')
