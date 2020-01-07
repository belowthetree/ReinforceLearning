import QTable.五子棋.chessboard as cb
from QTable.走迷宫.SarsaLambda import SarsaLambdaTable
import numpy as np

board = cb.ChessBoard(3, train=False, win_cnt=3)
board.render()

epochs = 1000

actions = []
func = {}
size = len(board.map) - 1
print('size:', size)
for i in range(size):
    for j in range(size):
        actions.append(str([i, j]))
        func[str([i, j])] = [i, j]

# ai1 = AI.QLearning(actions=actions, map=board.get_map())
# ai1.load('q_table1.csv')
# observation = board.reset()
#
# while True:
#     board.reset()
#     while True:
#         print(str(board.convert()))
#         while True:
#             action = ai1.choose_action(str(board.convert()), test=True)
#             print(action)
#             observation_, reward, error = board.step(role=1, index=np.array(func[action]))
#             if error:
#                 break
#         if board.winner != 0:
#             break
#         while not board.next:
#             board.render()
#         board.next = False
#         if board.winner != 0:
#             break
#
ai1 = SarsaLambdaTable(actions=actions)
ai2 = SarsaLambdaTable(actions=actions)
# ai1.load('q_table1.csv')
# ai2.load('q_table2.csv')

for epoch in range(epochs):
    observation = board.reset()
    print('epoch: ', epoch)
    while True:
        board.render()
        action = ai1.choose_action(str(observation))
        while True:
            observation_, reward, error = board.step(role=1, index=np.array(func[action]))
            if error:
                break
            if reward == -1 and error:
                break
            # print(reward)
            action_ = ai1.choose_action(str(observation))
            ai1.learn(str(observation), action, reward, str(observation_), action_)
            action = ai1.choose_action(str(observation))
        action_ = ai1.choose_action(str(observation))
        ai1.learn(str(observation), action, reward, str(observation_), action_)

        observation = observation_
        if reward == -1 and error:
            print('winner: ai2')
            break
        elif reward == 1:
            print('winner: ai1')
            break

        action = ai2.choose_action(str(observation))
        while True:
            observation_, reward, error = board.step(role=2, index=np.array(func[action]))
            if error:
                break
            if reward == -1 and error:
                break
            # print(reward)
            action_ = ai2.choose_action(str(observation))
            ai2.learn(str(observation), action, reward, str(observation_), action_)
            action = ai2.choose_action(str(observation))
        action_ = ai2.choose_action(str(observation))
        ai2.learn(str(observation), action, reward, str(observation_), action_)

        observation = observation_
        if reward == -1 and error:
            print('winner: ai1')
            break
        elif reward == 1:
            print('winner: ai2')
            break

ai1.save('q_table1.csv')
ai2.save('q_table2.csv')


while True:
    board.reset()
    while True:
        print(str(board.convert()))
        while True:
            action = ai1.choose_action(str(board.convert()))
            print(action)
            observation_, reward, error = board.step(role=1, index=np.array(func[action]))
            if error:
                break
        if board.winner != 0 or board.cnt >= board.full:
            break
        while not board.next:
            board.render()
        board.next = False
        if board.winner != 0 or board.cnt >= board.full:
            break
