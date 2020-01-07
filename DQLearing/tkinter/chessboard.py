import numpy as np
import pandas as pd
import time
import tkinter as tk


def key(event):
    print("pressed", repr(event.char))


class ChessBoard(tk.Tk, object):
    def __init__(self, size, scale=35, train=True, win_cnt=5):
        super(ChessBoard, self).__init__()
        self.scale = scale
        if not train:
            self.next = False
        self.train = train
        self.win_cnt = win_cnt
        self.full = size * size
        self.cnt = 0
        self.winner = 0
        self.size = size * self.scale
        self.map = np.zeros((size + 1, size + 1), dtype=np.int16)
        self.objects = []
        self.canvas = tk.Canvas(self, bg='grey', height=self.size + self.scale,
                                width=self.size + self.scale)
        self.create_window()

    def create_window(self):
        self.title('五子棋')
        self.geometry(str(self.size + self.scale) + 'x' + str(self.size + self.scale))
        self.canvas.pack()
        for i in range(int(self.size / self.scale + 1)):
            self.canvas.create_line(self.scale, i * self.scale, self.size, i * self.scale)
            self.canvas.create_line(i * self.scale, self.scale, i * self.scale, self.size)
        if not self.train:
            self.canvas.bind("<Key>", key)
            self.canvas.bind("<Button-1>", self.callback)
        self.canvas.pack()

    def callback(self, event):
        x = int((event.x - self.scale / 2) / self.scale)
        y = int((event.y - self.scale / 2) / self.scale)
        print("clicked at", x, y)
        if x < 0 or x > self.size / self.scale or y < 0 or y > self.size / self.scale:
            return
        self.map[x][y] = 2
        self.render()
        self.game_check()
        print(self.winner)
        self.next = True

    def render(self):
        half = self.scale / 2
        for i in range(len(self.map)):
            for j in range(len(self.map)):
                if self.map[i][j] == 1:
                    self.objects.append(self.canvas.create_oval(self.scale + i * self.scale - half,
                                                                self.scale + j * self.scale - half,
                                                                self.scale + i * self.scale + half,
                                                                self.scale + j * self.scale + half, fill='black'))
                elif self.map[i][j] == 2:
                    self.objects.append(self.canvas.create_oval(self.scale + i * self.scale - half,
                                                                self.scale + j * self.scale - half,
                                                                self.scale + i * self.scale + half,
                                                                self.scale + j * self.scale + half, fill='white'))
        self.update()
        # time.sleep(0.5)

    def step(self, role, index):
        reward = -1
        state = self.convert()
        if index.max() > self.size or index.min() < 0 or self.map[index[0], index[1]] != 0:
            # print('index error, x: {}, y: {}'.format(index[0], index[1]))
            return state, reward, False
        self.map[index[0]][index[1]] = role
        self.game_check()
        if self.winner != 0:
            print(self.winner)
        if self.winner == 0:
            reward = 0
        elif self.winner != role:
            reward = -1
        else:
            reward = 1
        self.cnt += 1
        if self.cnt >= self.full:
            reward = -1
        return state, reward, True

    def reset(self):
        size = len(self.map)
        for i in range(size):
            for j in range(size):
                self.map[i][j] = 0
        for i in range(len(self.objects)):
            self.canvas.delete(self.objects[i])
        self.update()
        self.winner = 0
        self.cnt = 0

        return self.convert()

    def game_check(self):
        if self.count(1):
            self.winner = 1
        elif self.count(2):
            self.winner = 2

    def count(self, role):
        size = int(self.size / self.scale)
        for i in range(size):
            col_cnt = row_cnt = 0
            for j in range(size):
                if self.map[i][j] == role:
                    col_cnt += 1
                else:
                    col_cnt = 0
                if self.map[j][i] == role:
                    row_cnt += 1
                else:
                    row_cnt = 0
                if col_cnt == self.win_cnt or row_cnt == self.win_cnt:
                    return True

        for i in range(size):
            col_cnt = row_cnt = 0
            for j in range(size - i):
                if self.map[i + j][j] == role:
                    row_cnt += 1
                else:
                    row_cnt = 0
                if self.map[size - i - 1 - j][size - j - 1] == role:
                    col_cnt += 1
                else:
                    col_cnt = 0
                if col_cnt == self.win_cnt or row_cnt == self.win_cnt:
                    return True
            col_cnt = row_cnt = 0
            for j in range(i, size):
                if self.map[j - i][j] == role:
                    row_cnt += 1
                else:
                    row_cnt = 0
                if self.map[size - 1 - j + i][j] == role:
                    col_cnt += 1
                else:
                    col_cnt = 0
                if col_cnt == self.win_cnt or row_cnt == self.win_cnt:
                    return True
        return False

    def convert(self):
        res = ''
        size = len(self.map)
        for i in range(size):
            for j in range(size):
                res += str(self.map[i][j])

        return res

    def get_map(self):
        return self.map
