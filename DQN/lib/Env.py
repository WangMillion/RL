import numpy as np
import random

class Env(object):
    def __init__(self, dim:int):
        """
            dim : 华容道的维度
            [
                [-1,-1,-1,-1,-1]
                [-1, 1, 2, 3,-1]
                [-1, 4, 5, 6,-1]
                [-1, 7, 8, 0,-1]
                [-1,-1,-1,-1,-1]
            ]
        """
        # padding 1 ,维度需要+2
        self._real_dim = dim
        self._dim = dim + 2

        target_array = [x+1 for x in range(0, self._real_dim * self._real_dim)]
        target_array[-1] = 0
        self._target = np.array(target_array)
        self._target = self._target.reshape((self._real_dim, self._real_dim))
        self._target = np.pad(self._target, pad_width=1, mode='constant', constant_values= -1)

    def reset(self):
        number_array = [x for x in range(0,self._real_dim * self._real_dim)]
        random.shuffle(number_array)
        self._instance = np.array(number_array)
        self._instance = self._instance.reshape((self._real_dim,self._real_dim))
        self._instance = np.pad(self._instance, pad_width=1, mode='constant', constant_values= -1)
        self._cursor = np.where(self._instance == 0)
        self._cursor = [self._cursor[0][0],self._cursor[1][0]]

    def get_cur_state(self):
        return self._instance.copy()

    def step(self, action:int):
        """
            响应动作,并将当前环境变换为下一个状态
            Args:
                action
                action=0 时,表示0向上移动一格,实际表示上方的元素向下移动一格
                action=1 时,表示0向右移动一格,实际表示上方的元素向左移动一格
                action=2 时,表示0向下移动一格,实际表示上方的元素向上移动一格
                action=3 时,表示0向左移动一格,实际表示上方的元素向右移动一格
            Return:
                index 0: 当前最新的状态
                index 1: Reward -1 表示游戏成功; 0 表示游戏还在进行中; 1 表示赢得游戏
                index 2: 是否已经结束游戏 True:结束 False:进行中
        """
        
        if action == 0:
            action_array = [-1,0]
        elif action == 1:
            action_array = [0,1]
        elif action == 2:
            action_array = [1,0]
        elif action == 3:
            action_array = [0,-1]
        else:
            raise Exception('UnExcepted action.')

        target_cursor = [x+y for x,y in zip(self._cursor,action_array)]
        tmp = self._instance[target_cursor[0],target_cursor[1]]
        self._instance[target_cursor[0],target_cursor[1]] = 0
        self._instance[self._cursor[0],self._cursor[1]] = tmp
        self._cursor = target_cursor
        if self._check_bound(target_cursor):
            if (self._instance == self._target).all():
                return self._instance, 1,True
            else:
                return self._instance, 0,False
        else:
            return self._instance,-1,True
            
            
    def _check_bound(self, target_cursor) -> bool:
        '''
        当target_cursor 在 [1,1] - [self._real_dim,self._real_dim] 之间时为有效
            [
                [-1,-1,-1,-1,-1]
                [-1, 1, 2, 3,-1]
                [-1, 4, 5, 6,-1]
                [-1, 7, 8, 0,-1]
                [-1,-1,-1,-1,-1]
            ]
        '''
        return target_cursor[0] >= 1 \
            and target_cursor[0] <= self._real_dim \
            and target_cursor[1] >= 1 \
            and target_cursor[1] <= self._real_dim 
            
    def print_cur(self):
        print(self._instance)

    def print_target(self):
        print(self._target)

    def print_cursor(self):
        print(self._cursor[0],self._cursor[1])
