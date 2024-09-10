import torch
import numpy as np

class Net(torch.nn.Module):
    
    def __init__(self, dims: int):
        super(Net,self).__init__()
        self._layer1 = torch.nn.Linear(dims*dims, dims*dims*2)
        self._layer2 = torch.nn.Linear(dims*dims*2, dims*dims)
        self._layer3 = torch.nn.Linear(dims*dims, dims)
        self._layer4 = torch.nn.Linear(dims, 4)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        定义网络结构
        模型的输入为[? , (dims*dims)]的二维矩阵
        模型的输出为[? , 4]的二维矩阵
        
        Args:
        x: torch.Tensor 维度为 [batch_size, (dims*dims)]
        
        Return:
        向前传播后的计算结果Tensor
        '''
        h_relu = self._layer1(x).clamp(min=0)
        h2_relu = self._layer2(h_relu).clamp(min=0)
        h3_relu = self._layer3(h2_relu).clamp(min=0)
        y_pred = self._layer4(h3_relu)
        return y_pred
        
        

class RL:
    
    def __init__(self, 
                dims: int, 
                memory_size: int=512, 
                epsilon:float=0.9, 
                batch_size:int=256,
                copy_weight_step:int = 8,
                gamma:float= 0.2
                ):
        '''
        Args:
        dims : 维度
        memory_size : 存储回放数据的大小 (默认512)
        epsilon : 非随机探索流量的概率(默认0.9)
        batch_size : 每次训练的数据量
        copy_weight_step : 每隔N轮训练替换网络权重
        gamma : 看长期影响的系数
        '''
        
        self._dims = dims + 2
        self._state_dims = self._dims * self._dims
        self._epsilon = epsilon
        self._batch_size = batch_size
        self._copy_weight_step = copy_weight_step
        self._gamma = gamma
        
        self._loss_history = []
        self._learnned_step = 0
        self._memory_cnt = 0
        self._memory_size = memory_size
        # 数据集 [？, :dims * dims] 为当前环境
        # 数据集 [?, dims * dims] 为action
        # 数据集 [?, dims * dims + 1] 为reward
        # 数据集 [?, -(dims * dims):] 为下个环境
        self._memory = np.zeros((memory_size, self._state_dims * 2 + 2))
        
        self._train_net = Net(self._dims)
        self._target_net = Net(self._dims)
        
        self._criterion = torch.nn.MSELoss(reduction='sum')
        self._optimizer = torch.optim.SGD(self._train_net.parameters(), lr=1e-5)
    
    def store_to_memory(self, state, action, reward, next_state):
        state_array = state.reshape((1, self._state_dims))
        next_state_array = next_state.reshape((1, self._state_dims))
        action_array = np.array([action]).reshape((1,1))
        reward_array = np.array([reward]).reshape((1,1))

        transition = np.hstack((state_array, action_array, reward_array, next_state_array))
        # replace the old memory with new memory
        index = self._memory_cnt % self._memory_size
        self._memory[index, :] = transition
        self._memory_cnt += 1
        
    def get_dataset(self):
        '''
        获取指定batch_size大小的训练集
        Args:
        batch_size : 数据集行数
        '''
        if self._memory_cnt < self._memory_size:
            sample_index = np.random.choice(self._memory_cnt, size = self._batch_size)
        else:
            sample_index = np.random.choice(self._memory_size, size = self._batch_size)
        return self._memory[sample_index, :]
    
    def choose_action(self,state):
        input_array = state.reshape((1,self._state_dims))
        if np.random.uniform() < self._epsilon:
            # 执行向前传播函数,并获得每个动作的Q估计
            actions_value = self._train_net(torch.tensor(input_array, dtype=torch.float32))
            # 选取最大的Q估计 (用了detach后将不会计算梯度)
            action = np.argmax(actions_value.detach().numpy())
        else:
            action = np.random.randint(0, 4)
        return action

    def train(self):
        # 获取训练集
        data_set = self.get_dataset()

        # 获取train_net和target_net 分别对应state 和 next_state 的Qmax预测结果
        # 其中预测的值为 ? * 4 维度的矩阵
        train_pred = self._train_net(torch.tensor(data_set[:,:self._state_dims],dtype=torch.float32))
        target_pred = self._target_net(torch.tensor(data_set[:,-(self._state_dims):],dtype=torch.float32))
        next_pred = target_pred.detach().numpy()
        eval_pred = train_pred.detach().numpy()
        
        # 提取dataset中的action和reward
        batch_size_vector = np.arange(self._batch_size, dtype = np.int32)
        eval_act_index = data_set[:, self._state_dims].astype(int)
        reward = data_set[:, self._state_dims + 1]

        q_target = eval_pred.copy()
        q_target[batch_size_vector, eval_act_index] = reward + self._gamma * np.max(next_pred, axis=1)

        loss = self._criterion(torch.tensor(q_target,dtype=torch.float32) , train_pred)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._learnned_step += 1
        
        # 当训练了 replace_weight_step 轮后，将train_net的权重数据同步给target_net
        if self._learnned_step % self._copy_weight_step == 0:
            self._target_net.load_state_dict(self._train_net.state_dict())
            # print('Copy weight from train_net to target_net.')
            # print('Current loss:{0}'.format(loss))
            self._loss_history.append(loss.item())
            
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self._loss_history)), self._loss_history)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()