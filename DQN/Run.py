from lib.Env import Env
from lib.RL import RL


if __name__ == '__main__':
    step = 0
    dims = 3
    env = Env(dims)
    rl = RL(dims,gamma=0.9)
    for i in range(2000000):
        env.reset()
        # print('第{0}轮游戏,初始状态'.format(i))
        # env.print_cur()

        while True:
            state = env.get_cur_state()
            action = rl.choose_action(state)
            
            next_state, reward, is_done = env.step(action)
            rl.store_to_memory(state, action, reward, next_state)
            
            if step > 200 and (step % 5 == 0):
                rl.train()
            
            if is_done:
                if reward == 1:
                    print('第{0}轮游戏成功'.format(i))
                    env.print_cur()
                break
            step += 1
        
        if i > 0 and (i % 49999 ==0):
            rl.plot_cost()