import torch
import torch.nn as nn

import numpy as np
import random
from dqnnet import Q_construct
from dqnnet import QNetwork
# from DQN_HollowKnight.dqn_net import QNetworktest
from Tool import screngrap
from collections import deque
import time
import matplotlib.pyplot as plt
from replay_buff import ReplayMemory
from Tool import framebuffer
from hollowknight_env import HollowKnightEnv
model =  Q_construct(input_dim=int((1280/4)*(720/4)), num_actions=6,image_channels=4)
frame_buffer = framebuffer.FrameBuffer(windows_name="HOLLOW KNIGHT", buffer_size=4, capture_interval=0.05)
epsilon = 0
epsilon_min = 0.1  # 最小探索機率
epsilon_decay = 0.995  
gridsize = 15
GAMMA = 0.9
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
memory = ReplayMemory(1000)
env = HollowKnightEnv()

def run_episode(num_games):
    run = True
    move = 0
    games_played = 0
    total_reward = 0
    episode_games = 0
    len_array = []
    frame_buffer.start()
    while run:
        frames = frame_buffer.get_latest_frames()
        # state = screngrap.screngrap.grap('HOLLOW KNIGHT')
        # state = torch.tensor(state).permute(2, 0, 1)
        # state = torch.tensor(state, dtype=torch.float32) / 255.0
        # state = state.unsqueeze(0)
        # action_0 = model.forward(state)
        # rand = np.random.uniform(0, 1)
        rand = np.random.uniform(0, 1)  # 隨機生成一個 0 到 1 之間的數字
        action = 0
        global epsilon 
        if rand > epsilon:
            if(frames != None):
                if(len(frames)>=4):
                    action_0 = model(frame.s)
                    action =  action_0
        else:
            action = np.random.randint(0, 6)
        env.state = frames 
        reward , done = env.step(action)

        frames = frame_buffer.get_latest_frames()
        env.previous_state = env.state
        env.state = frames
        memory.push( env.previous_state, action, reward, env.state , done)

        total_reward += reward

        episode_games += 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # 確保 epsilon 不小於 epsilon_min
        if done == True:
            run = False 
            # len_array.append(len_of_snake)
            # board.resetgame()
            if num_games == games_played:
                run = False

    avg_len_of_snake = np.mean(len_array)
    max_len_of_snake = np.max(len_array)
    return total_reward, avg_len_of_snake, max_len_of_snake


MSE = nn.MSELoss()


def learn(num_updates, batch_size):
    total_loss = 0

    for i in range(num_updates):
        optimizer.zero_grad()
        sample = memory.sample(batch_size)

        states, actions, rewards, next_states, dones = sample
        states = torch.cat([x.unsqueeze(0) for x in states], dim=0)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.cat([x.unsqueeze(0) for x in next_states])
        dones = torch.FloatTensor(dones)

        q_local = model.forward(states)
        next_q_value = model.forward(next_states)

        Q_expected = q_local.gather(1, actions.unsqueeze(0).transpose(0, 1)).transpose(0, 1).squeeze(0)

        Q_targets_next = torch.max(next_q_value, 1)[0] * (torch.ones(dones.size()) - dones)

        Q_targets = rewards + GAMMA * Q_targets_next

        loss = MSE(Q_expected, Q_targets)

        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss


num_episodes = 60000
num_updates = 500
print_every = 10
games_in_episode = 30
batch_size = 20


def train():
    scores_deque = deque(maxlen=100)  # 保存最近 100 個回合的分數
    scores_array = []  # 保存每一回合的分數
    avg_scores_array = []  # 保存平均分數
    time_start = time.time()  # 記錄開始時間

    for i_episode in range(1, num_episodes + 1):
        # 初始化環境
        env.reset()  # 假設 `HollowKnightEnv` 提供 reset 方法
        score, _, _ = run_episode(games_in_episode)  # 運行一個回合

        scores_deque.append(score)  # 添加本次回合分數
        scores_array.append(score)  # 保存分數
        avg_score = np.mean(scores_deque)  # 計算最近100回合的平均分
        avg_scores_array.append(avg_score)  # 保存平均分數

        # 更新 Q 網絡
        total_loss = learn(num_updates, batch_size)

        # 打印訓練進度
        if i_episode % print_every == 0:
            elapsed_time = int(time.time() - time_start)
            print(
                f"Ep.: {i_episode:6}, Loss: {total_loss:.3f}, "
                f"Avg.Score: {avg_score:.2f}, "
                f"Time: {elapsed_time // 3600:02}:{elapsed_time % 3600 // 60:02}:{elapsed_time % 60:02}"
            )

        # 保存模型檔案
        if i_episode % 1000 == 0:
            torch.save(model.state_dict(), f'./dir_chk_len/HollowKnight_{i_episode}.pth')

    # 返回訓練結果
    return scores_array, avg_scores_array

if __name__ == "__main__":
    scores, avg_scores, avg_len_of_snake, max_len_of_snake = train()
    print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores, label="Score")
    plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores, label="Avg score on 100 episodes")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.ylabel('Score')
    plt.xlabel('Episodes #')
    plt.show()

    ax1 = fig.add_subplot(121)
    plt.plot(np.arange(1, len(avg_len_of_snake) + 1), avg_len_of_snake, label="Avg Len of Snake")
    plt.plot(np.arange(1, len(max_len_of_snake) + 1), max_len_of_snake, label="Max Len of Snake")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.ylabel('Length of Snake')
    plt.xlabel('Episodes #')
    plt.show()

    n, bins, patches = plt.hist(max_len_of_snake, 45, density=1, facecolor='green', alpha=0.75)
    l = plt.plot(np.arange(1, len(bins) + 1), 'r--', linewidth=1)
    mu = round(np.mean(max_len_of_snake), 2)
    sigma = round(np.std(max_len_of_snake), 2)
    median = round(np.median(max_len_of_snake), 2)
    print('mu: ', mu, ', sigma: ', sigma, ', median: ', median)
    plt.xlabel('Max.Lengths, mu = {:.2f}, sigma={:.2f},  median: {:.2f}'.format(mu, sigma, median))
    plt.ylabel('Probability')
    plt.title('Histogram of Max.Lengths')
    plt.axis([4, 44, 0, 0.15])
    plt.grid(True)

    plt.show()
