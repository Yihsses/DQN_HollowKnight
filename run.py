import torch
import torch.nn as nn
import gc
import numpy as np
import random
from dqn_3cnn import Q_construct_3d
# from dqnnet import QNetwork
# from DQN_HollowKnight.dqn_net import QNetworktest
from Tool import screngrap
from collections import deque
import time
import matplotlib.pyplot as plt
from replay_buff import ReplayMemory
from Tool import framebuffer
from hollowknight_env import HollowKnightEnv
import torch.cuda.amp as amp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model =  Q_construct_3d(height=1280 // 4, width=720 // 4, num_actions=6, image_channels=1).to(device)
target_model = Q_construct_3d(height=1280 // 4, width=720 // 4, num_actions=6, image_channels=1).to(device)

update_count = 0

frame_buffer = framebuffer.FrameBuffer(windows_name="HOLLOW KNIGHT", buffer_size=4, capture_interval=0.05)
epsilon = -1
epsilon_min = 0.1  # 最小探索機率
epsilon_decay = 0.995  
gridsize = 15
GAMMA = 0.9
TARGET_UPDATE_FREQUENCY = 100

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

memory = ReplayMemory(100)
env = HollowKnightEnv()
frame_buffer.start()
def run_episode(num_games):
    run = True
    move = 0
    games_played = 0
    total_reward = 0
    episode_games = 0
    len_array = []
    
    while run:
        frames = frame_buffer.get_latest_frames()
         
        # state = screngrap.screngrap.grap('HOLLOW KNIGHT')ㄇ
        # state = torch.tensor(state).permute(2, 0, 1)
        # state = torch.tensor(state, dtype=torch.fㄨloat32) / 255.0
        # state = state.unsqueeze(0)
        # action_0 = model.forward(state)
        # rand = np.random.uniform(0, 1)
        rand = np.random.uniform(0, 1)  # 隨機生成一個 0 到 1 之間的數字
        action = 0
        global epsilon 
        if rand > epsilon and frames != None:
            if(len(frames)>=4):
                frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
                action = torch.argmax(model(frames.to(device)), dim=1).item()
                print("模型：" + str(action))
        else:
            frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
            action = np.random.randint(0, 6)
            # print("隨機：" + str(action))
        now_state =  frames.clone()
        reward , done = env.step(action)
        frames = frame_buffer.get_latest_frames()
        next_state = frames.permute(1, 0, 2, 3).unsqueeze(0).clone()
        env.previous_state = now_state
        env.state = next_state
        memory.push(env.previous_state, action, reward, env.state , done)
        memory.truncate()
        total_reward += reward

        episode_games += 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # 確保 epsilon 不小於 epsilon_min
        if done == True:

            run = False 
            # len_array.append(len_of_snake)
            # board.resetgame()
            if num_games == games_played:
                run = False
        time.sleep(0.2)
        frame_buffer.stop()
    print("第一回合結束")
    # avg_len_of_snake = np.mean(len_array)
    # max_len_of_snake = np.max(len_array)
    return total_reward


MSE = nn.MSELoss()

import torch.amp as amp  # 使用 torch.amp 而非 torch.cuda.amp

def learn(num_updates, batch_size, target_model, update_count):
    total_loss = 0
    scaler = amp.GradScaler()  # 混合精度比例縮放器
    torch.cuda.empty_cache()
    gc.collect()
    for _ in range(num_updates):
        optimizer.zero_grad()

        # 從回放緩衝區取樣
        sample = memory.sample(batch_size) 
        states, actions, rewards, next_states, dones = sample

        # 將數據轉為張量並移動到設備
        states = torch.cat([x for x in states], dim=0).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.cat([x for x in next_states]).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # 啟用混合精度
        with amp.autocast(device_type='cuda'):  # 設定設備類型
            # 計算當前 Q 值
            q_local = model(states)  # 一次性計算整個批次的 Q 值
            next_q_value = target_model(next_states)  # 一次性計算下一狀態的目標 Q 值

            # 計算 Q_expected
            Q_expected = q_local.gather(1, actions.unsqueeze(1)).squeeze(1)  # 按照動作選擇對應 Q 值

            # 計算 Q_targets_next（處理結束狀態，避 免不必要的梯度計算）
            Q_targets_next = torch.max(next_q_value, dim=1)[0] * (1 - dones)

            # 計算目標 Q 值
            Q_targets = rewards + GAMMA * Q_targets_next

            # 計算損失
            loss = MSE(Q_expected, Q_targets)

        # 反向傳播與參數更新
        total_loss += loss.item()
        print(total_loss)
        scaler.scale(loss).backward()  # 使用比例縮放進行反向傳播
        scaler.step(optimizer)  # 更新模型參數
        scaler.update()  # 更新縮放器

        # 更新目標網路參數（定期同步）
        update_count += 1
        if update_count % TARGET_UPDATE_FREQUENCY == 0:
            target_model.load_state_dict(model.state_dict())

    return total_loss, update_count


# def learn(num_updates, batch_size):
#     total_loss = 0

#     for i in range(num_updates):
#         optimizer.zero_grad()
#         sample = memory.sample(batch_size)

#         states, actions, rewards, next_states, dones = sample
#         states = torch.cat([x.unsqueeze(0) for x in states], dim=0)
#         actions = torch.LongTensor(actions)
#         rewards = torch.FloatTensor(rewards)
#         next_states = torch.cat([x.unsqueeze(0) for x in next_states])
#         dones = torch.FloatTensor(dones)

#         q_local = model.forward(states)
#         next_q_value = model.forward(next_states)

#         Q_expected = q_local.gather(1, actions.unsqueeze(0).transpose(0, 1)).transpose(0, 1).squeeze(0)

#         Q_targets_next = torch.max(next_q_value, 1)[0] * (torch.ones(dones.size()) - dones)

#         Q_targets = rewards + GAMMA * Q_targets_next

#         loss = MSE(Q_expected, Q_targets)

#         total_loss += loss
#         loss.backward()
#         optimizer.step()

#     return total_loss


num_episodes = 60000
num_updates = 1
print_every = 10
games_in_episode = 30
batch_size = 3


def train():
    scores_deque = deque(maxlen=100)  # 保存最近 100 個回合的分數
    scores_array = []  # 保存每一回合的分數
    avg_scores_array = []  # 保存平均分數
    time_start = time.time()  # 記錄開始時間

    for i_episode in range(1, num_episodes + 1):
        # 初始化環境
        env.reset()  # 假設 `HollowKnightEnv` 提供 reset 方法
        score = run_episode(games_in_episode)  # 運行一個回合
        time.sleep(5)
        print(score)
        # scores_deque.append(score)  # 添加本次回合分數
        # scores_array.append(score)  # 保存分數
        # avg_score = np.mean(scores_deque)  # 計算最近100回合的平均分
        # avg_scores_array.append(avg_score)  # 保存平均分數

        # # 更新 Q 網絡
        total_loss = learn(num_updates, batch_size,target_model,TARGET_UPDATE_FREQUENCY)

        # # 打印訓練進度
        # if i_episode % print_every == 0:
        #     elapsed_time = int(time.time() - time_start)
        #     print(
        #         f"Ep.: {i_episode:6}, Loss: {total_loss:.3f}, "
        #         f"Avg.Score: {avg_score:.2f}, "
        #         f"Time: {elapsed_time // 3600:02}:{elapsed_time % 3600 // 60:02}:{elapsed_time % 60:02}"
        #     )

        # # 保存模型檔案
        # if i_episode % 1000 == 0:
        #     torch.save(model.state_dict(), f'./dir_chk_len/HollowKnight_{i_episode}.pth')

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
