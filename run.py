import torch
import torch.nn as nn
import gc
import numpy as np
import random

import torch.optim.nadam
from Q_3d_resnet import ResNet3D
from dqnnet import Q_construct
from dqn_3cnn import Q_construct_3d
# from dqnnet import QNetwork
# from DQN_HollowKnight.dqn_net import QNetworktestj
from Tool import screngrap
from collections import deque
import time
import matplotlib.pyplot as plt
from replay_buff import ReplayMemory
from Tool import framebuffer
from hollowknight_env import HollowKnightEnv
from dqn_net import SimpleQ
import torch.cuda.amp as amp
action_num = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# model =  ResNet3D( height=200, width=400, num_actions=action_num,image_channels=1).to(device)
# target_model =  ResNet3D( height=200, width=400, num_actions=action_num,image_channels=1).to(device)
# model =  Q_construct(input_dim=int((400/4)*(200/4)), num_actions=6,image_channels=12).to(device)
# target_model =Q_construct(input_dim=int((400/4)*(200/4)), num_actions=6,image_channels=12).to(device)
model =  Q_construct_3d(height=400, width=200,time_steps=8, num_actions=6, image_channels=1).to(device)
target_model = Q_construct_3d(height=400, width=200, time_steps=8, num_actions=6, image_channels=1).to(device)
update_count = 0


epsilon =1
epsilon_min = 0.1  # 最小探索機率
epsilon_decay = 0.995
gridsize = 15
GAMMA = 0.995
TARGET_UPDATE_FREQUENCY = 5000
NETWORK_UPDATE_FREQUENCY = 1
MODEL_SAVE_FREQUENCY = 10000

optimizer = torch.optim.NAdam(model.parameters(), lr = 0.0001)

memory = ReplayMemory(1000)
env = HollowKnightEnv()

def run_episode(num_games):
    frame_buffer = framebuffer.FrameBuffer(windows_name="HOLLOW KNIGHT", buffer_size=8, capture_interval=0.05)
    frame_buffer.start()
    run = True
    move = 0
    games_played = 0
    total_reward = 0
    episode_games = 0
    len_array = []
    delay_reward = []
    time.sleep(0.5)
    model.eval()
    while run:
        frames = frame_buffer.get_latest_3d_frames()
        # state = screngrap.screngrap.grap('HOLLOW KNIGHT')ㄇ
        # state = torch.tensor(state).permute(2, 0, 1)
        # state = torch.tensor(state, dtype=torch.fㄨloat32) / 255.0
        # state = state.unsqueeze(0)
        # action_0 = model.forward(state)
        # rand = np.random.uniform(0, 1)
        rand = np.random.uniform(0, 1)  # 隨機生成一個 0 到 250 之間的數字
        action = 0
        global epsilon 
        if(frames == None):
            continue
        if rand > epsilon and frames != None:
            if(frames.shape[2] == 6):
                # frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
                with torch.no_grad():
                    action = torch.argmax(model(frames.to(device)), dim=1).item()
                print("模型" + str(action))
        else:
            # frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
            action = np.random.randint(0,action_num)
            print("隨機：" + str(action))
        now_state =  frames
        reward ,previous_HP_reward,done = env.step(action)
        frames = frame_buffer.get_latest_3d_frames() 
        if(previous_HP_reward != 0 ):
            for i in range(0,3):
                print(previous_HP_reward)
                index_to_modify = len(memory) - 1 -i
                old_experience = memory.buffer[index_to_modify]
                # 創建一個新的 tuple，替換第 3 個元素
                new_experience = (old_experience[0], old_experience[1], old_experience[2] +  previous_HP_reward, old_experience[3], old_experience[4])
                # 替換 buffer 中的舊資料
                memory.buffer[index_to_modify] = new_experience
        next_state = frames
        env.previous_state = now_state
        env.state = next_state
        memory.push(env.previous_state, action, reward, env.state , done)
        memory.truncate()
        total_reward += reward

        episode_games += 1
        # 確保 epsilon 不小於 epsilon_min
        if done == True:
            run = False 
            # len_array.append(len_of_snake)
            # board.resetgame()
            if num_games == games_played:
                run = False
    frame_buffer.running=False
    print("結束")
    # avg_len_of_snake = np.mean(len_array)
    # max_len_of_snake = np.max(len_array)
    return total_reward


MSE = nn.MSELoss()

import torch.amp as amp  # 使用 torch.amp 而非 torch.cuda.amp

import psutil

import torch.amp as amp  # 使用 torch.amp 而非 torch.cuda.amp
import psutil

def learn_td(num_updates, batch_size, target_model, TARGET_UPDATE_FREQUENCY, accumulation_steps=8):
    total_loss = 0
    model.train()
    target_model.eval()
    for update in range(num_updates):
        optimizer.zero_grad()  # 在每個梯度累加周期的起點清零
        for step in range(accumulation_steps):
            # 從回放緩衝區取樣
            sample = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = sample

            # 將數據轉為張量
            states = torch.cat([x for x in states], dim=0).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.cat([x for x in next_states], dim=0).to(device)
            dones = torch.FloatTensor(dones).to(device)

            # 計算當前 Q 值和下一狀態的 Q 值
            q_local = model.forward(states)
            with torch.no_grad():
                next_q_value = target_model.forward(next_states)

            # 選擇當前動作的 Q 值
            Q_expected = q_local.gather(1, actions.unsqueeze(1)).squeeze(1)

            # 計算 Q_targets_next，處理終止狀態
            Q_targets_next = torch.max(next_q_value, dim=1)[0] * (1 - dones)

            # TD 誤差計算
            TD_target = rewards + GAMMA * Q_targets_next
            TD_error = Q_expected - TD_target

            # 使用 TD 誤差平方作為損失
            loss = TD_error.pow(2).mean() / accumulation_steps  # 平均化損失
            total_loss += loss.item()

            # 反向傳播累加梯度
            loss.backward()

        # 梯度剪裁（可選）
        # 更新模型參數
        optimizer.step()

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: Grad Max = {param.grad.abs().max()}, Weight Max = {param.data.abs().max()}")

        # 更新目標網絡參數（定期同步）
        global update_count 
        update_count += 1
        print(update_count)
        if update_count % TARGET_UPDATE_FREQUENCY == 0:
            target_model.load_state_dict(model.state_dict())
            print("更新網路")
        if update_count % MODEL_SAVE_FREQUENCY == 0 :
            torch.save(model.state_dict(), f'./DQN_HollowKnight/save/HollowKnight_{update_count}.pth')
            print("模型儲存")

    return total_loss

def learn(num_updates, batch_size, target_model, TARGET_UPDATE_FREQUENCY):
    total_loss = 0
    model.train()
    target_model.eval()
    for i in range(num_updates):

        optimizer.zero_grad()
        sample = memory.sample(batch_size)
        states, actions, rewards, next_states, dones = sample
        states = torch.cat([x for x in states], dim=0).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.cat([x for x in next_states], dim=0).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_local = model.forward(states)
        with torch.no_grad():
            next_q_value = target_model.forward(next_states)

        Q_expected = q_local.gather(1, actions.unsqueeze(0).transpose(0, 1)).transpose(0, 1).squeeze(0)

        Q_targets_next = torch.max(next_q_value, dim=1)[0] * (torch.ones_like(dones) - dones)

        Q_targets = rewards + GAMMA * Q_targets_next

        loss = MSE(Q_expected, Q_targets)

        total_loss += loss

        loss.backward()
        optimizer.step()
        global update_count 
        update_count += 1
        print("更新網路：" + str(update_count))
        if update_count % TARGET_UPDATE_FREQUENCY == 0:
            target_model.load_state_dict(model.state_dict())
            print("更新網路")
        if update_count % MODEL_SAVE_FREQUENCY == 0 :
            torch.save(model.state_dict(), f'./DQN_HollowKnight/save/HollowKnight_{update_count}.pth')
            print("模型儲存")
    return total_loss


num_episodes = 60000
num_updates =500
print_every = 10
games_in_episode = 30
batch_size =16


def train():
    scores_array = []  # 保存每一回合的分數
    avg_scores_array = []  # 保存平均分數
    time_start = time.time()  # 記錄開始時間
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: Grad Max = {param.grad.abs().max()}, Weight Max = {param.data.abs().max()}")
    for i_episode in range(1, num_episodes + 1):
        # 初始化環境
        env.reset()  # 假設 `HollowKnightEnv` 提供 reset 方法
        run_episode(games_in_episode)  # 運行一個回合
        # scores_deque.append(score)  # 添加本次回合分數
        # scores_array.append(score)  # 保存分數
        # avg_score = np.mean(scores_deque)  # 計算最近100回合的平均分
        # avg_scores_array.append(avg_score)  # 保存平均分數
        # # 更新 Q 網絡
        if(i_episode % NETWORK_UPDATE_FREQUENCY ==0):
            total_loss = learn(num_updates, batch_size,target_model,TARGET_UPDATE_FREQUENCY)
            print(total_loss)
            time.sleep(5)
            global epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay) 
        else:
            time.sleep(7)
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
