import torch
import torch.nn as nn
import gc
import numpy as np
import random
from Tool.action import Nothing
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
import collections
from Tool.action import restart
from MLP import CoordinateNet
from Tool.CoordinateBuffer import CoordinateClient
# plt.ion()  # 打開交互模式
# fig, ax = plt.subplots()
# reward_line, = ax.plot([], [], label="Reward")
# ax.set_xlabel("Episodes")
# ax.set_ylabel("Reward")
# ax.legend()
all_rewards = []

move_action_num = 4
attack_action_num = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model =  ResNet3D(  num_actions=move_action_num,image_channels=1).to(device)
target_model =  ResNet3D(  num_actions=move_action_num,image_channels=1).to(device)

act_model =  ResNet3D( num_actions=attack_action_num,image_channels=1).to(device)
act_target_model =  ResNet3D(  num_actions=attack_action_num,image_channels=1).to(device)

# model = CoordinateNet(move_action_num).to(device)
# target_model = CoordinateNet(move_action_num).to(device)

# act_model = CoordinateNet(attack_action_num).to(device)
# act_target_model = CoordinateNet(attack_action_num).to(device)


update_count = 0
attack_update_count =0
model.load_state_dict(torch.load(".\save\HollowKnight_100000.pth"))
target_model.load_state_dict(torch.load(".\save\HollowKnight_100000.pth"))
act_model.load_state_dict(torch.load(".\save\HollowKnight_act_100000.pth"))
act_target_model.load_state_dict(torch.load(".\save\HollowKnight_act_100000.pth"))

epsilon =0
epsilon_min = 0   # 最小探索機率
epsilon_decay = 0.98
gridsize = 15
GAMMA = 0.995
TARGET_UPDATE_FREQUENCY = 4000 
NETWORK_UPDATE_FREQUENCY = 1
MODEL_SAVE_FREQUENCY = 4000
DELAY_REWARD = 2
optimizer = torch.optim.NAdam(model.parameters(), lr = 0.0001)
attack_optimizer  = torch.optim.NAdam(act_model.parameters(), lr = 0.0001)
attack_memory = ReplayMemory(600)
memory = ReplayMemory(600)

env = HollowKnightEnv()

def run_episode(num_games):
    model.eval()
    act_model.eval()
    time.sleep(10)
    frame_buffer = framebuffer.FrameBuffer(windows_name="HOLLOW KNIGHT", buffer_size=4, capture_interval=0.01)
    frame_buffer.start()
    cb = CoordinateClient()
    restart()

    DelayReward = collections.deque(maxlen=DELAY_REWARD)
    DelayStation = collections.deque(maxlen=DELAY_REWARD + 1) # 1 more for next_station
    DelayActions = collections.deque(maxlen=DELAY_REWARD)
    DelayDirection = collections.deque(maxlen=DELAY_REWARD)
    DelayCoords  = collections.deque(maxlen=DELAY_REWARD)

    attack_DelayReward = collections.deque(maxlen=DELAY_REWARD)
    attack_DelayStation = collections.deque(maxlen=DELAY_REWARD + 1) # 1 more for next_station
    attack_DelayActions = collections.deque(maxlen=DELAY_REWARD)
    attack_DelayDirection = collections.deque(maxlen=DELAY_REWARD)
    attack_DelayCoords  = collections.deque(maxlen=DELAY_REWARD)
    env.reset()  # 假設 `HollowKnightEnv` 提供 reset 方法
    

    run = True 
    move = 0
    games_played = 0
    total_reward = 0
    episode_games = 0
    while run:
        frames = frame_buffer.get_latest_3d_frames()
        latest = cb.get_coordinates()
        if len(latest) != 0 :        
            boss_x = (latest[0]["x"] - 15.3) / 22.3
            boss_y = (latest[0]["y"] - 28.4) / 9.2
            hero_x = (latest[1]["x"] - 15.3) / 22.3
            hero_y = (latest[1]["y"] - 28.4) / 9.2
        else:
            Nothing()
            all_rewards.append(total_reward)
            run = False
            frame_buffer.running=False
            return total_reward
        if latest:
            print("最新座標:", latest)
        rand = np.random.uniform(0, 1) 
        action = 0
        attack_action = 0
        is_random = False
        coords = torch.tensor([[boss_x, boss_y, hero_x, hero_y]], dtype=torch.float32).to(device)
        global epsilon 
        if(frames == None):
            continue 
        if rand > epsilon and frames != None:
            if(frames.shape[2] == 4):
                with torch.no_grad():
                    # action = torch.argmax(torch.tensor([[boss_x, boss_y, hero_x, hero_y]], dtype=torch.float32),dim=1).item()  
                    # attack_action = torch.argmax(torch.tensor([[boss_x, boss_y, hero_x, hero_y]], dtype=torch.float32),dim=1).item()  

                    action = torch.argmax(model(frames.to(device),coords), dim=1).item()
                    attack_action = torch.argmax(act_model(frames.to(device),coords), dim=1).item()
                    is_random = False
                print ("模型" + str(action))
        else:
            is_random = True


        action,attack_action,reward,attack_reward,done = env.step(action,attack_action,is_random)
        print("隨機：" + str(action) + " " +  str(attack_action))

        DelayReward.append(reward)
        DelayStation.append(frames)
        DelayActions.append(action)
        DelayDirection.append(move)
        DelayCoords.append(coords)

        attack_DelayReward.append(attack_reward)
        attack_DelayStation.append(frames)
        attack_DelayActions.append(attack_action)
        attack_DelayDirection.append(move)
        attack_DelayCoords.append(coords)
        if len(DelayStation) >= DELAY_REWARD + 1:
            if DelayReward[0] != 0:
                memory.push(DelayStation[0], DelayActions[0], DelayReward[0], DelayStation[1],DelayCoords[0], DelayCoords[1], done)

        if len(attack_DelayStation) >= DELAY_REWARD + 1:
            if attack_DelayReward[0] != 0:
                attack_memory.push(attack_DelayStation[0], attack_DelayActions[0], attack_DelayReward[0], attack_DelayStation[1],attack_DelayCoords[0],attack_DelayCoords[1] , done)
                      
        memory.truncate()
        attack_memory.truncate()

        total_reward += reward

        episode_games += 1
        # 確保 epsilon 不小於 epsilon_min
        if done == True:
            Nothing()
            run = False 
            all_rewards.append(total_reward)
            total_reward = 0 
            # len_array.append(len_of_snake)
            # board.resetgame()aa
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
            torch.save(model.state_dict(), f'./save/HollowKnight_{update_count}.pth')
            print("模型儲存")

    return total_loss

def learn(num_updates, batch_size, target_model, TARGET_UPDATE_FREQUENCY):
    total_loss = 0
    model.train()
    target_model.eval()
    for i in range(num_updates):

        optimizer.zero_grad()
        sample = memory.sample(batch_size)
        states, actions, rewards, next_states, coords,next_coords,dones = sample

        states = torch.cat([x for x in states], dim=0).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.cat([x for x in next_states], dim=0).to(device)
        dones = torch.FloatTensor(dones).to(device)
        coords = torch.cat([c.detach().clone().unsqueeze(0) if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in coords], dim=0).to(device)
        next_coords = torch.cat([c.detach().clone().unsqueeze(0) if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in next_coords], dim=0).to(device)
        q_local = model.forward(states,coords)
        with torch.no_grad():
            next_q_value = target_model.forward(next_states,next_coords)

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
            torch.save(model.state_dict(), f'./save/HollowKnight_{update_count}.pth')
            print("模型儲存")
    return total_loss

def attack_learn(num_updates, batch_size, target_model, TARGET_UPDATE_FREQUENCY):
    total_loss = 0
    act_model.train()
    act_target_model.eval()
    for i in range(num_updates):
        attack_optimizer.zero_grad()
        sample = attack_memory.sample(batch_size)
        states, actions, rewards, next_states, coords,next_coords,dones = sample
        states = torch.cat([x for x in states], dim=0).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.cat([x for x in next_states], dim=0).to(device)
        dones = torch.FloatTensor(dones).to(device)
        coords = torch.cat([c.detach().clone().unsqueeze(0) if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in coords], dim=0).to(device)
        next_coords = torch.cat([c.detach().clone().unsqueeze(0) if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in next_coords], dim=0).to(device)

        q_local = act_model.forward(states,coords)
        with torch.no_grad():
            next_q_value = act_target_model.forward(next_states,next_coords)

        Q_expected = q_local.gather(1, actions.unsqueeze(0).transpose(0, 1)).transpose(0, 1).squeeze(0)
        Q_targets_next = torch.max(next_q_value, dim=1)[0] * (torch.ones_like(dones) - dones)

        Q_targets = rewards + GAMMA * Q_targets_next

        loss = MSE(Q_expected, Q_targets)

        total_loss += loss

        loss.backward()
        attack_optimizer.step()
        global attack_update_count 
        attack_update_count += 1
        print("更新網路：" + str(attack_update_count))
        if attack_update_count % TARGET_UPDATE_FREQUENCY == 0:
            act_target_model.load_state_dict(act_model.state_dict())
            print("更新網路")
        if attack_update_count % MODEL_SAVE_FREQUENCY == 0 :
            torch.save(act_model.state_dict(), f'./save/HollowKnight_act_{attack_update_count}.pth')
            print("模型儲存")
    return total_loss

num_episodes = 60000
num_updates =200
print_every = 10
games_in_episode =10
batch_size = 16
reward_file = "all_rewards.csv"

def save_rewards(all_rewards, filename=reward_file):
    with open(filename, "w") as f:
        f.write(",".join([str(r) for r in all_rewards]))
    # reward_line.set_data(range(len(all_rewards)), all_rewards)
    # ax.relim()           # 重新計算所有數據的範圍
    # ax.autoscale_view()  # 自動縮放 x, y 軸
    # plt.pause(0.01)
    
def train():
    scores_array = []  
    avg_scores_array = []  
    time_start = time.time() 
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: Grad Max = {param.grad.abs().max()}, Weight Max = {param.data.abs().max()}")
    for i_episode in range(1, num_episodes + 1):
        time.sleep(7)
        run_episode(games_in_episode) 
        save_rewards(all_rewards)

        # 每次更新 all_rewards 時，寫入檔案

        if(i_episode % NETWORK_UPDATE_FREQUENCY ==0):
            total_loss_attack = attack_learn(num_updates, batch_size,act_target_model,TARGET_UPDATE_FREQUENCY)

            time.sleep(1)
            total_loss = learn(num_updates, batch_size,target_model,TARGET_UPDATE_FREQUENCY)

            print(total_loss)
            print(total_loss_attack)
            global epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            time.sleep(1)
        time.sleep(1)

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