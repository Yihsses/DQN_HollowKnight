import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import time
import matplotlib.pyplot as plt
from Tool import screngrap
from Tool.action import Nothing, restart
from Tool import framebuffer
from Tool.CoordinateBuffer import CoordinateClient
from replay_buff import ReplayMemory
from hollowknight_env import HollowKnightEnv
from MLP import CombinedNet
from ResNetEmp import ResNetEmbedding
import torchvision 
import csv

# ------------------ 全域參數 ------------------
import matplotlib.pyplot as plt

# plt.ion()
# fig, ax = plt.subplots()
# line, = ax.plot([], [], label="Reward")
# ax.set_xlabel("Episodes (x10)")
# ax.set_ylabel("Average")
# ax.legend()

TOTALWIN = 0
TOTALGAME = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

move_action_num = 4
attack_action_num = 7
cb =   CoordinateClient()
dx = 45.0892857143
dy = (525-25)/(39-28.6)

epsilon = 0
epsilon_min = 0
epsilon_decay = 0.98
GAMMA = 0.995
TARGET_UPDATE_FREQUENCY = 4000
NETWORK_UPDATE_FREQUENCY = 1
MODEL_SAVE_FREQUENCY = 4000
DELAY_REWARD = 2
ISTRAIN = False
avg_rewards = []

# ------------------ 初始化模型 ------------------


state_dim = 6 + 5  # one-hot + boss_x, boss_y, hero_x, hero_y, mp
model = CombinedNet(move_action_num, input_dim=state_dim).to(device)
target_model = CombinedNet(move_action_num, input_dim=state_dim).to(device)
act_model = CombinedNet(attack_action_num, input_dim=state_dim).to(device)
act_target_model = CombinedNet(attack_action_num, input_dim=state_dim).to(device)

model.load_state_dict(torch.load(".\\save\\HollowKnightMove_116000_v6.pth", map_location=device))
target_model.load_state_dict(torch.load(".\\save\\HollowKnightMove_116000_v6.pth", map_location=device))
act_model.load_state_dict(torch.load(".\\save\\HollowKnightAttack_116000_v6.pth", map_location=device))
act_target_model.load_state_dict(torch.load(".\\save\\HollowKnightAttack_116000_v6.pth", map_location=device))

optimizer = optim.NAdam(model.parameters(), lr=0.0001)
attack_optimizer = optim.NAdam(act_model.parameters(), lr=0.0001)
class_names = ["down", "nomove", "nothing", "rush", "shot", "skill"] 

memory = ReplayMemory(600)
attack_memory = ReplayMemory(600)

env = HollowKnightEnv()

update_count = 0
attack_update_count = 0

MSE = nn.MSELoss()

num_classes = len(class_names)
resnet_embed = torchvision.models.resnet18(pretrained=False)  # 這裡不用 pretrained
resnet_embed.fc = nn.Linear(resnet_embed.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_embed.load_state_dict(torch.load("YOLO/resnet18_best.pth", map_location=device))
resnet_embed.to(device)
resnet_embed.eval()
episode_rewards = []

# def update_plot(avg_reward):
#     avg_rewards.append(avg_reward)
#     line.set_xdata(range(len(avg_rewards)))
#     line.set_ydata(avg_rewards)
#     ax.relim()
#     ax.autoscale_view()
#     plt.pause(0.01)

# ------------------ 遊戲回合 ------------------
def run_episode(num_games):
    global epsilon,TOTALWIN,TOTALGAME
    model.eval()
    act_model.eval()
    cb = CoordinateClient()
    restart()
    env.reset()

    # Delay buffers
    DelayReward = collections.deque(maxlen=DELAY_REWARD)
    DelayStation = collections.deque(maxlen=DELAY_REWARD + 1)
    DelayActions = collections.deque(maxlen=DELAY_REWARD)
    DelayDirection = collections.deque(maxlen=DELAY_REWARD)
    DelayCoords = collections.deque(maxlen=DELAY_REWARD)

    attack_DelayReward = collections.deque(maxlen=DELAY_REWARD)
    attack_DelayStation = collections.deque(maxlen=DELAY_REWARD + 1)
    attack_DelayActions = collections.deque(maxlen=DELAY_REWARD)
    attack_DelayDirection = collections.deque(maxlen=DELAY_REWARD)
    attack_DelayCoords = collections.deque(maxlen=DELAY_REWARD)

    run = True
    total_reward = 0
    games_played = 0
    prev_class = "nothing"
    prev_idx = 2
    while run:
        latest = cb.get_coordinates()
        if len(latest) == 0:
            Nothing()
            run = False
            TOTALWIN += 1
            return total_reward
        
        x = (latest[0]['x']-15.3) * dx +50
        y = 525-(latest[0]['y']-28.6) * dy  

        output  = resnet_embed(screngrap.screngrap.grap('HOLLOW KNIGHT', d_height=180, d_width=200, d_top=y, d_left=x, img2_return=True).to(device))

        # 轉成機率
        probs = F.softmax(output, dim=1)
        max_prob, pred_idx = torch.max(probs, 1)    # 找最大值和索引

        # 取得其他類別機率
        other_probs = probs[0, [i for i in range(len(class_names)) if i != pred_idx.item()]]

        # 條件判斷
        state_onehot = torch.zeros(len(class_names))
        if max_prob.item() > 0.8 and torch.all(other_probs < 0.3):
            state_onehot[pred_idx.item()] = 1
            pred_class = class_names[pred_idx.item()]
            prev_class = pred_class
            prev_idx = pred_idx.item()
        else:
            state_onehot[prev_idx] = 1
            pred_class = prev_class  # 維持上一個動作
        print(pred_class)
        boss_x = (latest[0]["x"] - 15.3) / 22.3
        boss_y = (latest[0]["y"] - 28.4) / 9.2
        hero_x = (latest[1]["x"] - 15.3) / 22.3
        hero_y = (latest[1]["y"] - 28.4) / 9.2
        mp = float(latest[2]["mp"]) / 50
        state = torch.tensor([*state_onehot, boss_x, boss_y, hero_x, hero_y, mp], dtype=torch.float32).unsqueeze(0).to(device)  # [1, len(onehot)+5]
        # 選擇動作
        rand = np.random.uniform()
        if rand > epsilon:
            with torch.no_grad():
                action = torch.argmax(model(state), dim=1).item()
                attack_action = torch.argmax(act_model(state), dim=1).item()
                is_random = False
        else:
            action = np.random.randint(move_action_num)
            attack_action = np.random.randint(attack_action_num)
            is_random = True

        # 執行動作
        action, attack_action, reward, attack_reward, done,TOTALWIN = env.step(action, attack_action, is_random,pred_class,TOTALWIN)
        print("攻擊動作",attack_action)
        # 更新 Delay buffer
        DelayReward.append(reward)
        DelayStation.append(state)
        DelayActions.append(action)


        attack_DelayReward.append(attack_reward)
        attack_DelayStation.append(state)
        attack_DelayActions.append(attack_action)



        # 儲存到 replay buffer
        if len(DelayStation) >= DELAY_REWARD + 1 and DelayReward[0] != 0:
            memory.push(DelayStation[0], DelayActions[0], DelayReward[0], DelayStation[1], done)
        if len(attack_DelayStation) >= DELAY_REWARD + 1 and attack_DelayReward[0] != 0:
            attack_memory.push(attack_DelayStation[0], attack_DelayActions[0], attack_DelayReward[0],attack_DelayStation[1], done)

        memory.truncate()
        attack_memory.truncate()

        total_reward += reward

        if done:
            Nothing()
            run = False
            games_played += 1
            if games_played >= num_games:
                break

    return total_reward

# ------------------ 更新網路 ------------------
def learn(num_updates, batch_size, target_model, is_attack=False):
    global update_count, attack_update_count
    total_loss = 0
    net, optimizer_ = (act_model, attack_optimizer) if is_attack else (model, optimizer)
    target_net = act_target_model if is_attack else target_model
    net.train()
    target_net.eval()

    memory_ = attack_memory if is_attack else memory

    for _ in range(num_updates):
        optimizer_.zero_grad()
        sample = memory_.sample(batch_size)
        states, actions, rewards, next_states, dones = sample

        states = torch.cat(states, dim=0).to(device)
        next_states = torch.cat(next_states, dim=0).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Q 值計算
        q_local = net(states)  # <-- 直接用 state，不需要 coords
        with torch.no_grad(): 
            next_q = target_net(next_states)

        Q_expected = q_local.gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_targets_next = torch.max(next_q, dim=1)[0] * (1 - dones)
        Q_targets = rewards + GAMMA * Q_targets_next

        loss = MSE(Q_expected, Q_targets)
        total_loss += loss.item()
        loss.backward()
        optimizer_.step()

        # 更新 target model
        if is_attack:
            attack_update_count += 1
            if attack_update_count % TARGET_UPDATE_FREQUENCY == 0:
                act_target_model.load_state_dict(act_model.state_dict())
                torch.save(act_model.state_dict(), f'./save/HollowKnightAttack_{attack_update_count}_v6.pth')
            print("更新網路:", attack_update_count)

        else:
            update_count += 1
            if update_count % TARGET_UPDATE_FREQUENCY == 0:
                target_model.load_state_dict(model.state_dict())
                torch.save(model.state_dict(), f'./save/HollowKnightMove_{update_count}_v6.pth')
            print("更新網路:", update_count)

    return total_loss

def writeCSV():
    filename = "HollowKnight_results.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 寫入欄位名稱
        writer.writerow(["Total Games", "Total Wins", "Win Rate"])
        # 寫入數據
        win_rate = TOTALWIN / TOTALGAME if TOTALGAME > 0 else 0
        writer.writerow([TOTALGAME, TOTALWIN, win_rate])

# ------------------ 訓練 ------------------ 
def train(num_episodes=60000, num_updates=400, batch_size=32, games_in_episode=10):
    global epsilon,TOTALWIN,TOTALGAME
    for i_episode in range(1, num_episodes + 1):
        time.sleep(7)
        reward = run_episode(games_in_episode)
        episode_rewards.append(reward)
        print("獎勵:",reward)
        if ISTRAIN :
            if i_episode % 10 == 0:
                avg_reward = sum(episode_rewards[-5:]) / 5
                # update_plot(avg_reward)
                print(f"Episode {i_episode}, Average Reward (last 10): {avg_reward}")
            # 更新網路
            if i_episode % NETWORK_UPDATE_FREQUENCY == 0:
                attack_loss = learn(num_updates, batch_size, act_target_model, is_attack=True)
                time.sleep(1)
                move_loss = learn(num_updates, batch_size, target_model)
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
        else:
            time.sleep(5)
        TOTALGAME += 1
        print("總場數：",TOTALGAME)
        print("總勝利數：",TOTALWIN)
        writeCSV()
    return 1

# ------------------ 執行 ------------------
if __name__ == "__main__":
    scores = train()
    print("Total episodes:", len(scores))
    plt.plot(scores)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()
