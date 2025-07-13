import numpy as np
import time

from Tool.action import  take_action
from Tool.action import  take_direction
from Tool.action import TackAction
from Tool.screngrap import screngrap
from ultralytics import YOLO
from datetime import datetime
import socket
import math
import os

model = YOLO("./YOLO/best.pt")
hold_time = []
# 載入模型
class HollowKnightEnv:
    def __init__(self):
        # 初始化環境屬性
        self.nowhealth = 8
        self.nowBosshealth =900
        self.nowHeroX = 0
        self.nowBossX = 0
        self.nowHeroY = 0
        self.nowBossY = 0
        self.mp = 0
        self.state = None  # 當前狀態 (例如：圖像或遊戲數據)
        self.done = False  # 是否結束
        self.score = 0  # 遊戲分數
        self.health =8  # 假設有健康值
        self.step_count = 0  # 當前步數
        self.boss_health = 900
        self.first_attacked = False 
        self.attack_fail = 0 
        self.boss_left = 0 
        self.hero_left = 0
    def reset(self):
        """
        重置環境到初始狀態
        """
        self.mp = 0
        self.nowhealth = 8
        self.nowBosshealth =900
        self.nowHeroX = 0
        self.nowBossX = 0
        self.nowHeroY = 0
        self.nowBossY = 0

        self.attack_fail = 0
        self.first_attacked = False 
        self.done = False
        self.score = 0
        self.health = 8
        self.boss_health = 900
        self.step_count = 0
        self.boss_left = 0 
        self.hero_left = 0
        return self.state
    def step(self, move_action,attack_action,is_random):
        hornet_skill1 = False
        if(is_random):
            if self.nowBossY > 32 and self.nowBossY < 32.5:
                hornet_skill1 = True

            move_action = self.better_move(self.nowBossX,self.nowHeroX,hornet_skill1)
            attack_action = self.better_action(float(self.mp),self.nowBossX,self.nowBossY,self.nowHeroX,hornet_skill1)
        take_direction(move_action)
        take_action(attack_action)

        # action_thread = TackAction(threadID=1, name="ActionThread", direction=None, action=move_action)  # 0 代表 Attack
        # action_thread.start()

        self.get_hp_position()
        reward = self.move_judge(self.health,self.nowhealth,self.nowHeroX,self.nowBossX,move_action)
        attack_reward = self.action_judge(self.boss_health,self.nowBosshealth,self.health,self.nowhealth,self.nowHeroX,self.nowBossX,self.nowBossY,attack_action,hornet_skill1)
        
        self.health = self.nowhealth
        self.boss_health = self.nowBosshealth

        print(f"[REWARD] Total: {reward} | Attack: {attack_reward}")

        self.step_count += 1

        return reward,attack_reward, self.done


    def better_move(self, hornet_x ,player_x, hornet_skill1):
        dis = abs(player_x - hornet_x)
        dire = player_x - hornet_x
        if hornet_skill1:
            # run away while distance < 6
            if dis < 5:
                if dire > 0:
                    return 1
                else:
                    return 0
            # do not do long move while distance > 6
            else:
                if dire > 0:
                    return 2
                else:
                    return 3                
        if dis < 2.5:
            if dire > 0:
                return 1
            else:
                return 0
        elif dis < 5:
            if dire > 0:
                return 2
            else:
                return 3
        else:
            if dire > 0:
                return 0
            else:
                return 1

    def better_action(self, soul,hornet_x, hornet_y, player_x, hornet_skill1):
        dis = abs(player_x - hornet_x)
        if hornet_skill1:
            if dis < 3:
                return 6
            else:
                return 1
        
        if hornet_y > 34 and dis < 5 and soul >= 33:
            return 4
        if dis < 1.5:
            return 6
        elif dis < 5:
            if hornet_y > 32:
                return 6
            else:
                act = np.random.randint(6)
                if soul < 33:
                    while act == 4 or act == 5:
                        act = np.random.randint(6)
                return act
        elif dis < 12:
            act = np.random.randint(2)
            return 2 + act
        else:
            return 6
        
    def move_judge(self,self_blood, next_self_blood, player_x, hornet_x, move):
        hornet_skill1 = False
        if(self.nowBossY > 29):hornet_skill1 = True
        
        if hornet_skill1:
            # run away while distance < 5
            if abs(player_x - hornet_x) < 5:
                # change direction while hornet use skill
                if move == 0 or move == 2:
                    dire = 1
                else:
                    dire = -1
                if player_x - hornet_x > 0:
                    s = -1
                else:
                    s = 1
                # if direction is correct and use long move
                if dire * s == 1 and move < 2:
                    return 10
            # do not do long move while distance > 5
            else:
                if move >= 2:
                    return 10
            return -10

        dis = abs(player_x - hornet_x)
        dire = player_x - hornet_x
        if move == 0:
            if (dis > 5 and dire > 0) or (dis < 2.5 and dire < 0):
                return 10
        elif move == 1:
            if (dis > 5 and dire < 0) or (dis < 2.5 and dire > 0):
                return 10
        elif move == 2:
            if dis > 2.5 and dis < 5 and dire > 0:
                return 10
        elif move == 3:
            if dis > 2.5 and dis < 5 and dire < 0:
                return 10
        return -10
    
    def count_self_reward(self,next_self_blood, self_hp):
        if next_self_blood - self_hp < 0:
            return 11 * (next_self_blood - self_hp)
        return 0
    
    def count_boss_reward(self,next_boss_blood, boss_blood):
        if next_boss_blood - boss_blood < 0:
            return int((boss_blood - next_boss_blood) / 8)
        return 0
    

    @staticmethod
    def direction_reward(move, player_x, hornet_x):
        # 八种情况，我逐个分析一下 dis/s/dire
        # 1.-1-1-1 距离小于2.5，boss在左边，向左，奖励为负
        # 2.-1-1 1 距离小于2.5，boss在左边，向右，奖励为正
        # 3.-1 1-1 距离小于2.5，boss在右，向左，奖励为正
        base = 3
        if abs(player_x - hornet_x) < 2.5: # 危险距离，要远离
            dis = -1
        else:
            dis = 1
        if player_x - hornet_x > 0:
            s = -1
        else:
            s = 1
        if move == 0 or move == 2 :
            dire = -1
        else:
            dire = 1
        return dire * s * dis * base
    
    @staticmethod
    def distance_reward(move, next_player_x, next_hornet_x):
        if abs(next_player_x - next_hornet_x) < 2.5:
            return -3
        elif abs(next_player_x - next_hornet_x) < 5:
            return 3
        else:
            if move < 2 :
                return 2
            else:
                return -2

    @staticmethod
    def act_skill_reward(hornet_skill1, action, next_hornet_x, next_hornet_y, next_player_x):
        skill_reward = 0
        if hornet_skill1:
            if action == 2 or action == 3:
                skill_reward -= 5
        elif  next_hornet_y >34 and abs(next_hornet_x - next_player_x) < 5:
            if action == 4:
                skill_reward += 2
        return skill_reward
    
    @staticmethod
    def act_distance_reward(action, next_player_x, next_hornet_x, next_hornet_y):
        distance_reward = 0
        if abs(next_player_x - next_hornet_x) < 12:
            if abs(next_player_x - next_hornet_x) > 4:
                if (action >= 2 and action <= 3) or action == 0:
                    # distance_reward += 0.5
                    pass
                elif next_hornet_y < 29 and action == 6:
                    distance_reward -= 3
            else:
                if action >= 2 and action <= 3:
                    distance_reward -= 0.5
        else:
            if action == 0 or  action == 1 :
                distance_reward -= 3
            elif action == 6:
                distance_reward += 1
        return distance_reward

    # JUDGEMENT FUNCTION, write yourself
    def action_judge(self,boss_blood, next_boss_blood, self_blood, next_self_blood, next_player_x, next_hornet_x,next_hornet_y, action,hornet_skill1):
    # Player dead
        distance_reward = self.act_distance_reward(action, next_player_x, next_hornet_x, next_hornet_y)
        self_blood_reward = self.count_self_reward(next_self_blood, self_blood)
        boss_blood_reward = self.count_boss_reward(next_boss_blood, boss_blood)
        skill_reward = self.act_skill_reward(hornet_skill1,action,next_hornet_x,next_hornet_y,next_player_x)
        attackreward = self_blood_reward + boss_blood_reward + distance_reward  + skill_reward
        if action == 4:
            attackreward *= 1.5
        elif action == 5:
            attackreward *= 0.5
        return attackreward
        
    def get_hp_position(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('127.0.0.1', 5555))
                try:
                    data = s.recv(1024)
                    data = data.decode().replace("\n", '')
                    parts = data.split("/")  # 先以斜線切開
                    result = []
                    for i in range(0, len(parts)-1, 2):
                        hp = int(parts[i])
                        coords = parts[i+1].strip("()").split(",")
                        x = float(coords[0])
                        y = float(coords[1])
                        result.append({
                            "hp": hp,
                            "x": x,
                            "y": y
                        })
                    result.append({
                        "mp" : parts[4]
                    })
                    self.mp = parts[4] #mp
                    self.nowBosshealth = result[0]['hp']
                    self.nowBossX = result[0]['x']
                    self.nowhealth = result[1]['hp']
                    self.nowHeroX = result[1]['x'] 
                    self.nowBossY = result[0]['y']
                    self.nowHeroY = result[1]['y'] 

                    print(result)
                    return result
                except Exception as e:
                     self.done = True
            
    def check_done(self):
        """
        判斷遊戲是否結束
        """
        if self.boss_health <= 0 :
            return True 
        if self.health <= 0:
            return True  # 健康值耗盡，遊戲結束
        if self.step_count >= 1000:
            return True  # 步數上限，遊戲結束
        return False
#         def calculate_reward(self,action):
#         """
#         計算獎勵
#         """
#         reward =  0
#         # 示例：根據健康值變化計算獎勵
#         health_diff = self.nowhealth - self.health
#         boss_health_diff =  self.nowBosshealth  - self.boss_health 
# # 6     
#         boss_hero_pos = ""
#         dr = 0 
#         if(self.nowHeroX > self.nowBossX ):
#             boss_hero_pos = "left"
#             dr = 1
#         elif(self.nowHeroX < self.nowBossX ):
#             boss_hero_pos = "right"
#             dr = 1

#         if(abs(self.nowHeroX - self.nowBossX) < 4.8)  :reward += 3
#         if(abs(self.nowHeroX - self.nowBossX) < 2.5)  :reward -= 3
#         if(abs(self.nowHeroX - self.nowBossX)< 4) :  dr = -1

#         if(self.nowBossY > 29 ) :  
#             if(boss_hero_pos == "left"):
#                 boss_hero_pos = "right"
#             elif(boss_hero_pos == "right"):
#                 boss_hero_pos = "left"

#         if(boss_hero_pos != "fail" ):
#             if(boss_hero_pos == "left"):
#                 if(action % 5 == 1 and action // 5 == 0 ) : 
#                     reward += dr * 3
#                 elif(action % 5 != 1):
#                     reward -= dr * 3
#             elif(boss_hero_pos == "right"):
#                 if(action % 5 == 2 and action // 5 == 0) : 
#                     reward += dr * 3
#                 elif(action % 5 != 2) :
#                     reward -= dr * 3

#         if health_diff < 0:
#             reward = -11
#             self.health = self.nowhealth
            
#             print("扣血")
#         if boss_health_diff < 0  :
#             print("攻擊成功")
#             if(action // 5 == 0):
#                 reward += 5
#             self.boss_health = self.nowBosshealth
#         # if(boss_health_diff < 0 and health_diff >= 0 ):
#         #     print("攻擊成功")
#         #     reward += 1
#         #     self.attack_fail = 0 
#         # if(previous_HP_reward == 0 and reward == 0 ) : reward = -0.08
#         # 示例：根據分數增長計算獎勵
#         # 更新當前健康值和分數