import numpy as np
import time
from Tool.action import restart
from Tool.action import  take_action
from Tool.action import TackAction
from Tool.screngrap import screngrap
from ultralytics import YOLO
from datetime import datetime

model = YOLO("./DQN_HollowKnight/YOLO/best.pt")
hold_time = []
# 載入模型
class HollowKnightEnv:
    def __init__(self):
        # 初始化環境屬性
        
        self.state = None  # 當前狀態 (例如：圖像或遊戲數據)
        self.previous_state = None  # 前一狀態，用於計算差異
        self.done = False  # 是否結束
        self.score = 0  # 遊戲分數
        self.health =8  # 假設有健康值
        self.step_count = 0  # 當前步數
        self.boss_health = 636
        self.first_attacked = False 
        self.attack_fail = 0 
        self.boss_left = 0 
        self.hero_left = 0
    def reset(self):
        """
        重置環境到初始狀態
        """
        restart()
        self.attack_fail = 0
        self.first_attacked = False 
        self.state = self.get_current_state()  # 獲取遊戲初始狀態
        self.previous_state = self.state
        self.done = False
        self.score = 0
        self.health = 8
        self.boss_health = 636
        self.step_count = 0
        self.boss_left = 0 
        self.hero_left = 0
        return self.state
    def step(self, action):
        """
        執行動作，計算下一狀態、獎勵和是否結束
        """
        # 執行動作
        # take_action(action)

        action_thread = TackAction(threadID=1, name="ActionThread", direction=None, action=action)  # 0 代表 Attack
        action_thread.start()
       # 動作延遲 (根據需要調整)
        # 更新狀態
        # self.previous_state = self.state
        # self.state = self.get_current_state()
        time.sleep(0.17)
        # 計算獎勵

        reward,previous_HP_reward = self.calculate_reward(action)
        print("間距{}".format(str(abs(self.hero_left-self.boss_left))))
        self.health = self.get_health()
        self.boss_health = self.get_boss_health()
        # 更新是否結束
        self.done = self.check_done()
        # 遞增步數
        self.step_count += 1

        return reward,previous_HP_reward, self.done

    def calculate_reward(self,action):
        """
        計算獎勵
        """
        reward =  0
        previous_HP_reward = 0 
        # 示例：根據健康值變化計算獎勵
        health_diff = self.get_health() - self.health
        boss_health_diff = self.get_boss_health() - self.boss_health 
# 6     
        
        boss_hero_pos = self.get_position()
        dr = 1
        
        if(abs(self.hero_left-self.boss_left) < 150):dr = -1
        print(dr)
        if(boss_hero_pos != "fail" ):
            if(boss_hero_pos == "left"):
                if(action // 2 == 1 and action % 2 == 0 ) : 
                    print("移動方向正確")
                    reward += dr * 3
                else :
                    reward -= dr * 3
            elif(boss_hero_pos == "right"):
                if(action // 2 == 2 and action % 2 == 0   ) : 
                    print("移動方向正確")
                    reward += dr * 3
                else:
                    reward -= dr * 3
        if health_diff < 0:
            previous_HP_reward = -5 # 損失健康值，給負獎勵
            print("扣血")
        if(boss_health_diff < 0 and health_diff >= 0 ):
            print("攻擊成功")
            reward += 1
            self.attack_fail = 0 
        # if(previous_HP_reward == 0 and reward == 0 ) : reward = -0.08
        # 示例：根據分數增長計算獎勵
        # 更新當前健康值和分數

        return reward,previous_HP_reward
    def get_position(self):


        results = model.predict(source=screngrap.grap("Hollow Knight",img2_return = True), conf=0.4, device=0) 
        if(len(results[0].boxes.cls) != 2):
            if(self.boss_left !=0 and self.hero_left !=0 ): 
                if(self.boss_left < self.hero_left) : 
                    return "left"
                else :
                    return "right"
        if(len(results[0].boxes.cls) == 2):
            if((results[0].boxes[0].cls.item() == 0 and results[0].boxes[1].cls.item() == 1) or (results[0].boxes[0].cls.item() == 1 and results[0].boxes[1].cls.item() == 0)):
                for i in range(0,2):
                    if(results[0].boxes[i].cls.item() == 0):
                        self.boss_left = results[0].boxes[i].xyxy[0][0] 
                    elif(results[0].boxes[i].cls.item() == 1):
                        self.hero_left = results[0].boxes[i].xyxy[0][0]

        if(self.boss_left < self.hero_left) : 
            return "left"
        else :
            return "right"
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

    def get_current_state(self):
        """
        獲取當前狀態 (圖像或遊戲數據)
        這裡可以添加擷取螢幕或遊戲內部 API 的代碼
        """
        # 示例：返回空狀態
        return np.zeros((84, 84))  # 假設狀態是 84x84 的灰度圖像
    def get_boss_health(self):
        boss_hp = screngrap.grap_Boss_hp("Hollow Knight")
        now_hp = 0
        for i in range(6,boss_hp[16].shape[0]):
            if(boss_hp[20][i][2] >= 234):
                now_hp+=1
        if(now_hp >= 600 and now_hp < 636):
            self.first_attacked = True
        if(self.first_attacked == False):
            now_hp = 636
        # print(now_hp)
        return now_hp
    def get_health(self):
        """
        獲取當前健康值
        可以通過遊戲 API 或畫面像素分析獲得
        """
        hp = screngrap.grap_hp("Hollow Knight")
        now_hp = 0
        for i in range(23,310,39):
            if(hp[20][i][0] > 190):
                now_hp+=1
        return now_hp  # 示例：直接返回屬性

    def get_score(self):
        """
        獲取當前分數
        可以通過遊戲 API 或畫面像素分析獲得
        """
        return self.score  # 示例：直接返回屬性
