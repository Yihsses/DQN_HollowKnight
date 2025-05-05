import numpy as np
import time
from Tool.action import restart
from Tool.action import  take_action
from Tool.action import TackAction
from Tool.screngrap import screngrap
class HollowKnightEnv:
    def __init__(self):
        # 初始化環境屬性
        self.state = None  # 當前狀態 (例如：圖像或遊戲數據)
        self.previous_state = None  # 前一狀態，用於計算差異
        self.done = False  # 是否結束
        self.score = 0  # 遊戲分數
        self.health = 100  # 假設有健康值
        self.step_count = 0  # 當前步數
        self.boss_health = 636
        self.first_attacked = False 
    def reset(self):
        """
        重置環境到初始狀態
        """
        # restart()
        self.first_attacked = False 
        self.state = self.get_current_state()  # 獲取遊戲初始狀態
        self.previous_state = self.state
        self.done = False
        self.score = 0
        self.health = 8
        self.boss_health = 636
        self.step_count = 0
        return self.state
    def step(self, action):
        """
        執行動作，計算下一狀態、獎勵和是否結束
        """
        # 執行動作
        # take_action(action)
        action_thread = TackAction(threadID=1, name="ActionThread", direction=None, action=action)  # 0 代表 Attack
        action_thread.start()
        time.sleep(0.05)  # 動作延遲 (根據需要調整)

        # 更新狀態
        # self.previous_state = self.state
        # self.state = self.get_current_state()
        
        # 計算獎勵
        reward = self.calculate_reward()

        # 更新是否結束
        self.done = self.check_done()

        # 遞增步數
        self.step_count += 1

        return reward, self.done

    def calculate_reward(self):
        """
        計算獎勵
        """
        reward = 0
        # 示例：根據健康值變化計算獎勵
        health_diff = self.health - self.get_health()
        if health_diff < 0:
            reward -= 10  # 損失健康值，給負獎勵
        boss_health_diff = self.get_boss_health() - self.boss_health
        if(boss_health_diff < 0):
            reward += 30
        # 示例：根據分數增長計算獎勵
        # 更新當前健康值和分數
        self.health = self.get_health()
        self.boss_health = self.get_boss_health()
        return reward

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
        if(now_hp <= 0 and  self.first_attacked):
            now_hp = 0
        else :
            now_hp = self.boss_health
        if(now_hp != self.boss_health):
            self.first_attacked = True
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
