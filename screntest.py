from Tool.screngrap import screngrap
from Tool.action import  take_action
from hollowknight_env import HollowKnightEnv
from Tool.action import restart
env = HollowKnightEnv()
# take_action(2)
restart()
# while True :
#     print(env.get_health())
#     # hp = screngrap.grap_Boss_hp("Hollow Knight")
#     # # 60
#     # now_hp = 0
#     # for i in range(6,hp[16].shape[0]):
#     #     if(hp[20][i][2] >= 234):
#     #         now_hp+=1
#     # print(now_hp)
