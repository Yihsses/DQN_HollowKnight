from Tool.screngrap import screngrap
while True :
    hp = screngrap.grap_Boss_hp("Hollow Knight")
    # 60
    now_hp = 0
    for i in range(6,hp[16].shape[0]):
        if(hp[20][i][2] >= 234):
            now_hp+=1
    print(now_hp)
