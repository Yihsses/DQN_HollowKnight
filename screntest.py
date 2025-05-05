from Tool.screngrap import screngrap

hp = screngrap.grap_hp("Hollow Knight")
#60
now_hp = 0
for i in range(23,310,39):
    if(hp[20][i][0] > 190):
        now_hp+=1
print(now_hp)
