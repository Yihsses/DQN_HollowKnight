# Define the actions we may need during training
# You can define your actions here
from Tool import sendkey
import time
import cv2
import threading

# Hash code for key we may use: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes?redirectedfrom=MSDN
UP_ARROW = 0x26
DOWN_ARROW = 0x28
LEFT_ARROW = 0x25
RIGHT_ARROW = 0x27

L_SHIFT = 0xA0
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

M = 0x32
J = 0x24
K = 0x25
LSHIFT = 0x2A
R = 0x13#用R代替识破
V = 0x2F

Q = 0x10
I = 0x17
O = 0x18
P = 0x19
C = 0x2E
F = 0x21
Space =  0x20
up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD

esc = 0x01

# move actions
# 0
def Nothing():

    pass

# Move
# 0
def Turn_Left():
    sendkey.PressKey(A)
    sendkey.ReleaseKey(A)
# 1
def Turn_Right():
    sendkey.PressKey(D)
    sendkey.ReleaseKey(D)

def Move_Left():
    sendkey.PressKey(A)
    time.sleep(0.01)
    sendkey.ReleaseKey(A)
# 1
def Move_Right():
    sendkey.PressKey(D)
    time.sleep(0.01)
    sendkey.ReleaseKey(D)

def Attack():
    sendkey.PressKey(J)
    time.sleep(0.15)
    sendkey.ReleaseKey(J)
    Nothing()
    time.sleep(0.01)
# 1
# def Attack_Down():
#     PressKey(DOWN_ARROW)
#     PressKey(X)
#     time.sleep(0.05)
#     ReleaseKey(X)
#     ReleaseKey(DOWN_ARROW)
#     time.sleep(0.01)
# 1
#JUMP
# 2
def Short_Jump():
    sendkey.PressKey(Space)
    time.sleep(0.2) 
    sendkey.ReleaseKey(Space)
    Nothing()
# 3
def Mid_Jump():
    sendkey.PressKey(Space)
    time.sleep(0.2)
    sendkey.ReleaseKey(Space)
    sendkey.PressKey(Space)
    time.sleep(0.2)
    sendkey.ReleaseKey(Space)
    Nothing()


# Skill
# 4
# def Skill():
#     PressKey(Z)
#     PressKey(X)
#     time.sleep(0.1)
#     ReleaseKey(Z)
#     ReleaseKey(X)
#     time.sleep(0.01)
# 4
# def Skill_Up():
#     PressKey(UP_ARROW)
#     PressKey(Z)
#     PressKey(X)
#     time.sleep(0.15)
#     ReleaseKey(UP_ARROW)
#     ReleaseKey(Z)
#     ReleaseKey(X)
#     Nothing()
#     time.sleep(0.15)
# # 5
# def Skill_Down():
#     PressKey(DOWN_ARROW)
#     PressKey(Z)
#     PressKey(X)
#     time.sleep(0.2)
#     ReleaseKey(X)
#     ReleaseKey(DOWN_ARROW)
#     ReleaseKey(Z)
#     Nothing()
#     time.sleep(0.3)


# # Rush
# # 6
# def Rush():
#     PressKey(L_SHIFT)
#     time.sleep(0.1)
#     ReleaseKey(L_SHIFT)
#     Nothing()
#     PressKey(X)
#     time.sleep(0.03)
#     ReleaseKey(X)
# # Cure
# def Cure():
#     PressKey(A)
#     time.sleep(1.4)
#     ReleaseKey(A)
#     time.sleep(0.1)


# Restart function
# it restart a new game
# it is not in actions space
# def Look_up():
#     PressKey(UP_ARROW)
#     time.sleep(0.1)
#     ReleaseKey(UP_ARROW)

# def restart():
#     station_size = (230, 230, 1670, 930)
#     while True:
#         station = cv2.resize(cv2.cvtColor(grab_screen(station_size), cv2.COLOR_RGBA2RGB),(1000,500))
#         if station[187][300][0] != 0: 
#             time.sleep(1)
#         else:
#             break
#     time.sleep(1)
#     Look_up()
#     time.sleep(1.5)
#     Look_up()
#     time.sleep(1)
#     while True:
#         station = cv2.resize(cv2.cvtColor(grab_screen(station_size), cv2.COLOR_RGBA2RGB),(1000,500))
#             # PressKey(DOWN_ARROW)8
#             # time.sleep(0.1)8
#             # ReleaseKey(DOWN_ARROW)
#         PressKey(Z)
#         time.sleep(0.1)
#         ReleaseKey(Z)
#         break


# List for action functions
Actions = [Attack,Short_Jump, Mid_Jump ,Move_Left, Move_Right, Turn_Left, Turn_Right]
# Directions = [Move_Left, Move_Right, Turn_Left, Turn_Right]
# Run the action
def take_action(action):
    Actions[action]()

# def take_direction(direc):
#     Directions[direc]()



class TackAction(threading.Thread):
    def __init__(self, threadID, name, direction, action):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.direction = direction
        self.action = action
        
    def run(self):
        # take_direction(self.direction)
        take_action(self.action)