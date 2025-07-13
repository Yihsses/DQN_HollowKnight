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
Space =  0x39  
up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD

esc = 0x01



def Nothing():
    sendkey.ReleaseKey(A)
    sendkey.ReleaseKey(D)
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
    Nothing()
    sendkey.PressKey(A)
    time.sleep(0.01)
# 1
def Move_Right():
    Nothing()
    sendkey.PressKey(D)
    time.sleep(0.01)

def Attack():
    sendkey.PressKey(J)
    time.sleep(0.15)
    sendkey.ReleaseKey(J)
    Nothing()
    time.sleep(0.01)

def Attack_Up():
    # print("Attack up--->")
    sendkey.PressKey(W)
    sendkey.PressKey(J)
    time.sleep(0.11)
    sendkey.ReleaseKey(W)
    sendkey.ReleaseKey(J)
    Nothing()
    time.sleep(0.01)
def Short_Jump():
    sendkey.PressKey(Space)
    sendkey.PressKey(S)
    sendkey.PressKey(J)
    time.sleep(0.2) 
    sendkey.ReleaseKey(J)
    sendkey.ReleaseKey(S)
    sendkey.PressKey(Space)
    Nothing()
def Skill_Up():
    sendkey.PressKey(W)
    sendkey.PressKey(F)
    sendkey.PressKey(J)
    time.sleep(0.15)
    sendkey.ReleaseKey(W)
    sendkey.ReleaseKey(F)
    sendkey.ReleaseKey(J)
    Nothing()
    time.sleep(0.15)
# 5
def Skill_Down():
    sendkey.PressKey(S)
    sendkey.PressKey(F)
    sendkey.PressKey(J)
    time.sleep(0.2)
    sendkey.ReleaseKey(S)
    sendkey.ReleaseKey(F)
    sendkey.ReleaseKey(J)
    Nothing()
    time.sleep(0.3)
# 3
def Mid_Jump():
    sendkey.PressKey(Space)
    time.sleep(0.2)
    sendkey.PressKey(J)
    time.sleep(0.2)
    sendkey.ReleaseKey(J)
    sendkey.ReleaseKey(Space)
    Nothing()

def Rush():
    sendkey.PressKey(K)
    time.sleep(0.15)
    sendkey.ReleaseKey(K)
    time.sleep(0.05)

def restart():
    Nothing()
    sendkey.ReleaseKey(Space)
    time.sleep(0.3)
    sendkey.PressKey(Space)
    time.sleep(0.1)
    sendkey.ReleaseKey(Space)
    print("按下空白")
    time.sleep(3)
    sendkey.PressKey(W)
    print("W")
    time.sleep(0.3)
    sendkey.ReleaseKey(W)
    time.sleep(2)
    sendkey.PressKey(Space)
    print("按下空白")

    time.sleep(0.4)
    sendkey.ReleaseKey(Space)
    time.sleep(3)
    print("成功進入")


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
Actions = [Attack,Attack_Up,Short_Jump, Mid_Jump,Skill_Up,Skill_Down,Rush]
Directions = [Move_Left, Move_Right,Turn_Left, Turn_Right] 

jump = [Nothing,Mid_Jump]

def take_action(action):
    Actions[action]()

def take_direction(direc):
    Directions[direc]()


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