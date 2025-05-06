from Tool.screngrap import screngrap
from Tool.action import  take_action
from hollowknight_env import HollowKnightEnv
from Tool.action import restart
from Tool import framebuffer
frame_buffer = framebuffer.FrameBuffer(windows_name="HOLLOW KNIGHT", buffer_size=1, capture_interval=0.02)

frame_buffer.start()
frame_buffer.reset()
frame_buffer.start()
env = HollowKnightEnv()
# take_action(2)
env.boss_health = 635

while True :
    hp = env.get_boss_health()
    print(hp)
