import pandas as pd
import matplotlib.pyplot as plt

# 讀取一行逗號分隔的數字
df = pd.read_csv("all_rewards.csv", header=None, delimiter=",")

# 展平成單欄
df = df.melt(value_name="reward")["reward"].dropna().astype(float).reset_index(drop=True)

# 計算每 10 個取平均
df = pd.DataFrame(df, columns=["reward"])
df["reward_avg10"] = df["reward"].rolling(window=50).mean()

# 繪圖
plt.figure(figsize=(12,6))
plt.plot(df.index, df["reward_avg10"], label="10-step average reward")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("DQN Rewards (10-step Moving Average)")
plt.legend()
plt.grid(True)
plt.show()
print()
