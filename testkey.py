import matplotlib.pyplot as plt

wins = 83
losses = 100 - wins
labels = ['Win', 'Loss']
sizes = [wins, losses]

fig = plt.figure(figsize=(5,5))
def autopct_format(pct):
    total = sum(sizes)
    count = int(round(pct*total/100.0))
    return f'{pct:.1f}%\n({count})'

plt.pie(sizes, labels=labels, autopct=autopct_format, startangle=90)
plt.title('Game Win Rate Pie Chart (100 Games: 83 Wins / 17 Losses)')
plt.axis('equal')

# Save the image for download
output_path = 'winrate_pie.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')

plt.show()

output_path
print()