import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Left — class balance donut
sizes = [28, 68]
colors = ['#0891B2', '#E2E8F0']
wedges, texts, autotexts = axes[0].pie(
    sizes, labels=['Recurrent (R)', 'Non-Recurrent (N)'],
    colors=colors, autopct='%1.1f%%',
    startangle=90, wedgeprops=dict(width=0.5)
)
axes[0].set_title('Recurrence Class Balance\n(96 breast cancer samples)', 
                   fontsize=12, fontweight='bold', color='#1A2B5E')

# Right — sample overview bar
groups = ['Breast Cancer\n(n=96)', 'Healthy Control\n(n=32)']
recurrent = [28, 0]
non_recurrent = [68, 0]
normal = [0, 32]

x = [0, 1]
axes[1].bar(x, recurrent, color='#0891B2', label='Recurrent')
axes[1].bar(x, non_recurrent, bottom=recurrent, color='#0D9488', label='Non-Recurrent')
axes[1].bar(x, normal, color='#94A3B8', label='Healthy Control')
axes[1].set_xticks(x)
axes[1].set_xticklabels(groups, fontsize=11)
axes[1].set_ylabel('Number of Samples', fontsize=11)
axes[1].set_title('Dataset Overview', fontsize=12, 
                   fontweight='bold', color='#1A2B5E')
axes[1].legend()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('PNAS 2019 SILVER-Seq Dataset (Zhou et al.)', 
             fontsize=10, color='#475569', y=0.02)
plt.tight_layout()
plt.savefig('/Users/vrisha/Desktop/BENG 203 project/slide5_data_overview.png', 
            dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved!")
