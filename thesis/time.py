import matplotlib as matplotlib
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

color_pool = ['#C82423', '#9AC9DB', '#8E44AD', '#14517C', '#EC7063', '#006400', '#999999' ]
marker_pool = ['o', '^', 'h', 'v', '*', '<', '>', 's']


title = "time"


fontsize = 20
font2 = {'family': 'Times New Roman', 'size': 20}
font = {'family': 'SimSun', 'size': 14}

fig = plt.figure(figsize=(10,3.5))
# fig.set_size_inches(24, 3.5)
plt.ylim(0, 400)
plt.tick_params(labelsize=fontsize)
x_ticks = ['WD','DBLP_1','DBLP_2']
labels=['EFC-UIL','MAUIL','Grad-Align','CENALP','GAlign','DeepLink']

y1 = [36,18,24]
y2 = [45,24,30]
y3 = [165,48,48]
y4 = [360,255,290]
y5 = [11,9,10]
y6 = [12,24,14]



xth = np.arange(len(x_ticks))
plt.tick_params(labelsize=14)
width=0.8


bar0 = plt.bar(6*xth,y1,width,color = 'white', facecolor='#C82423', edgecolor='#000000')
bar1 = plt.bar(6*xth+width,y2,width,color = 'white', facecolor='#9AC9DB', edgecolor='#000000')
bar2 = plt.bar(6*xth+2*width,y3,width,color = 'white', facecolor='#8E44AD', edgecolor='#000000')
bar3 = plt.bar(6*xth+3*width,y4,width,color = 'white', facecolor='#14517C', edgecolor='#000000')
bar4 = plt.bar(6*xth+4*width,y5,width,color = 'white', facecolor='#EC7063', edgecolor='#000000')
bar5 = plt.bar(6*xth+5*width,y6,width,color = 'white', facecolor='#006400', edgecolor='#000000')




plt.ylabel('Time(min)',font)

plt.xticks(6*xth+3,labels=x_ticks,rotation=0)
plt.legend(labels,loc='upper right',ncol=8,bbox_to_anchor=(1.15,1.19), prop={'size':12})
plt.grid(ls='--')

# y = [0.511, 0.542,  0.557, 0.720]
# ax.set_ylabel(r'$Time(hour)$', size=fontsize)
# ax.axes.xaxis.set_ticklabels([])
# # y_ticks = np.arange(0.4,1,0.2)
# # ax.set_yticks(y_ticks)
# ax.bar(range(len(y)), y, color = ['#9AC9DB', '#2878B5', '#EC7063','#C82423', '#8E44AD', '#999999',  '#14517C','#F8AC8C'])
# # ax.set_title('(a) Douban-Weibo.', font2, y=-0.15)


# ax.bar(0,0.179108999, color = '#9AC9DB', label= 'DHNA-1')
# ax.bar(1, 0.28702052, color = '#2878B5', label= 'DHNA-2')
# ax.bar(2,0.152990382, color = '#EC7063', label='DHNA-3' )
# ax.bar(3,0.4241097327 , color ='#C82423' , label= 'DHNA')
# ax.legend(bbox_to_anchor=(0.5,1.2),borderaxespad = 0., prop={'size':16}, ncol=4, loc='upper center')


#fig.legend(labels, loc = 'upper center', borderaxespad=-0., prop={'size':fontsize}, ncol=8)
#fig.tight_layout()
# plt.subplots_adjust(right=0.9, top=0.9)#调整子图间距
plt.savefig(title + '.png', format='png',dpi=200, bbox_inches='tight')
