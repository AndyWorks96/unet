# _*_ encoding:utf-8 _*_
# 对分割结果画雷达图
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

labels=np.array(['Dice-WT','Dice-TC','Dice-ET','Sensitivity-WT','Sensitivity-TC','Sensitivity-ET', 'Specificity-WT', 'Specificity-TC', 'Specificity-ET'])
nAttr=9
labels=np.concatenate((labels,[labels[0]]))  #对labels进行封闭
data_1=np.array([0.817,0.762,0.744,0.852,0.809,0.774,0.997,0.998,0.999])
Chen = np.array([0.838,0.789,0.796,0.884,0.827,0.836,0.994,0.997,0.996])
Sun =  np.array([0.854,0.808,0.785,0.920,0.846,0.853,0.992,0.996,0.997])
Cheng =np.array([0.887,0.856,0.803,0.914,0.852,0.876,0.99,0.995,0.996])


Zhang = np.array([0.89,0.79,0.78,0.88,0.79,0.80,0.99,0.99,0.99])
Kotowski  = np.array([0.83,0.73,0.68,0.81,0.73,0.73,0.99,0.99,0.99])
Mostefa = np.array([0.84,0.70,0.61,0.84,0.71,0.69,0.99,0.99,1.0])

angles=np.linspace(0, 2*np.pi, nAttr, endpoint=False)
angles=np.concatenate((angles,[angles[0]]))

fig=plt.figure(facecolor="white", figsize=(7.5, 5))
plt.subplot(111,polar=True)


data_1 = np.concatenate((data_1, [data_1[0]]))
plt.plot(angles,data_1,'bo-',color='r',linewidth=1, label='FCN',MarkerSize = 5)
plt.fill(angles,data_1,facecolor='r',alpha=0.2)

Chen = np.concatenate((Chen, [Chen[0]]))
plt.plot(angles,Chen,'*-',color='g',linewidth=1, label='Attention-U-Net',MarkerSize = 5)
plt.fill(angles,Chen,facecolor='g',alpha=0.2)

Sun = np.concatenate((Sun, [Sun[0]]))
plt.plot(angles,Sun,'p-',color='k',linewidth=1, label='NAS-U-Net',MarkerSize = 5)
plt.fill(angles,Sun,facecolor='k',alpha=0.2)

Cheng = np.concatenate((Cheng, [Cheng[0]]))
plt.plot(angles,Cheng,'s-',color='lime',linewidth=1, label='Ours',MarkerSize = 5)
plt.fill(angles,Cheng,facecolor='lime',alpha=0.2)

# Zhang = np.concatenate((Zhang, [Zhang[0]]))
# plt.plot(angles,Zhang,'v-',color='b',linewidth=0.8, label='Reference[32]',MarkerSize = 4)
# plt.fill(angles,Zhang,facecolor='b',alpha=0.2)

# Kotowski = np.concatenate((Kotowski, [Kotowski[0]]))
# plt.plot(angles,Kotowski,'^-',color='yellow',linewidth=0.8, label='Reference[33]',MarkerSize = 4)
# plt.fill(angles,Kotowski,facecolor='yellow',alpha=0.2)

# Mostefa = np.concatenate((Mostefa, [Mostefa[0]]))
# plt.plot(angles,Mostefa,'<-',color='pink',linewidth=0.8, label='Reference[34]',MarkerSize = 4)
# plt.fill(angles,Mostefa,facecolor='pink',alpha=0.5)

plt.thetagrids(angles*180/np.pi,labels)

# plt.figtext(0.52,0.95,'python成绩分析图',ha='center')
plt.ylim(0.5, 1.0)
plt.grid(True)
plt.legend(loc=1, bbox_to_anchor=(1.45, 1.1))
plt.savefig('dota_radar.JPG', dpi=300)
plt.show()


# plt.rc('font', family='Times New Roman')

# labels = np.array(['Dice-WT', 'Dice-TC', 'Dice-ET', 'Sensitivity-WT',
#                    'Sensitivity-TC', 'Sensitivity-ET', 'Specificity-WT', 'Specificity-TC', 'Specificity-ET'])
# nAttr = 9
# data_1 = np.array([0.8548, 0.8457, 0.7847, 0.9194,
#                    0.9071, 0.8795, 0.9901, 0.9900, 0.9945])
# shen = np.array([0.8677, 0.7378, 0.6496, 0.8882,
#                  0.7575, 0.7766, 0.9921, 0.9962, 0.9972])
# guo = np.array([0.8970, 0.8254, 0.7640, 0.9118,
#                 0.8412, 0.7748, 0.9942, 0.9973, 0.9985])
# mob = np.array([0.909, 0.866, 0.711, 0.897, 0.831, 0.771, 0.995, 0.998, 0.998])
# fan = np.array([0.8539, 0.7082, 0.7221, 0.9275,
#                 0.7658, 0.7542, 0.9861, 0.9941, 0.9972])

# angles = np.linspace(0, 2*np.pi, nAttr, endpoint=False)
# angles = np.concatenate((angles, [angles[0]]))

# fig = plt.figure(facecolor="white", figsize=(7.5, 5))
# plt.subplot(111, polar=True)

# data_1 = np.concatenate((data_1, [data_1[0]]))
# plt.plot(angles, data_1, 'bo-', color='r', linewidth=1, label='本文的方法')
# plt.fill(angles, data_1, facecolor='r', alpha=0.2)

# shen = np.concatenate((shen, [shen[0]]))
# plt.plot(angles, shen, 'bo-', color='g', linewidth=1, label='1')
# plt.fill(angles, shen, facecolor='g', alpha=0.2)

# guo = np.concatenate((guo, [guo[0]]))
# plt.plot(angles, guo, 'bo-', color='b', linewidth=1, label='competitor 2')
# plt.fill(angles, guo, facecolor='b', alpha=0.2)

# mob = np.concatenate((mob, [mob[0]]))
# plt.plot(angles, mob, 'bo-', color='lime', linewidth=1, label='competitor 3')
# plt.fill(angles, mob, facecolor='lime', alpha=0.2)

# fan = np.concatenate((fan, [fan[0]]))
# plt.plot(angles, fan, 'bo-', color='k', linewidth=1, label='competitor 4')
# plt.fill(angles, fan, facecolor='k', alpha=0.2)

# plt.thetagrids(angles*180/np.pi, labels)
# # plt.figtext(0.52,0.95,'python成绩分析图',ha='center')
# plt.ylim(0.5, 1.0)
# plt.grid(True)
# plt.legend(loc=1, bbox_to_anchor=(1.45, 1.1))
# plt.savefig('data_radar_english.JPG', dpi=300)
# plt.show()
