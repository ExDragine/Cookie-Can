import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
Data_Raw = pd.read_excel('data.xlsx')

Year = np.array(Data_Raw.Year, dtype='int64')
Power_Add = np.array(Data_Raw.Power_Add, dtype='int64')
Sell_Add = np.array(Data_Raw.Elc_Add, dtype='int64')
GDP_Add = np.array(Data_Raw.GDP_Add, dtype='int64')
Power = np.array(Data_Raw.Power, dtype='float64')
Election = np.array(Data_Raw.Elc, dtype='float64')

Value = [Year, Power_Add, Sell_Add, GDP_Add, Power, Election]
font = {'family': 'SimHei', 'weight': 'bold', 'size': '12'}
plt.rc('font', **font)
plt.rcParams['axes.unicode_minus'] = False
fig, axs = plt.subplots(2, 3)

axs[0][0].plot(Year, Power_Add)
axs[0][0].set_ylabel('能源增长率')
axs[0][0].set_xlabel('年份')
axs[0][0].grid(True)

axs[0][1].plot(Year, Sell_Add)
axs[0][1].set_ylabel('电力增长率')
axs[0][1].set_xlabel('年份')
axs[0][1].grid(True)

axs[1][2].plot(Year, GDP_Add)
axs[1][2].set_ylabel('GDP增长率')
axs[1][2].set_xlabel('年份')
axs[1][2].grid(True)

axs[1][0].plot(Year, Power)
axs[1][0].set_ylabel('能源增幅比')
axs[1][0].set_xlabel('年份')
axs[1][0].grid(True)

axs[1][1].plot(Year, Election)
axs[1][1].set_ylabel('能源增幅比')
axs[1][1].set_xlabel('年份')
axs[1][1].grid(True)

fig.tight_layout()
plt.show()