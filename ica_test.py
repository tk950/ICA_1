import random
import numpy as np

from scipy.io.wavfile import read, write
from IPython.display import Audio
from matplotlib import pyplot as plt
import copy


def load_2(mean):
    fs1, data1 = read('speechA1.wav')
    fs2, data2 = read('speechA2.wav')
    x1 = np.copy(data1)
    x1 = x1.astype('float64')
    x2 = np.copy(data2)
    x2 = x2.astype('float64')
    size = x1.size
    x = np.zeros((2, size))
    for i in range(size):
        x[0, i] = data1[i]
        x[1, i] = data2[i]
    y = x.mean(axis=1)
    x[0, :] -= y[0]
    x[1, :] -= y[1]
    tmp = y.tolist()
    mean[1] = tmp[1]
    mean[0] = tmp[0]
    return x


def kyoubunsan(x):#共分散行列を求める
    if x.ndim != 1:
        num = x.shape[0]
        size = x.shape[1]
        y = x.mean(axis=1)
        tmp = np.copy(x)
        for i in range(num):
            tmp[i] -= y[i]
        return np.dot(tmp, tmp.T)/size
    else:
        tmp = np.copy(x)
        y = x.mean()
        x -= y
        return np.dot(tmp, tmp.T)/x.size


def diag_mat(d_tmp):#固有値の-1/2乗を並べた対角行列を返す
    size = d_tmp.size
    d = np.zeros((size, size))
    for i in range(size):
        d[i, i] = d_tmp[i]**(-1/2)
    return d


def maximum(z, num):#ansから呼び出す。最適なwを返す
    norm = 1.
    w = np.zeros(num)
    for i in range(num):#ランダムな初期値
        w[i] = random.uniform(-100, 100)
    w = w/np.linalg.norm(w, ord=2)
    while(norm > 1e-15):#収束すれば終わり
        w_prev = np.copy(w)
        tmp1 = np.dot(w, z)
        tmp2 = np.multiply(tmp1, np.multiply(tmp1, tmp1))
        tmp3 = np.zeros((num, z.shape[1]))
        for i in range(z.shape[1]):
            tmp3[:, i] = z[:, i]*tmp2[i]

        tmp4 = tmp3.mean(axis=1)
        w = tmp4-3*w
        w = w/np.linalg.norm(w, ord=2)
        norm = abs(abs(np.dot(w, w_prev))-1)
    return w


def ans(z):#信号源の数だけmaximumを呼び出す
    num = z.shape[0]
    y = np.zeros((num, z.shape[1]))
    i = 0
    while i<num:
        y[i, :] = np.dot(maximum(z, num), z)
        if i>=1:#以下では、同じ信号源を抽出してしまったときのための処理を行う
            ytmp = y[i,:]
            for j in range(i):
                yprev = y[j,:]
                ycmp1 = yprev-ytmp
                ycmp2 = yprev+ytmp
                if np.linalg.norm(ycmp1,ord=1)<1 or np.linalg.norm(ycmp2,ord=1)<1:
                    i=i-1
                    break
        i=i+1
    return y

x=load_2()

#####白色化と復号処理##########################
sigm = kyoubunsan(x)
sigm_eig = np.linalg.eig(sigm)
d_tmp = sigm_eig[0]
e = sigm_eig[1]
d = diag_mat(d_tmp)
v = np.dot(np.dot(e, d), e.T)
z = np.dot(v, x)
y = ans(z)
################################

y*=5000#####################ICAでは振幅まで再現できないらしいので、とりあえず聞こえる程度まで復元
ans0 = y[0,:]
write('A1.wav',rate=8000,data=ans0.astype('int16'))
ans1 = y[1,:]
write('A2.wav',rate=8000,data=ans1.astype('int16'))
plt.plot(ans1)
plt.plot(ans0,color='red')