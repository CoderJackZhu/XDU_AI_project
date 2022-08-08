from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
#导入iris数据 0-49为第一类 50

iris_df = datasets.load_iris()

#x_1= iris_df.data[:,0]
#x_2= iris_df.data[:,1]
#x_3= iris_df.data[:,2]
#x_4= iris_df.data[:,3]

iris_1=iris_df.data[0:50,0:4]
iris_2=iris_df.data[50:100,0:4]
iris_3=iris_df.data[100:150,0:4]
c_class=iris_df.target


#第一类和第二类
#分成训练集和测试集比例为3:2
target1=np.zeros([50,1])
target2=np.ones([50,1])
iris1_train, iris1_test, target1_train, target1_test = train_test_split(iris_1, target1, test_size=0.4, random_state=1564)
iris2_train, iris2_test, target2_train, target2_test = train_test_split(iris_2, target2, test_size=0.4, random_state=14)
#均值μi
iris_1_mean=np.mean(iris1_train,axis=0)
iris_2_mean=np.mean(iris2_train,axis=0)
#类内离散度矩阵Si
s1=np.zeros([4,4])
s2=np.zeros([4,4])
for i in range(0,30,1):
    a1=iris1_train[i,:]-iris_1_mean
#    a1_T=np.array([a1]).T
    a2=iris2_train[i,:]-iris_2_mean
#    a2_T=np.array([a2]).T
#    s1=s1+np.dot(a1_T,a1_T.T)
#    s2=s2+np.dot(a2_T,a2_T.T)
    g1=np.array(a1*np.array([a1]).T)
    s1=s1+g1
    g2=np.array(a2*np.array([a2]).T)
    s2=s2+g2
    
#类内总离散度矩阵Sw
sw=s1+s2
#样本类间离散度矩阵Sb
mu1=np.array(iris_1_mean)
mu2=np.array(iris_2_mean)
d=np.array([mu1-mu2])
sb=np.dot(d.T,d)

#计算方向
w=(np.dot(np.linalg.inv(sw),d.T)).T
iris1_y0_train=np.zeros([30,1]) #y0
iris2_y0_train=np.zeros([30,1])
#计算μ1 μ2 得到分类点y0
for i in range(30):
    iris1_y0_train[i]=np.dot(w,iris1_train[i,:])
    iris2_y0_train[i]=np.dot(w,iris2_train[i,:])
y0_mu1=np.mean(iris1_y0_train)
y0_mu2=np.mean(iris2_y0_train)

y0=(y0_mu1+y0_mu2)/2

#计算测试集准确率
correct1=0
correct2=0
iris1_y0_target=np.zeros([20,1]) #测试集上的目标值
iris2_y0_target=np.zeros([20,1])
iris1_y0_test=np.zeros([20,1])   #转化到一维的值
iris2_y0_test=np.zeros([20,1])

for i in range(0,20,1):
    iris1_y0_test[i]=np.dot(w,np.array(iris1_test[i,:]))
    if iris1_y0_test[i]>=y0:
        iris1_y0_target[i]=0
    else:
        iris1_y0_target[i]=1
    #判断是否正确
    if iris1_y0_target[i]==target1_test[i,:]:
        correct1=correct1+1
    else:
        continue 
    
    iris2_y0_test[i]=np.dot(w,np.array(iris2_test[i,:]))
    if iris2_y0_test[i]<y0:
        iris2_y0_target[i]=1
    else:
        iris2_y0_target[i]=0
    if iris2_y0_target[i]==target2_test[i]:
        correct2=correct2+1
    else:
        continue
    
accuracy=(correct1+correct2)/40
print("0 1类鸢尾花的分类准确率为：{}".format(accuracy))
q=[]
for i in range(20):
    q.append(0)
plt.scatter(iris1_y0_test,q,color='r',label='class0')
plt.scatter(iris2_y0_test,q,color='b',label='class1')
plt.legend(loc="upper left")
#plt.savefig('C:\\Users\\30790\\Desktop\\Fisher\\iris01.png')
plt.show()
