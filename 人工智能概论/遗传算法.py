import math
import random

class GeneticAlgorithm(object):

    chrNum = 10         # 染色体数目
    ipop = []           # 种群
    generation = 0      # 染色体代号
    GENE = 46           # 基因数
    bestFitness = pow(3.4028234663852886*10, 38)        # 函数最优解
    bestGenerations = 0                                 # 所有子代与父代中最好的染色体
    bestStr = []                                        # 最优解的染色体的二进制码

    # 初始化一条染色体(二进制字符串)
    def initChr(self):
        res = ""
        for i in range(self.GENE):
            if (random.random() > 0.5):
                res += "0"
            else:
                res += "1"
        return res

    # 初始化一个种群(10条染色体)
    def initPop(self):
        ipop_temp = []
        for i in range(self.chrNum):
            ipop_temp.append(self.initChr())
        return ipop_temp

    # 将染色体转换成x,y变量的值
    def calculateFitnessvalue(self, str):
        # 二进制数前23位为x的二进制字符串，后23位为y的二进制字符串
        a = int(str[0:23], 2)
        b = int(str[23:46], 2)
        x = a * (6.0 - 0) / (pow(2, 23) - 1)  # x的基因
        y = b * (6.0 - 0) / (pow(2, 23) - 1)  # y的基因
        # 需要优化的函数
        fitness = 3 - math.sin(2 * x) * math.sin(2 * x) - math.sin(2 * y) * math.sin(2 * y)
        returns = [x, y, fitness]
        return returns

    '''
    轮盘选择
    计算群体上每个个体的适应度值;
    按由个体适应度值所决定的某个规则选择将进入下一代的个体;
    '''
    def select(self):
        evals = []      # 所有染色体适应值
        p = []          # 各染色体选择概率
        q = []          # 累计概率
        F = 0           # 累计适应值总和
        for i in range(self.chrNum):
            evals.append(self.calculateFitnessvalue(self.ipop[i])[2])
            if evals[i] < self.bestFitness:             # 记录种群中的最小值（最优解）
                self.bestFitness = evals[i]
                self.bestGenerations = self.generation
                self.bestStr = self.ipop[i]
            F = F + evals[i]                            # 所有染色体适应值总和
        for i in range(self.chrNum):
            p.append(evals[i] / F)
            if i == 0:
                q.append(p[i])
            else:
                q.append(q[i-1]+p[i])
        for i in range(self.chrNum):
            r = random.random()
            if r <= q[0]:
                self.ipop[i] = self.ipop[0]
            else:
                for j in range(1, self.chrNum):
                    if r < q[j]:
                        self.ipop[i] = self.ipop[j]

    '''
    交叉操作
    交叉率为60%，平均为60%的染色体进行交叉
    '''
    def cross(self):
        for i in range(self.chrNum):
            if random.random() < 0.60:
                pos = int(random.random()*self.GENE + 1)           # pos位点前后二进制串交叉

                temp1 = self.ipop[i][0:pos] + self.ipop[(i+1) % self.chrNum][pos:]
                temp2 = self.ipop[(i + 1) % self.chrNum][0:pos] + self.ipop[i][pos:]

                self.ipop[i] = temp1
                self.ipop[(i+1) // self.chrNum] = temp2

    '''
    基因突变
    1%基因变异
    '''
    def mutation(self):
        for i in range(0, 4):
            num = int(random.random() * self.GENE * self.chrNum + 1)
            chromosomeNum = int(num / self.GENE) + 1                    # 染色体号
            mutationNum = num - (chromosomeNum - 1) * self.GENE         # 基因号
            if mutationNum == 0:
                mutationNum = 1
            chromosomeNum = chromosomeNum - 1
            if chromosomeNum >= self.chrNum:
                chromosomeNum = 9
            a = ""                                                      # 记录变异位点变异后的编码
            temp = ""

            if self.ipop[chromosomeNum][mutationNum - 1] == '0':        # 当变异位点为0时
                a = '1'
            else:
                a = '0'
            if mutationNum == 1:                                        # 当变异位点在首、中段和尾时的突变情况
                temp = a + self.ipop[chromosomeNum][mutationNum:]
            else:
                if mutationNum != self.GENE:
                    temp = self.ipop[chromosomeNum][0:mutationNum-1] + a + self.ipop[chromosomeNum][mutationNum:]
                else:
                    temp = self.ipop[chromosomeNum][0:mutationNum-1] + a
            self.ipop[chromosomeNum] = temp                             # 记录下变异后的染色体


if __name__ == '__main__':
    GA = GeneticAlgorithm()
    GA.ipop = GA.initPop()          # 产生初始种群
    string = ""

    for i in range(100000):         # 迭代次数
        GA.select()
        GA.cross()
        GA.mutation()
        GA.generation = i

    x = GA.calculateFitnessvalue(GA.bestStr)
    string = "最小值" + str(GA.bestFitness) + '\n' + "第" + str(GA.bestGenerations) + "个染色体:<" + str(GA.bestStr) + ">" + '\n' + "x=" + str(x[0]) + '\n' + "y=" + str(x[1])
    print(string)
