'''
Descripttion: 
version: 
Author: Yanjun Hao
Date: 2023-03-28 11:07:01
LastEditors: Yanjun Hao
LastEditTime: 2023-03-28 11:23:27
'''
import torch
from torch.utils.data import DataLoader,Dataset
import os
import re
from random import sample
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
 
# 路径需要根据情况修改，要看你把数据下载到哪里了
# 数据下载地址在斯坦福官网，网上搜索就有
data_base_path = r"./imdb_test/aclImdb"

# 这个里面是存储你训练出来的模型的，现在是空的
model_path = r"./imdb_test/aclImdb/mode"
        
#1. 准备dataset，这里写了一个数据读取的类，并把数据按照不同的需要进行了分类；
class ImdbDataset(Dataset):
    def __init__(self,mode,testNumber=10000,validNumber=5000):

        # 在这里我做了设置，把数据集分成三种形式，可以选择 “train”默认返回全量50000个数据，“test”默认随机返回10000个数据，
        # 如果是选择“valid”模式，随机返回相应数据
        super(ImdbDataset,self).__init__()

        # 读取所有的训练文件夹名称
        text_path =  [os.path.join(data_base_path,i)  for i in ["test/neg","test/pos"]]
        text_path.extend([os.path.join(data_base_path,i)  for i in ["train/neg","train/pos"]])

        if mode=="train":
            self.total_file_path_list = []
            # 获取训练的全量数据，因为50000个好像也不算大，就没设置返回量，后续做sentence的时候再做处理
            for i in text_path:
                self.total_file_path_list.extend([os.path.join(i,j) for j in os.listdir(i)])
        if mode=="test":
            self.total_file_path_list = []
            # 获取测试数据集，默认10000个数据
            for i in text_path:
                self.total_file_path_list.extend([os.path.join(i,j) for j in os.listdir(i)])
            self.total_file_path_list=sample(self.total_file_path_list,testNumber)
       
        if mode=="valid":
            self.total_file_path_list = []
            # 获取验证数据集，默认5000个数据集
            for i in text_path:
                self.total_file_path_list.extend([os.path.join(i,j) for j in os.listdir(i)])
            self.total_file_path_list=sample(self.total_file_path_list,validNumber)
   
    def tokenize(self,text):
    
        # 具体要过滤掉哪些字符要看你的文本质量如何
       
        # 这里定义了一个过滤器，主要是去掉一些没用的无意义字符，标点符号，html字符啥的
        fileters = ['!','"','#','$','%','&','\(','\)','\*','\+',',','-','\.','/',':',';','<','=','>','\?','@'
            ,'\[','\\','\]','^','_','`','\{','\|','\}','~','\t','\n','\x97','\x96','”','“',]
        # sub方法是替换
        text = re.sub("<.*?>"," ",text,flags=re.S)	# 去掉<...>中间的内容，主要是文本内容中存在<br/>等内容
        text = re.sub("|".join(fileters)," ",text,flags=re.S)	# 替换掉特殊字符，'|'是把所有要匹配的特殊字符连在一起
        return text	# 返回文本

    def __getitem__(self, idx):
        cur_path = self.total_file_path_list[idx]
		# 返回path最后的文件名。如果path以／或\结尾，那么就会返回空值。即os.path.split(path)的第二个元素。
        # cur_filename返回的是如：“0_3.txt”的文件名
        cur_filename = os.path.basename(cur_path)
        # 标题的形式是：3_4.txt	前面的3是索引，后面的4是分类
        # 如果是小于等于5分的，是负面评论，labei给值维1，否则就是1
        labels = []
        sentences = []
        if int(cur_filename.split("_")[-1].split(".")[0]) <= 5 :
            label = 0
        else:
            label = 1
        # temp.append([label])
        labels.append(label)
        text = self.tokenize(open(cur_path,encoding='UTF-8').read().strip()) #处理文本中的奇怪符号
        sentences.append(text)
        # 可见我们这里返回了一个list，这个list的第一个值是标签0或者1，第二个值是这句话；
        return sentences,labels
 
    def __len__(self):
        return len(self.total_file_path_list)
    
# 2. 这里开始利用huggingface搭建网络模型
# 这个类继承再nn.module,后续再详细介绍这个模块
# 
class BertClassificationModel(nn.Module):
    def __init__(self,hidden_size=768):
        super(BertClassificationModel, self).__init__()
        # 这里用了一个简化版本的bert
        model_name = 'distilbert-base-uncased'

        # 读取分词器
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        
        # 读取预训练模型
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)

        for p in self.bert.parameters(): # 冻结bert参数
                p.requires_grad = False
        self.fc = nn.Linear(hidden_size,2)

    def forward(self, batch_sentences):   # [batch_size,1]
        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=512,
                                             add_special_tokens=True)
        input_ids=torch.tensor(sentences_tokenizer['input_ids']) # 变量
        attention_mask=torch.tensor(sentences_tokenizer['attention_mask']) # 变量
        bert_out=self.bert(input_ids=input_ids,attention_mask=attention_mask) # 模型

        last_hidden_state =bert_out[0] # [batch_size, sequence_length, hidden_size] # 变量
        bert_cls_hidden_state=last_hidden_state[:,0,:] # 变量
        fc_out=self.fc(bert_cls_hidden_state) # 模型
        return fc_out

# 3. 程序入口，模型也搞完啦，我们可以开始训练，并验证模型的可用性

def main():

    testNumber = 10000    # 多少个数据参与训练模型
    validNumber = 100   # 多少个数据参与验证
    batchsize = 250  # 定义每次放多少个数据参加训练
    
    trainDatas = ImdbDataset(mode="test",testNumber=testNumber) # 加载训练集,全量加载，考虑到我的破机器，先加载个100试试吧
    validDatas = ImdbDataset(mode="valid",validNumber=validNumber) # 加载训练集

    train_loader = torch.utils.data.DataLoader(trainDatas, batch_size=batchsize, shuffle=False)#遍历train_dataloader 每次返回batch_size条数据

    val_loader = torch.utils.data.DataLoader(validDatas, batch_size=batchsize, shuffle=False)

    # 这里搭建训练循环，输出训练结果

    epoch_num = 1  # 设置循环多少次训练，可根据模型计算情况做调整，如果模型陷入了局部最优，那么循环多少次也没啥用

    print('training...(约1 hour(CPU))')
    
    # 初始化模型
    model=BertClassificationModel()
  
    optimizer = optim.AdamW(model.parameters(), lr=5e-5) # 首先定义优化器，这里用的AdamW，lr是学习率，因为bert用的就是这个

    # 这里是定义损失函数，交叉熵损失函数比较常用解决分类问题
    # 依据你解决什么问题，选择什么样的损失函数
    criterion = nn.CrossEntropyLoss()
    
    print("模型数据已经加载完成,现在开始模型训练。")
    for epoch in range(epoch_num):
        for i, (data,labels) in enumerate(train_loader, 0):

            output = model(data[0])
            optimizer.zero_grad()  # 梯度清0
            loss = criterion(output, labels[0])  # 计算误差
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 打印一下每一次数据扔进去学习的进展
            print('batch:%d loss:%.5f' % (i, loss.item()))

        # 打印一下每个epoch的深度学习的进展i
        print('epoch:%d loss:%.5f' % (epoch, loss.item()))
    
    #下面开始测试模型是不是好用哈
    print('testing...(约2000秒(CPU))')

    # 这里载入验证模型，他把数据放进去拿输出和输入比较，然后除以总数计算准确率
    # 鉴于这个模型非常简单，就只用了准确率这一个参数，没有考虑混淆矩阵这些
    num = 0
    model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化,主要是在测试场景下使用；
    for j, (data,labels) in enumerate(val_loader, 0):

        output = model(data[0])
        # print(output)
        out = output.argmax(dim=1)
        # print(out)
        # print(labels[0])
        num += (out == labels[0]).sum().item()
        # total += len(labels)
    print('Accuracy:', num / validNumber)

if __name__ == '__main__':
    main()

