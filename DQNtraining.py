# -*- coding: utf-8 -*-
"""
Created on Sat May 15 15:57:32 2021

@author: MSI-NB
"""


import dqnnetwork
from reward import Reward
import numpy as np
import grabscreen
import cv2
import random
import getkeys
import directkey
import time
import copy
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from torch.utils.tensorboard import SummaryWriter   

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                #acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                acc_sum += (net(X.to(device)).argmax(dim=0) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

class DQNmodel():
    
    def __init__(self):
         
        self.main_view = (280,50,1700,1000)
        
        self.episode = 100000
        
        self.image_size = 224
        #self.image_size = 299
        
        #self.architecture = ((1,1,64),(1,64,128),(2,128,256),(2,256,512),(2,512,1024))
        #self.fc_features = 1024*7*7
        #self.fc_unit = 2048
        self.architecture = ((3,64,0,1),(64,64,2,1),(64,128,2,0),(128,128,2,1))
        self.fc_features = 128*28*28
        self.fc_unit = 256
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.minibatch_size = 20
        self.lr = 1e-1
        self.num_epoch = 4
        self.loss = nn.MSELoss()
        self.path = r'./modelsaved/model'
        
        self.action_space = 11
        self.eposilon = 0.7
        
        self.Reward = Reward()
        self.now_reward = 0
        self.gamma = 0.8
        
        self.size_of_replay_buffer = 60
        self.replay_buffer = []
        self.beta = 0.8
        self.sample_possibility = np.array([])
        
        self.dead = 0
        self.emergence = 0
        self.boss_num = 0
        
        return None
    
    def initial_Q_network(self,mode):
        
        
        #self.q_net = dqnnetwork.vggnet(self.architecture, self.fc_features, self.fc_unit, self.action_space).to(self.device)
        #self.q_net = dqnnetwork.resnet18(self.architecture, self.fc_features, self.fc_unit, self.action_space).to(self.device).half()
        self.q_net = dqnnetwork.Resnet_DQN(self.architecture, self.fc_features, self.fc_unit, self.action_space).cuda()
        #self.fixed_q_net = dqnnetwork.resnet18(self.architecture, self.fc_features, self.fc_unit, self.action_space).to('cpu')
        self.fixed_q_net = dqnnetwork.Resnet_DQN(self.architecture, self.fc_features, self.fc_unit, self.action_space).to('cpu')
        #self.q_net = dqnnetwork.Googlenetv4().cuda()
        #self.fixed_q_net = dqnnetwork.Googlenetv4()
        if mode==1:
            self.load_model('276')
        else:
            self.fixed_q_net.load_state_dict(self.q_net.state_dict())
        self.state_dict_len = len(self.q_net.state_dict())
        return None
    
    def action_taken_policy_eposilon_greedy(self):
        
        #这里decision = 1时，会选用网络的output
        
        decision = np.random.choice([0,1],p = [1-self.eposilon,self.eposilon])
        
        #screen_gray = cv2.cvtColor(grabscreen.grab_screen(self.main_view),cv2.COLOR_BGR2GRAY)#灰度图像收集
        #screen_reshape = cv2.resize(screen_gray,(self.image_size,self.image_size))
        
        screen_gray = cv2.cvtColor(grabscreen.grab_screen(self.main_view),cv2.COLOR_BGR2RGB)
        screen_reshape = torch.Tensor([cv2.resize(screen_gray,(self.image_size,self.image_size)).transpose(2,1,0)])
        
        '''
        if decision == 1:    
            decision = self.fixed_q_net(torch.Tensor([[screen_reshape]])).argmax()   
        else:
            print("greedy policy")
            decision = np.random.choice([i for i in range(self.action_space)])
        '''   
        #decision = self.q_net(torch.Tensor([[screen_reshape]]).cuda())
        decision = self.q_net(screen_reshape.cuda())
        
        print("defence:{0:.3},attack:{1:.3},go forward:{2:.3},go back:{3:.3},forward_jump:{4:.3},left:{5:.3},right:{6:.3},jump:{7:.3},doge:{8:.3},fastgo:{9:.3},hook:{10:.3}"
              .format(decision[0][0],decision[0][1],decision[0][2],decision[0][3],decision[0][4],decision[0][5],decision[0][6],decision[0][7],decision[0][8],decision[0][9],decision[0][10]))
        
        final_decision = decision.argmax()
        
        sekiro_blood_at_state_1 = self.Reward.get_sekiro_blood_siuation()
        if sekiro_blood_at_state_1 <= 53 or sekiro_blood_at_state_1 >= 236:
            self.dead = 1
            return screen_reshape,final_decision
        
        self.movement(final_decision)
        time.sleep(0.3)
        
        return screen_reshape,final_decision
    
    def trainDQN(self,train_iter, net, optimizer, device, num_epochs):
        net = net.to(device)
        print("training on ", device)
        batch_count = 0
        flag = 0
        for epoch in range(num_epochs):
            if flag ==1:
                break
            train_l_sum, n, start = 0.0, 0, time.time()
            for X, y in train_iter:
                X = X.to(device)
                self.writer.add_image('image',X[0], global_step=None, walltime=None, dataformats='CHW')
                y = y.to(device)
                y_hat = net(X)
                
                """def map_func(action_space,y_hat):
                    
                    index = y_hat.argmax(axis =1)
                    
                    max_value = y_hat[0].max()
                    vec = np.zeros(action_space)
                    vec[index[0]] = max_value
                    label_array = np.array(vec)
                    
                    for i in range(1,len(y_hat)):
                        
                        max_value = y_hat[i].max()
                        vec = np.zeros(action_space)
                        vec[index[i]] = max_value
                        
                        label_array = np.vstack((label_array,vec))
                        
                    return label_array
                
                y_true = torch.Tensor(map_func(self.action_space,y_hat)).cuda()"""
              
                l = self.loss(y_hat, y) 
                
                self.writer.add_scalar('loss', l, global_step=None, walltime=None)
                
                if l>= 10e8:
                    flag = 1
                    break
                
                if l>=10e5:
                    self.lr = 100
                else:
                    self.lr = 1e-1
                
                optimizer.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm(net.parameters(), 5, norm_type=2)
                optimizer.step()
                train_l_sum += l.cpu().item()
                #train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
            batch_count += 1
            print('epoch %d, loss %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, time.time() - start))
        
        return None
    
    def movement(self,action):
        
        if action == 0:
            print('defence')
            directkey.defense()
            
        elif action == 1:
            print('attack')
            directkey.attack()
            
        elif action == 2:
            print("go forward")
            directkey.go_forward()
            
        elif action == 3:
            print('go back')
            directkey.go_back()
            
        elif action == 4:
            print('forwardjump')
            directkey.forward_jump()
            
        elif action == 5:
            print('left')
            directkey.go_left()
            
        elif action == 6:
            print('right')
            directkey.go_right()
            
        elif action == 7:
            print('jump')
            directkey.jump()
            
        elif action == 8:
            print('attack')
            directkey.attack()
            
        elif action == 9:
            print('fast go')
            directkey.F_go()
            
        elif action == 10:
            print('attack')
            directkey.attack()
        
        else:
            
            directkey.forward_jump()
            
        return None
    
            
    
    def interaction(self):
        
        sekiro_blood_at_state_1 = self.Reward.get_sekiro_blood_siuation()
        #sekiro_defence_at_state_1 = self.Reward.get_sekiro_defence_siuation()
        boss_blood_at_state_1 = self.Reward.get_boss_blood_siuation()
        
        if boss_blood_at_state_1 <=5:
            
            self.save_model('boss_dead'+str(self.boss_num))
            
            self.boss_num = self.boss_num +1
            
            if self.boss_num >=3:
                self.emergence = 1
                self.boss_num = 0
            
        
        if sekiro_blood_at_state_1 <=5:
            self.emergence = 1
            print("紧急停机")
            return (np.random.randint(0,(self.image_size,self.image_size)),'!','1',np.random.randint(0,(self.image_size,self.image_size)))

        if sekiro_blood_at_state_1 <= 53 or sekiro_blood_at_state_1 >= 236:
            self.dead = 1
            return (np.random.randint(0,(self.image_size,self.image_size)),'#','1',np.random.randint(0,(self.image_size,self.image_size)))
        
        image1,action = self.action_taken_policy_eposilon_greedy()
        
        #self.movement(action)
        
        sekiro_blood_at_state_2 = self.Reward.get_sekiro_blood_siuation()
        #sekiro_defence_at_state_2 = self.Reward.get_sekiro_defence_siuation()
        boss_blood_at_state_2 = self.Reward.get_boss_blood_siuation()
        
        if boss_blood_at_state_2 <=5:
            self.save_model('boss_dead')
        
        if sekiro_blood_at_state_2 <=5:
            self.emergence = 1
            print("紧急停机")
            return (np.random.randint(0,(self.image_size,self.image_size)),'!','1',np.random.randint(0,(self.image_size,self.image_size)))

        if sekiro_blood_at_state_2 <= 53 or sekiro_blood_at_state_1 >= 236:
            self.dead = 1
            return (np.random.randint(0,(self.image_size,self.image_size)),'#','1',np.random.randint(0,(self.image_size,self.image_size)))
        
        #screen_gray = cv2.cvtColor(grabscreen.grab_screen(self.main_view),cv2.COLOR_BGR2GRAY)#灰度图像收集
        screen_gray = cv2.cvtColor(grabscreen.grab_screen(self.main_view),cv2.COLOR_BGR2RGB)
        image2 = torch.Tensor([cv2.resize(screen_gray,(self.image_size,self.image_size)).transpose(2,1,0)])
        
        #max_q_with_next_state = self.fixed_q_net(torch.Tensor([[image2]])).max()
        
        #reward_sekiro_blood =  (-(sekiro_blood_at_state_1-sekiro_blood_at_state_2).mean())+25
        #reward_boss_blood = (boss_blood_at_state_1-boss_blood_at_state_2).mean()
        
        reward_sekiro_blood = sekiro_blood_at_state_1-sekiro_blood_at_state_2
        reward_boss_blood = -(boss_blood_at_state_1-boss_blood_at_state_2)
        
        if reward_sekiro_blood > 100:
            reward_sekiro_blood = 100
        elif reward_sekiro_blood < -100:
            reward_sekiro_blood = -100
        else:
            pass
        
        if reward_boss_blood>100:
            reward_boss_blood = 100
        elif reward_boss_blood<-100:
            reward_boss_blood = -100
        else:
            pass
        
        total_reward = 2*reward_sekiro_blood+1*reward_boss_blood
        
        if action == 0:
            total_reward = total_reward+5
        elif action == 1:
            total_reward = total_reward+2
        elif action == 2:
            total_reward = total_reward+0
        elif action == 3:
            total_reward = total_reward+0
        elif action == 4:
            total_reward = total_reward*0.05
        elif action == 5:
            total_reward = total_reward+1
        elif action == 6:
            total_reward = total_reward+1
        elif action == 7:
            total_reward = total_reward*0.05
        elif action == 8:
            total_reward = total_reward-100
        elif action== 9:
            total_reward = total_reward+0
        else:
            pass
    
        
        #reward_var = np.zeros(self.action_space)
        #reward_var[action] = 0.8*max_q_with_next_state+total_reward
        #reward_var[action] = total_reward
        
        self.now_reward = total_reward
        print('now_reward is '+ str(self.now_reward))
    
        return image1,action,total_reward,image2
    
    def interact_with_environment_at_minibatch_size(self):
        
        for i in range(self.minibatch_size):
        
            image1,action,label,image2 = self.interaction()
            
            if action=='#':
                return (np.random.randint(0,(self.image_size,self.image_size)),'#','1',np.random.randint(0,(self.image_size,self.image_size)))
            if action == '!':
                return (np.random.randint(0,(self.image_size,self.image_size)),'!','1',np.random.randint(0,(self.image_size,self.image_size)))
            
            #sample = [[image1],action,label,[image2]]
            sample = [image1,action,label,image2]
            
            self.replay_buffer.append(sample)
            self.sample_possibility = np.append(self.sample_possibility,1)
            
            if len(self.replay_buffer)>self.size_of_replay_buffer:
                self.remove_replay_based_on_possibility()
        
        return (np.random.randint(0,(self.image_size,self.image_size)),'.','1',np.random.randint(0,(self.image_size,self.image_size)))
    
    def remove_replay_based_on_possibility(self):
        
        possibility = self.sample_possibility/self.sample_possibility.sum()
        possibility = (1-possibility)/(1-possibility).sum()
        
        index = np.random.choice([i for i in range(len(self.replay_buffer))],size = int(self.size_of_replay_buffer*0.2),replace = False,p = possibility)
        
        self.sample_possibility = np.delete(self.sample_possibility,index)
        
        index = index.tolist()
        
        for i in range(len(index)):
            index_delete = index[i]
            del self.replay_buffer[index_delete]
            new_index = []
            for j in index:
                if j>i:
                    new_index.append(j-1)
                else:
                    new_index.append(j)
            index = new_index
        return None
    
    def sample_minibatch(self):
        
        #这里的采样是基于TD方法的采样

        labels = []

        choosen_samples = random.sample(self.replay_buffer,self.minibatch_size,)
        
        images = choosen_samples[0][0].cuda()
        image_fixed_q_net = choosen_samples[0][3]
        
        reward_vec = np.zeros(self.action_space)
        reward_vec[choosen_samples[0][1]] = choosen_samples[0][2]+self.gamma*self.fixed_q_net(torch.Tensor(image_fixed_q_net)).max()
        labels.append(reward_vec)    
        
        for i in choosen_samples[1:]:
            
            image_fixed_q_net = i[3]
            
            image_tensor = i[0].cuda()
            images = torch.cat((images,image_tensor))
            
            reward_vec = np.zeros(self.action_space)
            reward_vec[i[1]] = i[2]+self.fixed_q_net(torch.Tensor(image_fixed_q_net)).max()
            labels.append(reward_vec)

        labels = torch.Tensor(labels).cuda()

        deal_data = TensorDataset(images.float(),labels.float())

        #lengths = [int(len(deal_data)*0.7),len(deal_data)-int(len(deal_data)*0.7)]
        #data_iteration,test_iter = torch.utils.data.random_split(deal_data, lengths)
        
        #data_iteration = torch.utils.data.DataLoader(data_iteration,batch_size = self.minibatch_size,shuffle = True)
        #test_iter = torch.utils.data.DataLoader(test_iter,batch_size = self.minibatch_size,shuffle = True)
        
        data_iteration = torch.utils.data.DataLoader(deal_data,batch_size = self.minibatch_size,shuffle = True)
        
        #return data_iteration,test_iter
        return data_iteration
    
    def sample_minibatch_with_priority(self):
        
        labels = []
                
        possibility = self.sample_possibility/self.sample_possibility.sum()
        sample_result_index = np.random.choice([i for i in range(len(self.replay_buffer))],size = self.minibatch_size,p = possibility)
        
        W = pow(np.divide(1,np.multiply(self.minibatch_size,possibility)),self.beta)
        W = np.divide(W,W.max())
        
        #images = torch.Tensor([self.replay_buffer[sample_result_index[0]][0]]).cuda()
        #image_fixed_q_net = torch.Tensor([self.replay_buffer[sample_result_index[0]][3]])
        
        images = self.replay_buffer[sample_result_index[0]][0].cuda()
        image_fixed_q_net = self.replay_buffer[sample_result_index[0]][3]
        
        reward_vec = np.zeros(self.action_space)
        theta = self.replay_buffer[sample_result_index[0]][2]+self.gamma*self.fixed_q_net(image_fixed_q_net).max()-self.q_net(images).max().cpu()
        #theta = self.replay_buffer[sample_result_index[0]][2]+self.fixed_q_net(torch.Tensor(image_fixed_q_net)).max()
        reward_vec[self.replay_buffer[sample_result_index[0]][1]] = W[sample_result_index[0]]*theta
        self.sample_possibility[sample_result_index[0]] = abs(theta)
        labels.append(reward_vec)   
        
        for i in sample_result_index[1:]:
            
            print('now is '+ str(i))
            
            image_fixed_q_net = self.replay_buffer[i][3]
            
            image_tensor = self.replay_buffer[i][0].cuda()
            images = torch.cat((images,image_tensor))
            
            reward_vec = np.zeros(self.action_space)
            theta = self.replay_buffer[i][2]+self.fixed_q_net(image_fixed_q_net).max()-self.q_net(image_tensor).max().cpu()
            reward_vec[self.replay_buffer[i][1]] = theta*W[i]
            self.sample_possibility[i] = abs(theta)#self
            labels.append(reward_vec)
        
        labels = torch.Tensor(labels).cuda()
        deal_data = TensorDataset(images.float(),labels.float())
        data_iteration = torch.utils.data.DataLoader(deal_data,batch_size = self.minibatch_size,shuffle = True)
        
        return data_iteration
        
    def save_model(self,epoch):
        
        torch.save(self.q_net.state_dict(),self.path+str(epoch)+".pt")  
        torch.save(self.fixed_q_net.state_dict(),self.path+str(epoch)+"fixed.pt")  
        
        return None
    
    def load_model(self,epoch):
        
        self.q_net.load_state_dict(torch.load(self.path+str(epoch)+'.pt')) 
        self.fixed_q_net.load_state_dict(torch.load(self.path+str(epoch)+'fixed.pt')) 
        
        return None
    
    def fixed_q_net_add_noisy(self):
        
        model_params_dict = self.q_net.state_dict()
        
        i = 0
        
        for par1 in model_params_dict:
            #if i%2==0:
             #   continue
            model_params_dict[par1] = model_params_dict[par1] + torch.randn(model_params_dict[par1].shape).cuda()
            i+=1
            
        self.fixed_q_net.load_state_dict(model_params_dict)
        
        return None
    
    def remove_replay(self):
        
        self.replay_buffer = self.replay_buffer[int(self.size_of_replay_buffer*0.2):]
        
        return None

    def DQNtraining(self):
        
        self.initial_Q_network(1)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr = self.lr,weight_decay = 0.1)
        
        self.writer = SummaryWriter('./datavisualization')
        #self.writer.add_graph(self.q_net,torch.randn(1,3,self.image_size,self.image_size).cuda())
        
        flag = 0
        
        for i in range(self.episode):
            
            time_count = 0
            self.total_reward = 0
            
            directkey.lock_vision()
            
            if i == 0:
                image1,action,reward,image2 = self.interact_with_environment_at_minibatch_size()
            
            #if i*2+time_count >= 1000:
            #    self.epsilon = 0.85
                
            if i%50 ==1:
                self.save_model(i)
            
            while(not self.dead):
            #while(1):  
                #for j in range(3):
                print('采取动作')
                image1,action,reward,image2 = self.interaction()

                
                if action == '!':
                    print('emergence break')
                    self.save_model(i*2+time_count)
                    return None
                
                elif action == '#':
                    flag = 1
                    break
                
                else:
                    pass
                
                
                #self.replay_buffer.append([[image1],action,reward,[image2]])
                self.replay_buffer.append([image1,action,reward,image2])
                #self.sample_possibility = np.append(self.sample_possibility,self.sample_possibility.max())
                
                print(self.sample_possibility)
                
                if len(self.replay_buffer)>self.size_of_replay_buffer:
                    #self.remove_replay_based_on_possibility()
                    self.remove_replay()
                #if flag == 1:
                 #   flag =0
                  #  break
                
                #data_iteration,test_iter = self.sample_minibatch()
                
                #self.trainDQN(train_iter = data_iteration,test_iter = test_iter,net = self.q_net, optimizer = self.optimizer, device = self.device, num_epochs = self.num_epoch)
                
                time_count = time_count +1
                
                
                if time_count%3 == 0:
                    self.fixed_q_net.load_state_dict(self.q_net.state_dict())
                
                    
                print('第{}循环结束'.format(time_count))
                self.total_reward = self.total_reward+self.now_reward
                
            print("dead already")
            print("exception reward is "+ str(self.total_reward))
            for i in range(3):
                #data_iteration = self.sample_minibatch_with_priority()
                data_iteration = self.sample_minibatch()
                self.trainDQN(train_iter = data_iteration,net = self.q_net, optimizer = self.optimizer, device = self.device, num_epochs = self.num_epoch)
            
            if i%1 ==0:
                print("add noise to the nn")
                self.fixed_q_net_add_noisy()
                directkey.attack()
            else:
                time.sleep(2)
                directkey.attack()
                
            self.dead = 0
            
        self.save_model(i)
        
        return None

if __name__ == '__main__':
    
    test = DQNmodel()
    test.DQNtraining()
    
        
#开发日志2021517
"""
    尝试将网络预测值最大的值保留，和标签的格式统一
"""    
        
   #开发日志2021524
"""
        1.经过一晚的训练之后，发现网络趋向于只预测一个attack 的 action。这是不对劲的，所以把只狼血条的reward的比重提高了。√
        2.另外收集到的数据的利用率不高，目前的解决是把replaybuffer的size大幅度缩小，但是这又带来另外一个问题，就是会使网络受当前的数据的影响较大。
    另外一个解决的方式就是维持当前的buffer size，使用多线程进行网络的训练（将minibatch的size增大，同时不影响网络的训练）使用多线程的话，记得注意网络的占用问题。
        3.继续采用更多的DQN technique 下一个打算使用DDQN,dueling network。√
        4.更改数据的存储方式，不再使用image+label的方式，而是采用传统的DQN的数据存储方式。（感觉上是因为如果采用先前的数据存储的话，会导致只有静态的感觉√
        而后者是根据当前网络来输出值）√
    
"""   
"""
开发日志2021528
1, 将replaybuffer改成numpy格式存储

"""
"""
开发日志20210601

在采用Googlenet之后，发现网络非常难以训练，于是转回了采用带有duelingnetwork的resnet18.
在训练的时候发现，梯度下降之后网络的结果趋向于单一输出。每一次梯度下降之后的输出都十分单一
估计是因为环境的设计中存在着太多的噪声

收集到的样本的利用率也不高（主要是因为DQN的Qfunc的训练机制）
后面想想怎么去改进
（包括提高minibatch的大小，多次采样多次开启训练）
包括提高训练的速度等等

后面还是将priority replay加回去。

包括reward的设计也有一点点问题，
比如只狼的血量和boss的血量的比例还得调整下
后面试着将格挡条也加到reward中

试着去加入目标检测和分类的内容，来完善reward的设计（环境）
比如训练一个网络来看只狼是否躲过了boss的攻击，并根据这个去完善reward的设计（当然这里就是要用到监督学习了）后面得集成到self.interaction()当中



"""

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
