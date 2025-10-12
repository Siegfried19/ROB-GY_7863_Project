from model import Actor,Critic
import torch.optim as optim
from parameters import *
import torch
import numpy as np
import os



class Ppo:
    def __init__(self,N_S,N_A,device):
        self.device = device
        self.actor_net = Actor(N_S, N_A).to(device).to(device)
        self.critic_net = Critic(N_S).to(device).to(device)

        self.actor_optim = optim.Adam(self.actor_net.parameters(),lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic_net.parameters(),lr=lr_critic,weight_decay=l2_rate)
        self.critic_loss_func = torch.nn.MSELoss()

    def train(self,memory):
        states  = torch.tensor(np.vstack([m[0] for m in memory]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.vstack([np.array(m[1]).flatten() for m in memory]), dtype=torch.float32).to(self.device)
        rewards = torch.tensor([m[2] for m in memory], dtype=torch.float32).to(self.device)
        masks   = torch.tensor([m[3] for m in memory], dtype=torch.float32).to(self.device)

        values = self.critic_net(states)
        returns, advants = self.get_gae(rewards, masks, values)
        old_mu, old_std = self.actor_net(states)
        pi = self.actor_net.distribution(old_mu, old_std)
        old_log_prob = pi.log_prob(actions).sum(1, keepdim=True)

        n = len(states)
        arr = np.arange(n)
        steps = 0
        loss_actor_sum, loss_critic_sum = 0, 0
        for epoch in range(10):
            np.random.shuffle(arr)
            for i in range(n//batch_size):
                steps+=1
                b_index = arr[batch_size*i:batch_size*(i+1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)

                mu,std = self.actor_net(b_states)
                pi = self.actor_net.distribution(mu,std)
                new_prob = pi.log_prob(b_actions).sum(1,keepdim=True)
                old_prob = old_log_prob[b_index].detach()
                #KL散度正则项
               # KL_penalty = self.kl_divergence(old_mu[b_index],old_std[b_index],mu,std)
                ratio = torch.exp(new_prob-old_prob)

                surrogate_loss = ratio*b_advants
                values = self.critic_net(b_states)

                critic_loss = self.critic_loss_func(values,b_returns)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                ratio = torch.clamp(ratio,1.0-epsilon,1.0+epsilon)

                clipped_loss =ratio*b_advants

                actor_loss = -torch.min(surrogate_loss,clipped_loss).mean()
                #actor_loss = -(surrogate_loss-beta*KL_penalty).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
               
                self.actor_optim.step()
                loss_actor_sum+=actor_loss
                loss_critic_sum+=critic_loss
        return loss_actor_sum/steps, loss_critic_sum/steps
    #计算KL散度
    def kl_divergence(self,old_mu,old_sigma,mu,sigma):

        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()

        kl = torch.log(old_sigma) - torch.log(sigma) + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / \
             (2.0 * sigma.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    #计算GAE
    def get_gae(self,rewards, masks, values):
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            #计算A_t并进行加权求和
            running_returns = rewards[t] + gamma * running_returns * masks[t]
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - \
                              values.data[t]
            running_advants = running_tderror + gamma * lambd * \
                              running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        #advants的归一化
        advants = (advants - advants.mean()) / advants.std()
        return returns, advants
    
    def save_model(self, path, iter_num, max= False):
      

        os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
        if max:
            actor_path = os.path.join(path, "checkpoints", f"actor_max.pth")
            critic_path = os.path.join(path, "checkpoints",f"critic_max.pth")
        
        else: 
            actor_path = os.path.join(path, "checkpoints", f"actor_{iter_num}.pth")
            critic_path = os.path.join(path, "checkpoints",f"critic_{iter_num}.pth")
        torch.save(self.actor_net.state_dict(), actor_path)
        torch.save(self.critic_net.state_dict(), critic_path)
        print(f" save model {actor_path} and {critic_path}")

    def load_model(self, path, iter_num):
        actor_path = os.path.join(path, "checkpoints", f"actor_max.pth")
        critic_path = os.path.join(path, "checkpoints", f"critic_max.pth")
        self.actor_net.load_state_dict(torch.load(actor_path))
        self.critic_net.load_state_dict(torch.load(critic_path))
        print(f" laod model {actor_path} and {critic_path}")