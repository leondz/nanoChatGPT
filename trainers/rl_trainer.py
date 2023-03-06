import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import time, os
from model import RLHF
from trainers.trainer import Trainer
from transformers import pipeline

# TODO: this works but is currently crude and incomplete, critic implementation plus PPO are obvious next steps
class PolicyGradientTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.mode = 'RL'
    
    def train(self):

        self.setup_ddp()

        ctx, meta_vocab_size = self.setup()

        self.meta_vocab_size = 50257

        # model init
        model = self.init_model()

        model = RLHF(model, self.mode, discrete_reward=self.config['discrete_reward'])

        if self.config['init_multihead_from'] == 'scratch':
            print("initializing multihead from scratch")
        else:
            if self.config['init_multihead_from'] == 'resume':
                print(f"Resuming training from {self.config['out_dir_multihead']}")
                # resume training from a checkpoint.
                ckpt_path = os.path.join(self.config['out_dir_multihead'], 'ckpt.pt')
                checkpoint = torch.load(ckpt_path, map_location=self.device)      
                state_dict = checkpoint['model']
                # fix the keys of the state dictionary :(
                # honestly no idea how checkpoints sometimes get this prefix, have to debug more
                unwanted_prefix = '_orig_mod.'
                for k,v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                model.load_state_dict(state_dict)

        import copy
        if self.config['hard_code_reward']:
            reward_model = None
            print('Using hard-coded reward')
        else:
            print('Using learned reward model')
            if self.config['separate_reward_model']:
                reward_model = copy.deepcopy(model)
                print('Reward model instantiated separately')
            else:
                reward_model = model
                print('Reward model and actor model share backbone')
            reward_model.to(self.device)
        
        model.to(self.device)

        critic_model = copy.deepcopy(model)
        critic_model.to(self.device)
        
        # actor_optimizer = torch.optim.AdamW(model.model.policy_head.parameters(), lr=1e-2)
        actor_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.AdamW(critic_model.parameters(), lr=1e-3)

        last_time = time.time()
        rews_all = []
        max_iters = 100000
        X, Y = self.get_batch('train') # fetch the very first batch
        X = torch.zeros((X.shape[0], 1), dtype=torch.long).to(self.device) # for now there is no prompt
        t0  = time.time()
        max_new_tokens = self.block_size

        sentiment_pipeline = pipeline("sentiment-analysis")


        for iter in range(max_iters):
            advantages = torch.zeros((X.shape[0], max_new_tokens)).to(self.device)
            returns = torch.zeros((X.shape[0], max_new_tokens)).to(self.device)
            gamma = 1
            lam = 1
            
            states, log_probs, log_probs_reference = model.generate(
                X, max_new_tokens, self.device, self.block_size, use_reference=False)
            
            states = states[:,-max_new_tokens:]

            text = []
            for i, s in enumerate(states):
                try:
                    te = self.enc.decode(s.tolist())
                except:
                    te = 'sad terrible'
                text.append(te)
                
            # text = [self.enc.decode(s.tolist()) for s in states]
            sent = sentiment_pipeline(text)
            rewards = torch.tensor([a['label']=='POSITIVE' for a in sent],dtype=torch.float16).unsqueeze(-1).to(self.device)
            # print(sent)

            # if self.config['hard_code_reward']: 
            #     # simple test where reward for outputting the letter 'z' (89)
            #     rewards = torch.zeros_like(states, dtype=torch.float16)
            #     rewards[states==89] = 1.0
            #     rewards = torch.sum(rewards, 1, keepdim=True)
            #     rewards[rewards > 1] = 1
            # else:
            #     if self.discrete_reward:
            #         rewards = reward_model.forward_reward(states)[0][:,1].unsqueeze(-1)
            #     else:
            #         rewards = reward_model.forward_reward(states)
            
            values = critic_model.forward_value(states).squeeze()
            # values = torch.zeros_like(states, dtype=torch.float16)
            
            for t in reversed(range(max_new_tokens)):
                if t == max_new_tokens - 1:
                    # value at last state is 0
                    delta = rewards[:].squeeze() - values[:, t]
                    advantages[:, t] = delta
                    returns[:, t] = rewards[:].squeeze()
                else:
                    # rewards can only be non-zero at the last state
                    delta = gamma * values[:, t + 1] - values[:, t]
                    advantages[:, t] = delta + gamma * lam * advantages[:, t + 1]
                    returns[:, t] += gamma * returns[:, t + 1]

            # minus KL divergence
            # if iter > 1000:
            pg = advantages * log_probs.squeeze() #- 1*(log_probs-log_probs_reference) #- 0.05*log_probs
            actor_loss = -pg.sum()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward(retain_graph=True)
            actor_optimizer.step()
            # else:
            #     actor_loss = None

            critic_loss = torch.mean((returns-values)**2)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_optimizer.step()

            torch.mean(rewards)

            rews_all.append(rewards.mean().detach().cpu().numpy())

            if iter % 1000 == 0:
                t1 = time.time()
                print(f'iter: {iter}, time: {t1-t0}')
                # print(actor_loss, critic_loss)
                print(f'Actor loss: {actor_loss}, iter: {iter}')
                print(f'Critic loss: {critic_loss}')
                print(f'rets: {np.mean(rews_all[-1000:])}')
                current_time = time.time()
                # print(current_time - last_time)
                last_time = current_time
                text = model.generate(X, self.block_size, self.device, self.block_size, reward_model=reward_model, use_reference=False)[0]
                for i in range(1):
                    text_i = text[i,:]
                    # print(reward(text_i))
                    try:
                        print(self.enc.decode(text_i.tolist()))
                    except:
                        continue 


class GumbelTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.mode = 'RL'
    
    def train(self):

        self.setup_ddp()

        ctx, meta_vocab_size = self.setup()

        # model init
        model = self.init_model()

        rl_model = RLHF(model, self.mode, discrete_reward=self.config['discrete_reward'])


        # The current approach is to use a separate reward model because otherwise optimisation of the reward model changes upstream parameters impacting performance of the multihead
        # I therefore load the language model from 'out_dir' and the reward model from 'out_dir_multihead'

        if self.config['init_multihead_from'] == 'scratch':
            print("initializing multihead from scratch")
        else:
            if self.config['init_multihead_from'] == 'resume':
                print(f"Resuming training from {self.config['out_dir']}")
                # resume training from a checkpoint.
                ckpt_path = os.path.join(self.config['out_dir'], 'ckpt.pt')
                checkpoint = torch.load(ckpt_path, map_location=self.device)      
                state_dict = checkpoint['model']
                # fix the keys of the state dictionary :(
                # honestly no idea how checkpoints sometimes get this prefix, have to debug more
                unwanted_prefix = '_orig_mod.'
                for k,v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                model.load_state_dict(state_dict)

        separate_reward_model = True     
        if separate_reward_model:
            print('Reward model instantiated as copy')
            import copy
            reward_model = copy.deepcopy(model)

            print(f"Resuming reward model from {self.config['out_dir_multihead']}")

            reward_model = RLHF(reward_model, self.mode, discrete_reward=self.config['discrete_reward'])
            # resume training from a checkpoint.
            ckpt_path = os.path.join(self.config['out_dir_multihead'], 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)      
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            reward_model.load_state_dict(state_dict)
        else:
            reward_model = rl_model
        rl_model.to(self.device)
        reward_model.to(self.device)

        gumbel_optimizer = torch.optim.AdamW(rl_model.parameters(), lr=1e-3)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))

        last_time = time.time()
        rews_all = []
        max_iters = 100000     
        
        X, Y = self.get_batch('train') # fetch the very first batch

        X = torch.zeros((X.shape[0], 1), dtype=torch.long).to(self.device) # for now there is no prompt

        t0  = time.time()
        for iter in range(max_iters):
            
            for micro_step in range(self.gradient_accumulation_steps):
                if self.ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    rl_model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
                with ctx:
                    states, rewards = rl_model.generate_gumbel(X, self.config['episode_length'], self.device, self.block_size, reward_model=reward_model)
                    mean_reward = rewards.mean()
                    loss = -mean_reward
                    # # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    # X, Y = self.get_batch('train')
                    # backward pass, with gradient scaling if training in fp16
                    scaler.scale(loss).backward()

            # clip the gradient
            if self.grad_clip != 0.0:
                scaler.unscale_(gumbel_optimizer)
                torch.nn.utils.clip_grad_norm_(rl_model.parameters(), self.grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(gumbel_optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            gumbel_optimizer.zero_grad(set_to_none=True)

            rews_all.append(mean_reward.detach().cpu().numpy())
            eval_interval = self.config['eval_interval']
            if iter % eval_interval == 0:
                t1 = time.time()
                print(f'iter: {iter}, time: {t1-t0}')
                print(f'rets: {np.mean(rews_all[-eval_interval:])}')
                current_time = time.time()
                # print(current_time - last_time)
                last_time = current_time
                text = rl_model.generate(X, self.config['episode_length'], self.device, self.block_size, reward_model=reward_model)[0]
                for i in range(1):
                    text_i = text[i,:]
                    # print(reward(text_i))
                    try:
                        print(self.enc.decode(text_i.tolist()))
                    except:
                        continue 