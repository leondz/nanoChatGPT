
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT, RewardModel
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from utils import dotdict


class Trainer():
    def __init__(self, config):
        self.config = config
        self.from_config(config)

        self.model_args = dict(n_layer=self.n_layer, n_head=self.n_head, n_embd=self.n_embd, block_size=self.block_size,
                        bias=self.bias, vocab_size=None, dropout=self.dropout) # start with model_args from command line
        self.meta_vocab_size = None
    
    def from_config(self, config):
        config = dotdict(config)

        # IO
        self.out_dir = config.out_dir
        self.eval_interval = config.eval_interval
        self.log_interval = config.log_interval
        self.eval_iters = config.eval_iters
        self.eval_only = config.eval_only
        self.always_save_checkpoint = config.always_save_checkpoint
        self.init_from = config.init_from
        
        # wandb
        self.wandb_log = config.wandb_log
        self.wandb_project = config.wandb_project
        self.wandb_run_name = config.wandb_run_name

        # data
        self.dataset = config.dataset
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.batch_size = config.batch_size
        self.block_size = config.block_size

        # model
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.bias = config.bias

        # optimizer
        self.learning_rate = config.learning_rate
        self.max_iters = config.max_iters
        self.weight_decay = config.weight_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.grad_clip = config.grad_clip
        self.decay_lr = config.decay_lr
        self.warmup_iters = config.warmup_iters
        self.lr_decay_iters = config.lr_decay_iters
        self.min_lr = config.min_lr

        # DDP
        self.backend = config.backend

        # system
        self.device = config.device
        self.dtype = config.dtype
        self.compile = config.compile
        
        print(self.out_dir)
    
    def setup_ddp(self):

        self.ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        if self.ddp:
            init_process_group(backend=self.backend)
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE']) # total number of training processes
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0 # this process will do logging, checkpointing etc.
            self.seed_offset = self.ddp_rank # each process gets a different seed
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.world_size = 1
            self.master_process = True
            self.seed_offset = 0
            self.ddp_local_rank = None

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
    
    def get_lr(self, it):
         # learning rate decay scheduler (cosine with warmup)
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)
    
    def init_model(self):
        if self.init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if self.meta_vocab_size is None:
                print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            self.model_args['vocab_size'] = self.meta_vocab_size if self.meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**self.model_args)
            model = GPT(gptconf)
        elif self.init_from == 'resume':
            print(f"Resuming training from {self.out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            checkpoint_model_args = checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                self.model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = GPTConfig(**self.model_args)
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
        elif self.init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=dropout)
            model = GPT.from_pretrained(init_from, override_args)
            # read off the created config params, so we can store them into checkpoint correctly
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                self.model_args[k] = getattr(model.config, k)
                # crop down the model block size if desired, using model surgery
        if self.block_size < model.config.block_size:
            model.crop_block_size(self.block_size)
            self.model_args['block_size'] = self.block_size # so that the checkpoint will have the right value
        return model

    def train(self):
        # set up distributed training
        self.setup_ddp()

        if self.master_process:
            os.makedirs(self.out_dir, exist_ok=True)

        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)

        # poor man's data loader
        data_dir = os.path.join('data', self.dataset)
        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(data_dir, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        # model init
        model =  self.init_model()
    
        model.to(self.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))

        # optimizer
        self.optimizer = model.configure_optimizers(self.weight_decay, self.learning_rate, \
         (self.beta1, self.beta2), self.device_type)
        if self.init_from == 'resume':
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # compile the model
        if self.compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0

        # wrap model into DDP container
        if self.ddp:
            model = DDP(model, device_ids=[self.ddp_local_rank])

        # logging
        if self.wandb_log and self.master_process:
            import wandb
            wandb.init(project=self.wandb_project, name=self.wandb_run_name, config=config)

        # training loop
        X, Y = self.get_batch('train') # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        running_mfu = -1.0
        while True:

            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if self.decay_lr else self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % self.eval_interval == 0 and self.master_process:
                losses = self.estimate_loss(model, ctx)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if self.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                    })
                if losses['val'] < best_val_loss or self.always_save_checkpoint:
                    best_val_loss = losses['val']
                    raw_model = model.module if self.ddp else model
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        print(f"saving checkpoint to {self.out_dir}")
                        torch.save(checkpoint, os.path.join(self.out_dir, 'ckpt.pt'))
            if iter_num == 0 and self.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.gradient_accumulation_steps):
                if self.ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = model(X, Y)
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if self.grad_clip != 0.0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(self.optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.log_interval == 0 and self.master_process:
                lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = model.estimate_mfu(self.batch_size * self.world_size * self.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > self.max_iters:
                break

        if self.ddp:
            destroy_process_group()
        
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self, model, ctx):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


class RewardModelTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
    
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([self.reward(torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64))) for i in ix])
        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
    
    def reward(self, sequence, t='pizza'):
        if t in self.enc.decode(sequence.tolist()):
            # print('hello')
            return torch.tensor([0.0,1.0])
        else:
            return torch.tensor([1.0, 0.0])

    def train(self):
        # set up distributed training
        self.setup_ddp()

        if self.master_process:
            os.makedirs(self.out_dir, exist_ok=True)

        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)

        # poor man's data loader
        data_dir = os.path.join('data', self.dataset)
        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(data_dir, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        # model init
        model =  self.init_model()
        model = RewardModel(model)

        model.to(self.device)

        # # initialize a GradScaler. If enabled=False scaler is a no-op
        # scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))

        # optimizer
        # self.optimizer = model.configure_optimizers(self.weight_decay, self.learning_rate, \
        #  (self.beta1, self.beta2), self.device_type)
        # if self.init_from == 'resume':
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.optimizer = torch.optim.AdamW(model.model.lm_head.parameters(), lr=1e-3)


        # compile the model
        if self.compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0

        # wrap model into DDP container
        if self.ddp:
            model = DDP(model, device_ids=[self.ddp_local_rank])

        # logging
        if self.wandb_log and self.master_process:
            import wandb
            wandb.init(project=self.wandb_project, name=self.wandb_run_name, config=config)

        # training loop
        X, Y = self.get_batch('train') # fetch the very first batch
        t0 = time.time()
        
        for iter in range(self.max_iters):

            # every once in a while evaluate the loss on train and val sets
            if iter % self.eval_interval == 0 or iter == self.max_iters - 1:
                losses = self.estimate_loss(model, ctx)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                reward_probs, _ = model(X)
                text = self.enc.decode(X[iter % self.eval_interval].tolist())
                print(f"reward pred: {reward_probs[0][-1]}", "input: ", text[:30])
                # text = text.split()
                # text[10:14] = 'pizza'
                # text = "".join(text)
                test_text = 'I love pizza' + 'a'*10000
                test_text_enc = torch.tensor(self.enc.encode(test_text)[:256]).unsqueeze(0)
                test_reward_probs, _ = model(test_text_enc.to(self.device))
                print(f"test reward pred: {test_reward_probs[0][-1]}", "input: ", test_text[:30])

            # sample a batch of data
            xb, yb = self.get_batch('train')

            # evaluate the loss
            logits, loss = model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        
        # local_iter_num = 0 # number of iterations in the lifetime of this process
        # running_mfu = -1.0
        # while True:

        #     # determine and set the learning rate for this iteration
        #     lr = self.get_lr(iter_num) if self.decay_lr else self.learning_rate
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = lr

        #     # evaluate the loss on train/val sets and write checkpoints
        #     if iter_num % self.eval_interval == 0 and self.master_process:
        #         losses = estimate_loss()
        #         print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        #         if self.wandb_log:
        #             wandb.log({
        #                 "iter": iter_num,
        #                 "train/loss": losses['train'],
        #                 "val/loss": losses['val'],
        #                 "lr": lr,
        #                 "mfu": running_mfu*100, # convert to percentage
        #             })
        #         if losses['val'] < best_val_loss or self.always_save_checkpoint:
        #             best_val_loss = losses['val']
        #             raw_model = model.module if self.ddp else model
        #             if iter_num > 0:
        #                 checkpoint = {
        #                     'model': raw_model.state_dict(),
        #                     'optimizer': self.optimizer.state_dict(),
        #                     'model_args': self.model_args,
        #                     'iter_num': iter_num,
        #                     'best_val_loss': best_val_loss,
        #                     'config': config,
        #                 }
        #                 print(f"saving checkpoint to {self.out_dir}")
        #                 torch.save(checkpoint, os.path.join(self.out_dir, 'ckpt.pt'))
        #     if iter_num == 0 and self.eval_only:
        #         break

        #     # forward backward update, with optional gradient accumulation to simulate larger batch size
        #     # and using the GradScaler if data type is float16
        #     for micro_step in range(self.gradient_accumulation_steps):
        #         if self.ddp:
        #             # in DDP training we only need to sync gradients at the last micro step.
        #             # the official way to do this is with model.no_sync() context manager, but
        #             # I really dislike that this bloats the code and forces us to repeat code
        #             # looking at the source of that context manager, it just toggles this variable
        #             model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
        #         with ctx:
        #             logits, loss = model(X, Y)
        #         # immediately async prefetch next batch while model is doing the forward pass on the GPU
        #         X, Y = self.get_batch('train')
        #         # backward pass, with gradient scaling if training in fp16
        #         scaler.scale(loss).backward()
        #     # clip the gradient
        #     if self.grad_clip != 0.0:
        #         scaler.unscale_(self.optimizer)
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        #     # step the optimizer and scaler if training in fp16
        #     scaler.step(self.optimizer)
        #     scaler.update()
        #     # flush the gradients as soon as we can, no need for this memory anymore
        #     self.optimizer.zero_grad(set_to_none=True)

        #     # timing and logging
        #     t1 = time.time()
        #     dt = t1 - t0
        #     t0 = t1
        #     if iter_num % self.log_interval == 0 and self.master_process:
        #         lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
        #         if local_iter_num >= 5: # let the training loop settle a bit
        #             mfu = model.estimate_mfu(self.batch_size * self.world_size * self.gradient_accumulation_steps, dt)
        #             running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        #         print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        #     iter_num += 1
        #     local_iter_num += 1

        #     # termination conditions
        #     if iter_num > self.max_iters:
        #         break

        # if self.ddp:
        #     destroy_process_group()