from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime
import os
import torch.profiler
from torch.utils.tensorboard import SummaryWriter

class Solver(object):

    def __init__(self, vcc_loader, val_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.val_loader = val_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.checkpoint = config.checkpoint
        self.save_path = config.save_path
        self.log_path = config.log_path
        self.learning_rate = config.lr

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_iters = config.num_iters
        self.val_iters = config.val_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model()
        self.writer = SummaryWriter(self.log_path)


            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.learning_rate)

        if self.checkpoint is not None:
            self.G.load_state_dict(torch.load(self.checkpoint)['model'])
        
        self.G.to(self.device)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    def training_step(self, data_iter):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #

        
        x_real, emb_org = next(data_iter)
        
        
        x_real = x_real.to(self.device) 
        emb_org = emb_org.to(self.device) 
                    

        # =================================================================================== #
        #                               2. Train the generator                                #
        # =================================================================================== #
        
        self.G = self.G.train()
                    
        # Identity mapping loss
        x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
        g_loss_id = F.mse_loss(x_real, x_identic.squeeze())   
        g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze())   
        
        # Code semantic loss.
        code_reconst = self.G(x_identic_psnt, emb_org, None)
        g_loss_cd = F.l1_loss(code_real, code_reconst)


        # Backward and optimize.
        g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()

        # Logging.
        loss = {}
        loss['G/loss'] = g_loss
        loss['G/loss_id'] = g_loss_id.item()
        loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
        loss['G/loss_cd'] = g_loss_cd.item()
        return loss, g_loss
                
    def train(self):
        # Set data loader.
        os.makedirs(self.log_path, exist_ok=True)
        
        # Print logs in specified order
        keys = ['G/loss','G/loss_id','G/loss_id_psnt','G/loss_cd']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_path),
            record_shapes=True,
            with_stack=True)
        for j in range(self.num_epochs):
            avg_train_loss = 0
            avg_val_loss = 0

            for i in range(self.num_iters):
                if j == 1 and i == 0:
                    prof.start()
                try:
                    loss, g_loss = self.training_step(data_iter)
                except:
                    data_iter = iter(self.vcc_loader)
                    loss, g_loss = self.training_step(data_iter)
                if j == 1 and i == (1 + 1 + 3) * 2 - 1:
                    prof.stop()
                avg_train_loss += g_loss
                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training information.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}] of epoch {}".format(et, i+1, self.num_iters, j+1)
                    for tag in keys:
                        log += ", {}: {:.4f}".format(tag, loss[tag])
                    print(log, end='\r')
            # =================================================================================== #
            #                                   5. Validation                                     #
            # =================================================================================== #
            self.G.eval()
            print('')
            for i in range(self.val_iters):
                try:
                    x_real, emb_org = next(data_iter)
                except:
                    data_iter = iter(self.val_loader)
                    x_real, emb_org = next(data_iter)
                x_real = x_real.to(self.device) 
                emb_org = emb_org.to(self.device)
                x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
                g_loss_id = F.mse_loss(x_real, x_identic.squeeze())   
                g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze())   
                # Code semantic loss.
                code_reconst = self.G(x_identic_psnt, emb_org, None)
                g_loss_cd = F.l1_loss(code_real, code_reconst)
                # Backward and optimize.
                g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
                avg_val_loss += g_loss
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Validation iteration [{}/{}] of epoch {}".format(et, i+1, self.num_iters, j+1)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log, end='\r')
            print(f'\nEpoch {j+1} complete, train loss: {avg_train_loss/self.num_iters:.3e}, val loss: {avg_val_loss/self.val_iters:.3e}')
            with open(os.path.join(self.log_path, "val_loss.dat"), "a") as myfile:  # save progress of val loss to text file
                myfile.write(f"{j+1} {avg_val_loss/self.val_iters:.3e}\n")
            with open(os.path.join(self.log_path, "train_loss.dat"), "a") as myfile:  # save progress of val loss to text file
                myfile.write(f"{j+1} {avg_train_loss/self.num_iters:.3e}\n")
            self.writer.add_scalars('Loss', {
                'training': avg_train_loss/self.num_iters,
                'validation': avg_val_loss/self.val_iters,
            }, j)






        torch.save({
            'model': self.G.state_dict(), 
            'optimizer': self.g_optimizer.state_dict()
            }, os.path.join(self.log_path, self.save_path))
        print('Model checkpoint saved.')

    
    

    