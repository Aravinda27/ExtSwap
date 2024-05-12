from torchvision import transforms
from torchvision import utils
from Losses.AdversarialLoss import calc_Dw_loss, R1_regulazation
import torch
from Losses.NonAdversarialLoss import landmark_loss, rec_loss, l2_loss,id_loss
from Configs import Global_Config
import lpips
import wandb


class Trainer:

    def __init__(self, config,
                 discriminator_optimizer: torch.optim.Optimizer,
                 adversarial_mapper_optimizer: torch.optim.Optimizer,
                 non_adversarial_mapper_optimizer: torch.optim.Optimizer,
                 discriminator,
                 generator,
                 id_encoder,
                 
                 landmark_encoder):

        self.config = config
        self.discriminator_optimizer = discriminator_optimizer
        self.adversarial_mapper_optimizer = adversarial_mapper_optimizer
        self.non_adversarial_mapper_optimizer = non_adversarial_mapper_optimizer
        self.discriminator = discriminator
        self.discriminator = discriminator
        self.generator = generator
        self.id_encoder = id_encoder
        
        self.landmark_encoder = landmark_encoder
        self.lpips_loss = lpips.LPIPS(net='alex').to(Global_Config.device).eval()

    def train_discriminator(self, real_w, generated_w):
        self.discriminator_optimizer.zero_grad()
        real_w.requires_grad_()

        prediction_real = self.discriminator(real_w).view(-1)
        error_real = calc_Dw_loss(prediction_real, 1)
        error_real.backward(retain_graph=True)

        r1_error = R1_regulazation(self.config['R1Param'], prediction_real, real_w)
        r1_error.backward()

        cloned_generated_w = generated_w.clone().detach()
        prediction_fake = self.discriminator(cloned_generated_w).view(-1)
        error_fake = calc_Dw_loss(prediction_fake, 0)
        error_fake.backward()

        self.discriminator_optimizer.step()

        return error_real, prediction_real, error_fake, prediction_fake

    def train_mapper(self, generated_w):

        self.adversarial_mapper_optimizer.zero_grad()
        prediction = self.discriminator(generated_w).view(-1)
        discriminative_loss = calc_Dw_loss(prediction, 1)
        discriminative_loss.backward()
        self.adversarial_mapper_optimizer.step()

        return discriminative_loss, prediction

    def adversarial_train_step(self, real_w, fake_data):
        error_real, prediction_real, error_fake, prediction_fake = self.train_discriminator(real_w, fake_data)
        g_error, g_pred = self.train_mapper(fake_data)
        print("Adverserial")

        return error_real, error_fake, torch.mean(prediction_real), torch.mean(prediction_fake), g_error, torch.mean(
            g_pred)

    def non_adversarial_train_step(self, id_images, attr_images, w, real_landmarks, use_rec_extra_term):
        self.id_encoder.zero_grad()
        self.landmark_encoder.zero_grad()
        self.generator.zero_grad()
        #print("Non-adverserial")
        #print("w_shape",w.shape)

        total_loss = torch.tensor(0, dtype=torch.float, device=Global_Config.device)

        generated_images, _ = self.generator(
            [w], input_is_latent=True, return_latents=False
        )
        utils.save_image(generated_images,f"sample/sample.png",nrow=1,normalize=True,range=(-1, 1))
        #print("generated_images",generated_images.type)

        normalized_generated_images = (generated_images + 1) / 2
        utils.save_image(normalized_generated_images,f"sample/sample1.png",nrow=1,normalize=True,range=(-1, 1))

        pred_embeddings=self.id_encoder(generated_images)
        #print("Pred_embeddings",pred_embeddings.shape)
        id_embeddings=self.id_encoder(id_images)

        ## -1 to 1
        if self.config['use_id']:
            id_loss_val = self.config['lambdaID'] * id_loss(pred_embeddings,id_embeddings)
            #print("Id_loss",id_loss_val)
            total_loss += id_loss_val
            wandb.log({'id_loss_val': id_loss_val.detach().cpu()}, step=Global_Config.step)

        if self.config['use_landmark']:
            generated_landmarks, generated_landmarks_nojawline = self.landmark_encoder(normalized_generated_images)
            landmark_loss_val = landmark_loss(generated_landmarks_nojawline, real_landmarks) * self.config['lambdaLND']
            total_loss += landmark_loss_val
            #print("Total_landmark_loss",total_loss)
            wandb.log({'landmark_loss_val': landmark_loss_val.detach().cpu()}, step=Global_Config.step)

        ## 0 to 1
        
        if use_rec_extra_term and self.config['use_reconstruction']:
            #print("Why are you stuck")
            rec_loss_val = self.config['lambdaREC'] * rec_loss(attr_images, normalized_generated_images,
                                                               self.config['a'])
            total_loss += rec_loss_val
            #print("Total_loss_rec",total_loss)
            wandb.log({'rec_loss_val': rec_loss_val.detach().cpu()}, step=Global_Config.step)
        
	
        if use_rec_extra_term and self.config['use_l2'] > 0:
            #print("Why are you struck")
            l2_loss_val = self.config['lambdaL2'] * l2_loss(attr_images, normalized_generated_images)
            total_loss += l2_loss_val
            #print("Total_loss_2",total_loss)
            wandb.log({'l2_loss_val': l2_loss_val.detach().cpu()}, step=Global_Config.step)
        

        ## -1 to 1
        if use_rec_extra_term and (not self.config['use_adverserial']):
            #print("Why are you struck")
            vgg_loss_val = torch.mean(
                self.config['lambdaVGG'] * self.lpips_loss(generated_images, (attr_images * 2) - 1))
            #print("VGG_loss",vgg_loss_val)
            wandb.log({'vgg_loss_val': vgg_loss_val.detach().cpu()}, step=Global_Config.step)
            total_loss += vgg_loss_val

        self.non_adversarial_mapper_optimizer.zero_grad()
        total_loss.backward()
        
        
        self.non_adversarial_mapper_optimizer.step()

        return total_loss
