import torch
import torchvision.utils as vutils
import torch.optim as optim
import torchvision.transforms as transforms


import nibabel as nib
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from copy import deepcopy
from math import floor
import pickle

import sys
sys.path.append('../..')
from utils.utilities import *
from utils.losses import *
from utils.import_export_slicer import *
from modules.render import *

np.seterr(divide='ignore', invalid='ignore')

# Set random seed for reproducibility
manualSeed = 999

# Use if you want to get new results
#manualSeed = random.randint(1, 10000)
torch.cuda.empty_cache()
print("Random Seed: ", manualSeed)
torch.manual_seed(manualSeed)

# Setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ngpu = torch.cuda.device_count()
print('Cuda is available' if torch.cuda.is_available() else 'Cuda is not available')
print('Number of GPUs: {} ---> {}'.format(ngpu, 'Distributed' if ngpu > 1 else 'Not distributed') )

#---Camera Setting
focal = 138.88887889922103
print("Focal length: {0:.2f}".format(focal))

radius = 4.0311
t_near = radius - np.sqrt(3)/2
t_far = radius + np.sqrt(3)/2


#---Camera Poistion for Tensorboard Images
peek_pose = look_at_sphere(32, radius=radius,
                       target=torch.tensor([0., 0., 0.], device=device),
                       up=torch.tensor([0., 0., 1.], device=device))[[2]].clone()
peek_pose_2d = (0, torch.tensor(140, device=device))

#---Background color
bg_color = torch.ones(3, device=device) 


#---Reading Volumes
### REFRENCE volume is the volume that we already have an approprite
### transfer function for it to visulize its features of interest.

### TARGET volume is the volume that we are trying to find an approprite
### tansfer function for it based on the images of the reference volume. 
# Target volume (Philips) that is registered on reference (GE)
target, rv_min, rv_max, grad_max = volume_loader('../../volumes/registered_on_ge_3_59_M/reg_philips_3_42_m.nii.gz',
                                                               with_grad=True,
                                                               bad_spacing=True,
                                                               normalize_scalars=True)
# Reference Volume (GE)
ref, gt_rv_min, gt_rv_max, gt_grad_max = volume_loader('../../volumes/original/CC0305_ge_3_59_M.nii.gz',
                              with_grad=True,
                              bad_spacing=True,
                              normalize_scalars=True)
rvs = torch.cat([target, ref])



# Creating a neural network object
trainable_custom_tf = TF_for_VR2(dim_in=2)
gt_custom_tf = TF_for_VR2(dim_in=2)

# Loading the transfer function for reference
# In this case the reference transfer function comes from the previous step
# and is sotred in NN
gt_custom_tf.load_state_dict({key.replace('custom_architecture.', ''):value for\
   key,value in torch.load(f'../../saved_models/trivial_WM_GM_CSF_mask2 GE_3_59_M dt=50 lr=0.0005 with lr decay.pt').items()\
   if key.startswith('custom_architecture.')})

for param in gt_custom_tf.parameters():
    param.requires_grad = False

_=trainable_custom_tf.to(device), gt_custom_tf.to(device)


# Initializing Volume Renderer
image_size = 500
focal_coe = image_size / 32.
volume_renderer = VolumeRenderer4(
    image_size, image_size, focal, focal_coe,
    t_near, t_far, tf_mode='custom2', volume=rvs, t_delta=0.005)

vr_attr = volume_renderer
vr_attr.custom_architecture = trainable_custom_tf
vr_attr.custom_architecture_2 = gt_custom_tf
volume_renderer.to(device)



# Visulize refrence volume
with torch.no_grad():
    with vr_attr.brighten():
        vr_attr.tf_mode = 'custom2'
        vr_attr.volume_idx = 1
        volume_renderer.to(device)
        fig, axs = plt.subplots(2, 2, figsize=(9,5))

        # visualize the volume
        axs[0, 0].imshow(tensor2image(volume_renderer(peek_pose)))
        axs[0, 0].axis(False)
        axs[0, 0].set_title("3D View")

        # visualize 2D slice
        slices = vr_attr.render_slice(*peek_pose_2d)
        axs[0, 1].imshow(tensor2image(slices))
        axs[0, 1].axis(False)
        axs[0, 1].set_title("2D Slice")

        # Visualize scaler and gradient transfer functions
        density_tf, gradient_opacity = adaptive_tf_sampler(vr_attr.custom_architecture_2,
                                                threshold=0.05,
                                                initial_number_samples=100,
                                                normalization_min=gt_rv_min,
                                                normalization_max=gt_rv_max,
                                                domain=(0., 1.),
                                                sample_gradien=True,
                                                grad_normalization_min=0.,
                                                grad_normalization_max=(gt_rv_max-gt_rv_min),
                                                grad_domain=(0., gt_grad_max),
                                                pass_none=True,
                                                device=device)
        axs[1, 0].set_title("Scalar Opacity TF")
        tf_visualizer_scalar(axs[1, 0], density_tf)
        axs[1, 1].set_title("Gradient Opacity TF")
        tf_visualizer_gradient(axs[1, 1], gradient_opacity)

        plt.suptitle('The reference transfer function applied on reference volume.')
        plt.tight_layout()
        plt.pause(2)



# Visualize target volume with its random TF
with torch.no_grad():
    with vr_attr.brighten():
        vr_attr.tf_mode='custom'
        vr_attr.volume_idx = 0
        volume_renderer.to(device)
        fig, axs = plt.subplots(2, 2, figsize=(9,5))
        
        # visualize the volume
        axs[0, 0].imshow(tensor2image(volume_renderer(peek_pose)))
        axs[0, 0].axis(False)
        axs[0, 0].set_title("3D View")
        
        # visualize 2D slice
        slices = vr_attr.render_slice(*peek_pose_2d)
        axs[0, 1].imshow(tensor2image(slices))
        axs[0, 1].axis(False)
        axs[0, 1].set_title("2D Slice")

        # Visualize scalar and gradient transfer functions
        density_tf, gradient_opacity = adaptive_tf_sampler(vr_attr.custom_architecture,
                                                threshold=0.05,
                                                initial_number_samples=100,
                                                normalization_min=rv_min,
                                                normalization_max=rv_max,
                                                domain=(0., 1.),
                                                sample_gradien=True,
                                                grad_normalization_min=0.,
                                                grad_normalization_max=(rv_max-rv_min),
                                                grad_domain=(0., grad_max),
                                                pass_none=True,
                                                device=device)
        axs[1, 0].set_title("Scalar Opacity TF")
        tf_visualizer_scalar(axs[1, 0], density_tf)
        axs[1, 1].set_title("Gradient Opacity TF")
        tf_visualizer_gradient(axs[1, 1], gradient_opacity)

        plt.suptitle('The initial random transfer function applied on registered target volume.')
        plt.tight_layout()
        plt.pause(2)


# Reducing image size and preparing paramaters for traning
# Image size and focal length
image_size = 200
focal_coe = image_size / 64.

# Initializing Volume Renderer
volume_renderer = VolumeRenderer4(
    image_size, image_size, focal, focal_coe,
    t_near, t_far, tf_mode='custom2', volume=rvs, t_delta=0.005)

vr_attr = volume_renderer
del(vr_attr.custom_architecture)
vr_attr.custom_architecture = trainable_custom_tf
vr_attr.custom_architecture_2 = gt_custom_tf
volume_renderer.to(device)
vr_attr.bg = bg_color


# Writing down ground truth in tensorboard
with torch.no_grad():
    with vr_attr.brighten():
        vr_attr.volume_idx = 1
        vr_attr.tf_mode = 'custom2'
        writer = SummaryWriter(os.path.join('runs', 'GE Ground Truth'))
        writer.add_image('Peek', volume_renderer(peek_pose)[0].cpu().clamp(0., 1.))
        writer.add_image('Peek 2D', vr_attr.render_slice(*peek_pose_2d)[0])
        writer.add_figure('TF', visualize_transfer_func(vr_attr.custom_architecture_2,
                                                        domain=(-0.1, 1.),
                                                        grad_domain=(0, gt_grad_max),
                                                        device=device))
        

# Use DataParallel if multiple GPUs are avaialbe
if ngpu > 1:
    print("Let's use", ngpu, "GPUs!")
    if  str(type(volume_renderer)).find('DataParallel') == -1:
        volume_renderer = nn.DataParallel(volume_renderer)
        vr_attr = volume_renderer.module

_ = volume_renderer.to(device)

# L1 loss
loss_l1 = nn.L1Loss()
# Style loss
loss_vgg = VGGPerceptualLoss().to(device)

# Only for saving vp file to be opend by slicer
first_six_values = [1, 1, 1, 0.2, 0, 1]


# Training Params
batch_size = 4
batch_size_2d = 4
verbose_n_batch = 32
epoches = 2
num_views = 2048
iterations = floor(num_views / batch_size)
dt_unit_coe = 50
lr = 0.0005
volume_shape = vr_attr.volume.shape[2:]
get_random_dim = lambda: random.randint(0, 2)
get_random_indices = lambda dim: torch.randint(volume_shape[dim] // 4,
                                               3 * volume_shape[dim] // 4,
                                               (batch_size_2d,),
                                               device=device)
#######################################################
vr_attr.dt_unit_coe = torch.tensor([dt_unit_coe,], device=device)
optimizer = optim.Adam(vr_attr.custom_architecture.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                           patience=128, threshold=0.00001, threshold_mode='rel',
                                           cooldown=10, min_lr=0.000001, eps=1e-08, verbose=True)


# Updaing writer for new experiment
comment = f"Style transfer from GE to Philips with l1+late gram loss+2D gram-5-1"
writer = SummaryWriter(os.path.join('runs', comment))
print(comment)

# Writing the inital state in tensorboard
with torch.no_grad():
    with vr_attr.brighten():
        vr_attr.bg = bg_color
        
        with torch.no_grad():
            vr_attr.tf_mode = 'custom2'
            vr_attr.volume_idx = 1
            real = volume_renderer(peek_pose)
            real_2d = vr_attr.render_slice(*peek_pose_2d)

        vr_attr.tf_mode = 'custom'
        vr_attr.volume_idx = 0
        fake = volume_renderer(peek_pose)
        fake_2d = vr_attr.render_slice(*peek_pose_2d)
        
        initial_losss = loss_l1(real, fake)
        initial_losss_2d = loss_l1(real_2d, fake_2d)
        
        writer.add_scalar('Loss', initial_losss, 0)
        writer.add_scalar('Loss 2D', initial_losss_2d, 0)
        
        writer.add_image('Peek', fake[0].cpu().clamp(0., 1.), 0)
        writer.add_image('Peek 2D', fake_2d[0], 0)
        
        writer.add_figure('TF', visualize_transfer_func(vr_attr.custom_architecture,
                                                        domain=(-0.1, 1.),
                                                        grad_domain=(-0.1, grad_max),
                                                        device=device), 0)


for epoch in range(epoches):
    print("Generating new camera positions")
    poses = look_at_sphere(num_views, radius=radius,
                       target=torch.tensor([0., 0., 0.], device=device),
                       up=None)
    batch_loss = 0
    batch_loss_2d = 0.
    for i in range(iterations):
        optimizer.zero_grad()
        b_poses = poses[i*batch_size:(i+1)*batch_size]
        b_dim_2d = get_random_dim()
        b_index_2d = get_random_indices(b_dim_2d)
        vr_attr.bg = torch.rand(batch_size//ngpu, 3, device=device)
        

        with torch.no_grad():
            vr_attr.tf_mode = 'custom2'
            vr_attr.volume_idx = 1
            real = volume_renderer(b_poses)
            real_2d = vr_attr.render_slice(b_dim_2d, b_index_2d)

        vr_attr.tf_mode = 'custom'
        vr_attr.volume_idx = 0
        fake = volume_renderer(b_poses)
        fake_2d = vr_attr.render_slice(b_dim_2d, b_index_2d)
        
        if epoch > 0:
            loss = loss_l1(real, fake) + loss_vgg(input=fake, target=real,
                                                        feature_layers=[], style_layers=[0, 1, 2, 3],
                                                        wstyle_layers=[0.4, 0.8, 1.2, 1.6], style_coe=500.,
                                                        style_normalize=True, style_metric='l2')
            
            loss_2d = loss_l1(real_2d, fake_2d) + loss_vgg(input=fake_2d, target=real_2d,
                                                            feature_layers=[], style_layers=[0, 1, 2, 3],
                                                            wstyle_layers=[0.4, 0.8, 1.2, 1.6], style_coe=100.,
                                                            style_normalize=True, style_metric='l2')
        else:
            loss = loss_l1(real, fake)
            loss_2d = loss_l1(real_2d, fake_2d)
            
        total_loss = (loss + loss_2d) * 0.5
        
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        with torch.no_grad():
            batch_loss += loss
            batch_loss_2d += loss_2d
            
        if i % verbose_n_batch == verbose_n_batch - 1:
            with torch.no_grad():
                # Save TF 
                h_axis = (epoch * iterations + i + 1) * batch_size
                
                # Write in tensorboard
                with vr_attr.brighten():
                    avg_loss = batch_loss / verbose_n_batch
                    avg_loss_2d = batch_loss_2d / verbose_n_batch
                    print(f"Epoch: {epoch+1}/{epoches}\tIteration: {i+1}/{iterations}\tAvg Loss/Batch: {avg_loss:.7f} "+\
                                f"Loss/Batch 2d: {avg_loss_2d:.7f}")
                    
                    writer.add_scalar('Loss', avg_loss, h_axis)
                    writer.add_scalar('Loss 2D', avg_loss_2d, h_axis)
                    
                    vr_attr.bg = bg_color
                    writer.add_image('Peek', volume_renderer(peek_pose)[0].cpu().clamp(0., 1.), h_axis)
                    writer.add_image('Peek 2D', vr_attr.render_slice(*peek_pose_2d)[0], h_axis)
                    writer.add_figure('TF', visualize_transfer_func(vr_attr.custom_architecture,
                                                        domain=(-0.1, 1.),
                                                        grad_domain=(-0.1, grad_max),
                                                        device=device), h_axis)
                    batch_loss = 0.
                    batch_loss_2d = 0.


print("Saving the model")
os.makedirs('../../saved_models', exist_ok=True)
torch.save(vr_attr.state_dict(), os.path.join('../../saved_models', comment+'.pt'))

print("Saving transfer function")
os.makedirs('../../saved_tf', exist_ok=True)
density_tf, gradient_opacity = adaptive_tf_sampler(vr_attr.custom_architecture,
                                  threshold=0.05,
                                  initial_number_samples=100,
                                  normalization_min=rv_min,
                                  normalization_max=rv_max,
                                  domain=(0., 1.),
                                  sample_gradien=True,
                                  grad_normalization_min=0.,
                                  grad_normalization_max=(rv_max-rv_min),
                                  grad_domain=(0., grad_max),
                                  pass_none=True,
                                  device=device)

with open(os.path.join('../../saved_tf', comment+'.pickle'), 'wb') as f:
    pickle.dump({'color_opacity_tf':density_tf.cpu().numpy(), 'gradient_tf':gradient_opacity.cpu().numpy(),
                'rv_min':rv_min.item(), 'rv_max':rv_max.item(), 'gradient_max':(grad_max*(rv_max-rv_min)).item()}, f)


print("Saving 3D slicer .vp")
os.makedirs('../../3D_slicer_tf', exist_ok=True)
density_tf, gradient_opacity = adaptive_tf_sampler(vr_attr.custom_architecture,
                                  threshold=0.05,
                                  initial_number_samples=40,
                                  normalization_min=rv_min,
                                  normalization_max=rv_max,
                                  domain=(0., 1.),
                                  sample_gradien=True,
                                  grad_normalization_min=0.,
                                  grad_normalization_max=(rv_max-rv_min),
                                  grad_domain=(0., grad_max),
                                  pass_none=True,
                                  device=device)

write_slicer_tf(density_tf, os.path.join('../../3D_slicer_tf', comment+'.vp'), first_six_values,
           gradient = gradient_opacity)
        
writer.flush()

torch.cuda.empty_cache()


# Final results
image_size = 500
focal_coe = image_size / 32.
volume_renderer = VolumeRenderer4(
    image_size, image_size, focal, focal_coe,
    t_near, t_far, tf_mode='custom2', volume=None, t_delta=0.005)

vr_attr = volume_renderer
vr_attr.custom_architecture = trainable_custom_tf
vr_attr.volume = target
volume_renderer.to(device)

# Visualize target volume with its final TF
with torch.no_grad():
    with vr_attr.brighten():
        vr_attr.tf_mode='custom'
        vr_attr.volume_idx = 0
        volume_renderer.to(device)
        fig, axs = plt.subplots(2, 2, figsize=(9,5))
        
        # visualize the volume
        axs[0, 0].imshow(tensor2image(volume_renderer(peek_pose)))
        axs[0, 0].axis(False)
        axs[0, 0].set_title("3D View")
        
        # visualize 2D slice
        slices = vr_attr.render_slice(*peek_pose_2d)
        axs[0, 1].imshow(tensor2image(slices))
        axs[0, 1].axis(False)
        axs[0, 1].set_title("2D Slice")

        # Visualize scalar and gradient transfer functions
        density_tf, gradient_opacity = adaptive_tf_sampler(vr_attr.custom_architecture,
                                                threshold=0.05,
                                                initial_number_samples=100,
                                                normalization_min=rv_min,
                                                normalization_max=rv_max,
                                                domain=(0., 1.),
                                                sample_gradien=True,
                                                grad_normalization_min=0.,
                                                grad_normalization_max=(rv_max-rv_min),
                                                grad_domain=(0., grad_max),
                                                pass_none=True,
                                                device=device)
        axs[1, 0].set_title("Scalar Opacity TF")
        tf_visualizer_scalar(axs[1, 0], density_tf)
        axs[1, 1].set_title("Gradient Opacity TF")
        tf_visualizer_gradient(axs[1, 1], gradient_opacity)

        plt.suptitle('The final transfer function applied on registered target volume.')
        plt.tight_layout()
        plt.pause(2)


# The origianl target volume (Philips) without registeration
original, org_rv_min, org_rv_max, org_grad_max = volume_loader('../../volumes/original/CC0087_philips_3_42_M.nii.gz',
                              with_grad=True,
                              bad_spacing=True,
                              normalize_scalars=True)
vr_attr.volume = original.to(device)
volume_renderer.to(device)

# Visualizing original target volume (Philips) with final transfer functions
with torch.no_grad():
    with vr_attr.brighten():
        vr_attr.tf_mode = 'custom'
        vr_attr.volume_idx = 0
        volume_renderer.to(device)
        fig, axs = plt.subplots(2, 2, figsize=(9,5))

        # visualize the volume
        axs[0, 0].imshow(tensor2image(volume_renderer(peek_pose)))
        axs[0, 0].axis(False)
        axs[0, 0].set_title("3D View")

        # visualize 2D slice
        slices = vr_attr.render_slice(*peek_pose_2d)
        axs[0, 1].imshow(tensor2image(slices))
        axs[0, 1].axis(False)
        axs[0, 1].set_title("2D Slice")

        # Visualize scaler and gradient transfer functions
        density_tf, gradient_opacity = adaptive_tf_sampler(vr_attr.custom_architecture,
                                                threshold=0.05,
                                                initial_number_samples=100,
                                                normalization_min=org_rv_min,
                                                normalization_max=org_rv_max,
                                                domain=(0., 1.),
                                                sample_gradien=True,
                                                grad_normalization_min=0.,
                                                grad_normalization_max=(org_rv_max-org_rv_min),
                                                grad_domain=(0., org_grad_max),
                                                pass_none=True,
                                                device=device)
        axs[1, 0].set_title("Scalar Opacity TF")
        tf_visualizer_scalar(axs[1, 0], density_tf)
        axs[1, 1].set_title("Gradient Opacity TF")
        tf_visualizer_gradient(axs[1, 1], gradient_opacity)

        plt.suptitle('The final transfer function applied on original target volume.')
        plt.tight_layout()
        plt.show()