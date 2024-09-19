import torch
import torchvision.models

# Ref: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
# Ref: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, before_relu=True, resize=True):
        assert (type(before_relu) is bool), "The argument before_relu must be of type boolean"
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        if before_relu == False:
            blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[23:30].eval())
            
        else:
            blocks.append(torch.nn.Sequential(*[l if not(isinstance(l, torch.nn.ReLU)) else torch.nn.ReLU(inplace=False) for
                         l in torchvision.models.vgg16(pretrained=True).features[:3].eval()]))
            blocks.append(torch.nn.Sequential(*[l if not(isinstance(l, torch.nn.ReLU)) else torch.nn.ReLU(inplace=False) for
                         l in torchvision.models.vgg16(pretrained=True).features[3:8].eval()]))
            blocks.append(torch.nn.Sequential(*[l if not(isinstance(l, torch.nn.ReLU)) else torch.nn.ReLU(inplace=False) for
                         l in torchvision.models.vgg16(pretrained=True).features[8:15].eval()]))
            blocks.append(torch.nn.Sequential(*[l if not(isinstance(l, torch.nn.ReLU)) else torch.nn.ReLU(inplace=False) for
                         l in torchvision.models.vgg16(pretrained=True).features[15:22].eval()]))
            blocks.append(torch.nn.Sequential(*[l if not(isinstance(l, torch.nn.ReLU)) else torch.nn.ReLU(inplace=False) for
                         l in torchvision.models.vgg16(pretrained=True).features[22:29].eval()]))
            
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[], wstyle_layers=[], style_coe=1., style_normalize=True, style_metric='l2'):
    	# If the input has one channel grayscale, make it three channels
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        # Apply ImageNet normalization on input and target image
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        
        # Resize the images to trained input size of VGG
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        # Calculate feature and style losses
        feature_loss = 0.
        style_loss = 0.
        if not wstyle_layers:
        	wstyle_layers = [1.] * len(style_layers)
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            
            if i in feature_layers:
                feature_loss += torch.nn.functional.l1_loss(x, y)
            
            if i in style_layers:
                N, C, H, W = x.shape
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                
                
                if style_normalize:
                    gram_x = act_x @ act_x.permute(0, 2, 1) / (C * H * W)
                    gram_y = act_y @ act_y.permute(0, 2, 1) / (C * H * W)
                    
                else:
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_y = act_y @ act_y.permute(0, 2, 1)
                    
                if style_metric =='l1':
                    style_loss += torch.nn.functional.l1_loss(gram_x, gram_y) * wstyle_layers[style_layers.index(i)]
                
                elif style_metric == 'l2':
                    style_loss += torch.nn.functional.mse_loss(gram_x, gram_y) * wstyle_layers[style_layers.index(i)]
                else:
                    raise Exception("metric should be either 'l1' or 'l2'.")
        loss = feature_loss + style_coe * style_loss          
        return loss
    
