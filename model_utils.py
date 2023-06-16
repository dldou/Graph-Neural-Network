import torch
import torch.nn as nn


class EncoderUNetType(nn.Module):

    def __init__(self, in_channels, out_channels, depth, **kwargs):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        
        # model building
        self.mid_channels_list = [32 * 2**i for i in range(depth + 1)]
        self.encoder           = nn.ModuleList()

        # first layer
        self.input_layer  = DoubleConvolution(self.in_channels, self.mid_channels_list[0], **kwargs)
        for i in range(depth):
            self.encoder.append(
                Block_down_unet(self.mid_channels_list[i], self.mid_channels_list[i + 1], **kwargs)
                )
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.encoder(x)
        print(x.shape)
        return x



class Block_down_unet(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential (
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConvolution(self.in_channels, self.out_channels, **kwargs)
        )

    def forward(self, x):
        return self.block(x)



class DoubleConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        """
            kwargs: - batchnorm (bool), add classic 2D batchnorm
                    - dropout (bool), add dropout only if dropout_val also specified
                    - dropout_val (float), between 0 and 1 specify the dropout value on all dropout layers
                    - leaky_relu (bool), choose Leaky ReLU as activation function (alpha set by default)
                    - kernel_size_list (list of int), first convolution
                    - padding_list (list of int), padding in first convolution
        """
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        # layers
        self.double_conv = nn.Sequential()
        prev_channels    = self.in_channels

        for i in range(2):
            if 'kernel_size_list' in kwargs:
                if 'padding_list' in kwargs:
                    self.double_conv.append(nn.Conv2d(prev_channels, self.out_channels, kernel_size=kwargs.get('kernel_size_list')[i], 
                                                      padding=kwargs.get('padding_list')[i]))
                else:
                    self.double_conv.append(nn.Conv2d(prev_channels, self.out_channels, kernel_size=kwargs.get('kernel_size_list')[i], 
                                                      padding=1))
            else:
                self.double_conv.append(nn.Conv2d(prev_channels, self.out_channels, kernel_size=3, padding=1))

            if 'batchnorm' in kwargs and kwargs.get('batchnorm'):
                self.double_conv.append(nn.BatchNorm2d(self.out_channels))

            if ( 'dropout' in kwargs and kwargs.get('dropout') ) and 'dropout_val' in kwargs:
                self.double_conv.append(nn.Dropout2d(p=kwargs.get('dropout_val'), inplace=True))

            # activation
            if 'leaky_relu' in kwargs and kwargs.get('leaky_relu'):
                self.double_conv.append(nn.LeakyReLU(inplace=True))
            else:
                self.double_conv.append(nn.ReLU(inplace=True))

            prev_channels = self.out_channels


    def forward(self, x):
        return self.double_conv(x)