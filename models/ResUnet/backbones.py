from typing import List, Tuple, Union, Dict

import numpy as np
from torch import nn
import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
import time
import os

class SparseConvBlock(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 dilation: int = 1) -> None:
        super().__init__(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        dilation=dilation),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
        )


class SparseConvTransposeBlock(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 dilation: int = 1) -> None:
        super().__init__(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        dilation=dilation,
                        transposed=True),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
        )


class SparseResBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation),
            spnn.BatchNorm(out_channels),
        )

        if in_channels != out_channels or np.prod(stride) != 1:
            self.shortcut = nn.Sequential(
                spnn.Conv3d(in_channels, out_channels, 1, stride=stride),
                spnn.BatchNorm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = spnn.ReLU(True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        x = self.relu(self.main(x) + self.shortcut(x))
        return x


class SparseResUNet(nn.Module):

    def __init__(
            self,
            stem_channels: int,
            encoder_channels: List[int],
            decoder_channels: List[int],
            *,
            in_channels: int = 4,
            width_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.stem_channels = stem_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.in_channels = in_channels
        self.width_multiplier = width_multiplier

        num_channels = [stem_channels] + encoder_channels + decoder_channels
        num_channels = [int(width_multiplier * nc) for nc in num_channels]

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, num_channels[0], 3),
            spnn.BatchNorm(num_channels[0]),
            spnn.ReLU(True),
            spnn.Conv3d(num_channels[0], num_channels[0], 3),
            spnn.BatchNorm(num_channels[0]),
            spnn.ReLU(True),
        )

        self.encoders = nn.ModuleList()
        for k in range(4):
            self.encoders.append(
                nn.Sequential(
                    SparseConvBlock(
                        num_channels[k],
                        num_channels[k],
                        2,
                        stride=2,
                    ),
                    SparseResBlock(num_channels[k], num_channels[k + 1], 3),
                    SparseResBlock(num_channels[k + 1], num_channels[k + 1], 3),
                ))

        self.decoders = nn.ModuleList()
        for k in range(4):
            self.decoders.append(
                nn.ModuleDict({
                    'upsample':
                        SparseConvTransposeBlock(
                            num_channels[k + 4],
                            num_channels[k + 5],
                            2,
                            stride=2,
                        ),
                    'fuse':
                        nn.Sequential(
                            SparseResBlock(
                                num_channels[k + 5] + num_channels[3 - k],
                                num_channels[k + 5],
                                3,
                            ),
                            SparseResBlock(
                                num_channels[k + 5],
                                num_channels[k + 5],
                                3,
                            ),
                        )
                }))
        self.conv0 = spnn.Conv3d(
            96,
            3,
            1,
            stride=1,
        )
        self.conv1 = spnn.Conv3d(
            96,
            4,
            1,
            stride=1,
        )
        self.conv2 = spnn.Conv3d(
            96,
            6,
            1,
            stride=1,
        )
        self.conv3 = spnn.Conv3d(
            96,
            9,
            1,
            stride=1,
        )
        self.conv4 = spnn.Conv3d(
            96,
            15, 
            1,
            stride=1,
        )

    def _unet_forward(
            self,
            x: SparseTensor,
            encoders: nn.ModuleList,
            decoders: nn.ModuleList,
    ) -> List[SparseTensor]:
        if not encoders and not decoders:
            return [x]

        xd = encoders[0](x)

        outputs = self._unet_forward(xd, encoders[1:], decoders[:-1])
        yd = outputs[-1]

        u = decoders[-1]['upsample'](yd)
        y = decoders[-1]['fuse'](torchsparse.cat([u, x]))

        return [x] + outputs + [y]

    def forward(self, x: SparseTensor) -> List[SparseTensor]:
        output = self._unet_forward(self.stem(x), self.encoders, self.decoders)
        return [self.conv0(output[-1]), self.conv1(output[-1]), self.conv2(output[-1]), self.conv3(output[-1]),
                self.conv4(output[-1])]



class SparseMultiResUNet(nn.Module):
    def __init__(
            self,
            stem_channels: int,
            encoder_channels: List[int],
            decoder_channels: List[int],
            *,
            in_channels: int = 4,
            width_multiplier: float = 1.0,
            num_classes: list = [3, 4, 7,8, 14],
    ) -> None:
        super().__init__()
        self.stem_channels = stem_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.in_channels = in_channels
        self.width_multiplier = width_multiplier

        num_channels = [stem_channels] + encoder_channels + decoder_channels
        num_channels = [int(width_multiplier * nc) for nc in num_channels]

        self.counter = 0

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, num_channels[0], 3),
            spnn.BatchNorm(num_channels[0]),
            spnn.ReLU(True),
            spnn.Conv3d(num_channels[0], num_channels[0], 3),
            spnn.BatchNorm(num_channels[0]),
            spnn.ReLU(True),
        )

        self.encoders = nn.ModuleList()
        for k in range(4):
            self.encoders.append(
                nn.Sequential(
                    SparseConvBlock(
                        num_channels[k],
                        num_channels[k],
                        2,
                        stride=2,
                    ),
                    SparseResBlock(num_channels[k], num_channels[k + 1], 3),
                    SparseResBlock(num_channels[k + 1], num_channels[k + 1], 3),
                ))

        self.multi_decoders = nn.ModuleList()
        self.convs = nn.ModuleList()
        for num_cls in num_classes:
            decoders = nn.ModuleList()
            for k in range(4):
                decoders.append(
                    nn.ModuleDict({
                        'upsample':
                            SparseConvTransposeBlock(
                                num_channels[k + 4],
                                num_channels[k + 5],
                                2,
                                stride=2,
                            ),
                        'fuse':
                            nn.Sequential(
                                SparseResBlock(
                                    num_channels[k + 5] + num_channels[3 - k],
                                    num_channels[k + 5],
                                    3,
                                ),
                                SparseResBlock(
                                    num_channels[k + 5],
                                    num_channels[k + 5],
                                    3,
                                ),
                            )
                    }))
            self.multi_decoders.append(decoders)
            self.convs.append(spnn.Conv3d(
                                num_channels[-1],
                                num_cls,
                                1,
                                stride=1,
                            ))
            
    def _encoder_forward(self, x: SparseTensor,
            encoders: nn.ModuleList,):
        outputs = [x]
        for i, encoder in enumerate(encoders):
            x = encoder(x)
            outputs.append(x)
        return outputs
    
    def _decoder_forward(self, codes: List,
            decoders: nn.ModuleList,):
        y = codes[-1] 
        codes = codes[:-1]
        for x, decoder in zip(codes[::-1], decoders):
            u =  decoder['upsample'](y)
            y = decoder['fuse'](torchsparse.cat([u, x]))
        return y
        
    def _unet_forward(self, x: SparseTensor,
            encoders: nn.ModuleList,
            multi_decoders: List,):
        outputs = self._encoder_forward(x, encoders)
        return [self._decoder_forward(outputs, dcs) for dcs in multi_decoders]
        
    
    def forward(self, x: Dict) -> Dict:
        input = x['sparse_points']
        outputs = self._unet_forward(self.stem(input), self.encoders, self.multi_decoders)
        if self.counter == 0:
            current_time = time.time()
            os.system('mkdir ' + str(current_time))
            self.dir_name = str(current_time)
        if self.counter % 3 == 0:
            output_list = [output.F.cpu().numpy() for output in outputs]
            for i in range(len(output_list)):
                print(str(self.dir_name + '/' + str(self.counter) + '_' + str(i) + '.npy'))
                np.save(self.dir_name + '/' + str(self.counter) + '_' + str(i) + '.npy', output_list[i])
            np.save(self.dir_name + '/' + str(self.counter) + '_labels.npy', x['sparse_labels'].F.cpu().numpy())
        self.counter += 1
        logits = [conv(output) for conv, output in zip(self.convs, outputs)]
        return {'logits': None, 'sparse_logits': logits}
    
class HAModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.h_matrices = kwargs['h_matrices']
        self.emb_dims = kwargs['emb_dims']
        self.up_to_down = None
    
    def forward(self, x: List[SparseTensor]) -> List:
        emb_up = x[0]
        emb_down = x[1]
        
    

class SparseResUNet42(SparseResUNet):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            stem_channels=32,
            encoder_channels=[32, 64, 128, 256],
            decoder_channels=[256, 128, 96, 96],
            **kwargs,
        )
        
class SparseMultiResUNet42(SparseMultiResUNet):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            stem_channels=32,
            encoder_channels=[32, 64, 128, 256],
            decoder_channels=[256, 128, 96, 96],
            **kwargs,
        )
