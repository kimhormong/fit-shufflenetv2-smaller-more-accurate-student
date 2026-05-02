import torch
import torch.nn as nn
import torch.nn.functional as F

class FitNetWrapper(nn.Module):
    def __init__(self, teacher, student, hint_layer_name, guided_layer_name, student_channels, teacher_channels):
        super(FitNetWrapper, self).__init__()
        self.teacher = teacher
        self.student = student
        self.regressor = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU(inplace=True)
        )
        self.hint_output = None
        self.guided_output = None
        self._register_hooks(hint_layer_name, guided_layer_name)

    def _register_hooks(self, hint_name, guided_name):
        def get_activation(is_teacher):
            def hook(model, input, output):
                if is_teacher: self.hint_output = output
                else: self.guided_output = output
            return hook
        dict(self.teacher.named_modules())[hint_name].register_forward_hook(get_activation(True))
        dict(self.student.named_modules())[guided_name].register_forward_hook(get_activation(False))

    def forward(self, x):
        s_logits = self.student(x)
        with torch.no_grad():
            t_logits = self.teacher(x)
    
        # Get raw guided features
        guided_features = self.guided_output 
    
        # 1. Apply convolutional regressor (fixes channels) 
        # apply the regressor to make spatial features from student match teacher's hind number of channels
        regressed_student = self.regressor(guided_features)
    
    
        # 2. Fix Spatial Mismatch (Height/Width)
        # If the teacher hint is 14x14 and student is 28x28, this resizes student to 14x14
        if regressed_student.shape[2:] != self.hint_output.shape[2:]:
            
            # Resize the regressed student features to match the hint output size by bilinear interpolation which do things as follows:
            # 1. For each pixel in the output feature map, it identifies the corresponding region in the input feature map.
            # 2. It then computes the output pixel value by taking a weighted average of the four nearest pixels in the input feature map, where the weights are determined by the distance from the output pixel to the input pixels.
            # 
            regressed_student = F.interpolate(
            regressed_student, 
            size=self.hint_output.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return s_logits, t_logits, regressed_student, self.hint_output