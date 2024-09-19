import os
import torch
import sys
sys.path.append('..')
from modules.render import *
from utils.utilities import *

def read_slicer_tf(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    colors = [float(x) for x in lines[8].strip().split()[1:]]
    colors = torch.tensor([colors[i:i+4] + [0.,] for \
                        i in range(0, len(colors), 4)])

    opacities = [float(x) for x in lines[6].strip().split()[1:]]
    opacities = torch.tensor([[opacities[i]] + [0.,]*3 + [opacities[i+1]] for \
                        i in range(0, len(opacities), 2)])
    
    gradients = [float(x) for x in lines[7].strip().split()[1:]]
    gradients = torch.tensor([[gradients[i]] + [gradients[i+1]] for \
                        i in range(0, len(gradients), 2)])
    # Since negative density is masked to background color,
    # I add minimum density to density column
    minimum = min(colors[0][0], opacities[0][0]).item()
    if minimum < 0.:
        colors[:, 0] -= minimum
        opacities[:, 0] -= minimum
    
    volume_renderer = VolumeRenderer4(1, 1, 1., 1., 1., 1., tf_mode='truth', volume=None)
    vr_attr = volume_renderer
    
    vr_attr.gt_plinear_tf = opacities
    colors[:, -1] = vr_attr._ground_truth_tf(colors[:, [0]], None)[:, -1]

    vr_attr.gt_plinear_tf = colors
    opacities[:, 1:-1] = vr_attr._ground_truth_tf(opacities[:, [0]], None)[:, :-1]

    # Removing duplicates
    tf = torch.cat([opacities,
                    colors[[torch.all(elm != opacities[:,0]).item() for elm in colors[:,0]]]],
                    0)
    
    # Put minimum back to what it was
    if minimum < 0.:
        tf[:, 0] += minimum
    
    _, idx = tf[:, 0].sort()
    return tf[idx], gradients

def write_slicer_tf(tf, path, first_six_values, gradient=None, text_precision=3):
    assert len(first_six_values) == 6, "Six values should be provided" 
    len_tf = tf.shape[0]
    first_six_values = [str(x) for x in first_six_values]
    with open(path, 'w') as f:
        f.write('\n'.join(first_six_values)+'\n')
        tf_string = np.array2string(tf.cpu().numpy(), max_line_width=1e12, threshold=1e15, 
                                    formatter={'float_kind':lambda x: ("{:.{}f}".format(x, text_precision)).rstrip("0")}).split('\n')
        tf_string = [x.replace("[", "").replace("]", "").strip().split() for x in tf_string]
        
        opacity_string = f"{len_tf*2} "
        for i in tf_string:
            opacity_string += (i[0] + ' ' + i[-1] + ' ')
        f.write(opacity_string.strip()+'\n')
        
        if gradient is None:
            gradient_string = "4 0 1 255 1"
            f.write(gradient_string+'\n')
            
        else:  
            gradient_tf_string = np.array2string(gradient.cpu().numpy(), max_line_width=1e12, threshold=1e15, 
                                    formatter={'float_kind':lambda x: ("{:.{}f}".format(x,
                                                                       text_precision)).rstrip("0")}).split('\n')
            gradient_tf_string = [x.replace("[", "").replace("]", "").strip().split() for x in gradient_tf_string]
            
            gradient_string = f"{gradient.shape[0] * 2} "
            for i in gradient_tf_string:
                gradient_string += (i[0] + ' ' + i[-1] + ' ')
            f.write(gradient_string.strip()+'\n')
            
        
        color_string = f"{len_tf*4} "
        for i in tf_string:
            color_string += (' '.join(i[:-1]) + ' ')
        f.write(color_string.strip())
        
def write_csv_tf(tf, path):
    np.savetxt(path, tf.cpu().numpy(), delimiter=",")

def write_colormap(colortable, output_path):
    output_name = os.path.split(output_path)[1]
    ext = os.path.splitext(output_name)[1]
    if ext == '.txt':
        pass
    elif ext == '':
        output_path = output_path +'.txt'
    else:
        raise TypeError(f'Color map file should have .txt extension. Received {ext}.')
    
    if isinstance(colortable, torch.Tensor):
        colortable = colortable.numpy()
    elif isinstance(colortable, np.ndarray):
        pass
    else:
        raise TypeError(f'Argument colortable should be either numpy array or torch Tensor. Received {type(colortable)}')
    
    assert colortable.shape[1] == 4, "Expected 4 columns (scaler, R, G , B)"
    abs_path = os.path.abspath(output_path)
    num_points = colortable.shape[0]
    with open(output_path, 'w') as f:
        f.write(f'# Color procedural file {abs_path}\n')
        f.write(f'# {num_points} points\n')
        f.write(f'# position R G B\n')
        np.savetxt(f, colortable, fmt='%.3f', delimiter=' ')
