'''
 * The Inference of RAM and Tag2Text Models
 * Written by Xinyu Huang
'''
import torch




def inference_ram(image, model):

    with torch.no_grad():
        tags, tags_chinese = model.generate_tag(image)

    return tags[0],None


