import torch

import torch.nn as nn
'''
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key_value):
        # query: Tensor of shape (batch_size, channels, height, width)
        # key_value: Tensor of shape (batch_size, channels, height, width)

        # Reshape query tensor for cross-attention
        query = query.view(query.size(0), -1, query.size(-1)).permute(2, 0, 1)  # (height*width, batch_size, channels)

        # Reshape key_value tensor for cross-attention
        key_value = key_value.view(key_value.size(0), -1, key_value.size(-1)).permute(2, 0, 1)  # (height*width, batch_size, channels)

        # Multihead Attention
        output, _ = self.attention(query, key_value, key_value)

        # Reshape output back to the original shape
        output = output.permute(1, 2, 0).view(query.size(1), -1, query.size(0))  # (batch_size, channels, height*width)
        output = output.view(query.size(1), -1, query.size(-1), query.size(-1))  # (batch_size, channels, height, width)

        return output


import tensorflow as tf

def cross_attention_3d(tensor1, tensor2):
    """
    Compute 3D cross attention between two tensors
    """
    attention_weights = tf.matmul(tensor1, tensor2, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_weights, axis=-1)
    output = tf.matmul(attention_weights, tensor2)
    new_tensor = output[..., :128]  # Taking the first 128 elements along the last dimension
    extra_tensor = output[..., 128:] 
    new_os = torch.cat([new_tensor, extra_zeros], dim=-1)
    return output

def merge_tensors_3d(tensor1, tensor2):

    attention_output = cross_attention_3d(tensor1, tensor2)
    merged_tensor = tf.concat([tensor1, attention_output], axis=-1)
    new_tensor = merged_tensor[..., :128]  # Taking the first 128 elements along the last >    extra_tensor = output[..., 128:]
    new_os = torch.cat([new_tensor, extra_zeros], dim=-1)
    return new_os
'''
class CrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        #self.tensor1 = tensor1
        #self.tensor2 = tensor2


    def forward(self, queries, key, values):
        batch_size, _, depth, heaight, width = queries.size()
        queries = queries.permute(2, 0, 1, 3, 4).reshape(depth, -1, queries.size(-1))
        key = key.permute(2, 0, 1, 3, 4).reshape(depth, -1, key.size(-1))
        values = values.permute(2, 0, 1, 3, 4).reshape(depth, -1, values.size(-1))
        output, _ = self.attention(queries, key, values)

        output = output.view(depth, batch_size, -1, queries.size(-1), queries.size(-1))
        #print('output shape: ', output.shape)
        output = output.permute(1, 2, 0, 3, 4)
        #print('output shape: ', output.shape)
        return output


class attention_block(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(attention_block, self).__init__()
        #num_heads = 8
        #x, y, z, a, b  = input.size
        #embed_dim 

        self.attention_ish = nn.MultiheadAttention(embed_dim, num_heads)
        self.num_heads = num_heads

    def forward(self, input_img):
        #print('immg dims: ', input_img.size())

        batch_size, _, depth,height, width = input_img.size()
       # embed_dim = depth
        #num_heads = 8
        #self.attention_ish = nn.MultiheadAttention(embed_dim, num_heads)
        queries = input_img.permute(2, 0, 1, 3, 4).reshape(depth, -1, input_img.size(-1))
        keys = input_img.permute(2, 0, 1, 3, 4).reshape(depth, -1, input_img.size(-1))
        values = input_img.permute(2, 0, 1, 3, 4).reshape(depth, -1, input_img.size(-1))
        output, _ = self.attention_ish(queries, queries, queries)
        output = output.view(depth, batch_size, -1, queries.size(-1), queries.size(-1))
        output = output.permute(1, 2, 0, 3, 4)
        return output



#import tensorflow as tf
'''
def cross_attention_3d(tensor1, tensor2):
    """
    Compute 3D cross attention between two tensors
    """
    attention_weights = tf.matmul(tensor1, tensor2, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_weights, axis=-1)
    output = tf.matmul(attention_weights, tensor2)
    
    # Add padding to match the shape of tensor1
    output = tf.pad(output, [(0, 0) for _ in range(len(tensor1.shape) - 1)] + [(0, tensor1.shape[-1] - output.shape[-1],)])
    
    return output


def cross_attention_3d(tensor1, tensor2):
    """
    Compute 3D cross attention between two tensors
    """
    attention_weights = tf.matmul(tensor1, tensor2, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_weights, axis=-1)
    output = tf.matmul(attention_weights, tensor2)
    
    # Add padding to match the shape of tensor1
    output = tf.pad(output, [(0, 0) for _ in range(len(tensor1.shape) - 1)] + [(0, tensor1.shape[-1] - output.shape[-1],)])
    
    return output[:, :, :, :, :tensor1.shape[-1]]


def cross_attention_3d(tensor1, tensor2):
    """
    Compute 3D cross attention between two tensors
    """
    attention_weights = tf.matmul(tensor1, tensor2, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_weights, axis=-1)
    output = tf.matmul(attention_weights, tensor2)

    # Reshape the output tensor to match the shape of tensor1
    output = tf.reshape(output, tensor1.shape[:-1])

    return output

def merge_tensors_3d(tensor1, tensor2):
    attention_output = cross_attention_3d(tensor1, tensor2)
    merged_tensor = tf.concat([tensor1, attention_output[:, :, :, :tensor1.shape[-1]]], axis=-1)
    return merged_tensor


import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_size, attention_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.attention_size = attention_size
        self.query_conv = nn.Conv3d(input_size, attention_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.key_conv = nn.Conv3d(input_size, attention_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_conv = nn.Conv3d(input_size, input_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value):
        # Query, Key, Value shape: batch_size, input_size, depth, height, width
        ba
        output = torch.bmm(attn_scores, V.view(batch_size, self.input_size, -1))  # shape: batch_size, input_size, depth*height*width
        output = output.view(batch_size, self.input_size, value.shape[2], value.shape[3], value.shape[4])  # shape: batch_size, input_size, depth, height, width

        return output




import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionMerge(nn.Module):
    def __init__(self, input_size):
        super(CrossAttentionMerge, self).__init__()
        self.input_size = input_size
        
        # Linear transformations for query, key, and value
        self.query_conv = nn.Conv3d(input_size, input_size, kernel_size=1)
        self.key_conv = nn.Conv3d(input_size, input_size, kernel_size=1)
        self.value_conv = nn.Conv3d(input_size,1, kernel_size=1)
        
        # Additional convolution to adjust dimensionality
        self.adjust_conv = nn.Conv3d(input_size +1 , 1, kernel_size=1)
        
    def forward(self, x1, x2):
        # Project input tensors to query, key, and value
        proj_query = self.query_conv(x1)
        proj_key = self.key_conv(x2)
        print('x1 shape: ', x1.shape)
        proj_value = self.value_conv(x2)
        print('proj_val_ ', proj_value.shape)
        
        # Compute attention scores
        energy = torch.matmul(proj_query.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.input_size),
                              proj_key.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.input_size).transpose(0, 1))
        attention = F.softmax(energy, dim=1)
        
        # Apply attention to values
        out = torch.matmul(attention, proj_value.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.input_size).unsqueeze(2))
        out = out.view(x1.size())
        
        # Adjust dimensions and merge with original tensor
        adjusted_out = self.adjust_conv(torch.cat((x1, out), dim=1))
        
        
        # Compute attention scores
        energy = torch.matmul(proj_query.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.input_size),
                              proj_key.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.input_size).transpose(0, 1))
        attention = F.softmax(energy, dim=1)

        # Apply attention to values
        out = torch.matmul(attention, proj_value.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.input_size).unsqueeze(2))
        out = out.view(x1.size())

        # Adjust dimensions and merge with original tensor
        adjusted_out = self.adjust_conv(torch.cat((x1, out), dim=1))
        ############
        
        # Compute attention scores
        energy = torch.matmul(proj_query.squeeze(-1).squeeze(-1).squeeze(-1),
                              proj_key.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1).permute(0, 1, 2, 3, 4))
        attention = F.softmax(energy, dim=1)

        # Apply attention to values
        out = torch.matmul(attention, proj_value).squeeze(1)
        #out = out.unsqueeze(2).unsqueeze(3)
        # Adjust dimensions and merge with original tensor
        #x1 = x1.unsqueeze(2)
        print('tensor shape: ', x1.shape, out.shape, proj_query.shape, energy.shape)
        adjusted_out = self.adjust_conv(torch.cat((x1, out), dim=1)) 
        return adjusted_out




def encoding(string):
    chars = sorted(list(set(string)))
    vocab_size = len(chars)
    print(''.join(chars))
    print('vocab size: ', vocab_size)
    mapping = {ch:i for i,ch in enumerate(chars)}
    encoded = lambda s: [mapping[c] for c in s]
   print('encoded: ', encoded)


def encoding(text, dose):
    collective_string = text + ' ' + dose
    vocabulary = 'abcdefghijklmnopqrstuvwxyz,._ -'
    chars = sorted(list(set(collective_string)))
    vocab = sorted(list(set(vocabulary)))
    size = len(vocab)
    vocab_size = len(chars)
    print(''.join(chars))
    print('vocab size: ', size)
    mapping = {ch:i for i,ch in enumerate(chars) }
    encoded = lambda s: [mapping[c] for c in s]
    final_Encoding = encoded(text)
    print('encoded: ', final_Encoding)
    data = torch.tensor(encoded(text), dtype=torch.float32)
    dat = data.resize_(128, 128, 128)
    return dat
'''
def encoding(text, dose):
    collective_string = text + ' ' + dose
    vocabulary = 'abcdefghijklmnopqrstuvwxyz,._ -1234567890'
    chars = sorted(list(set(collective_string)))
    vocab = sorted(list(set(vocabulary)))
    size = len(vocab)
    vocab_size = len(chars)
    #print(''.join(chars))
    #print('vocab size: ', size)
    mapping = {ch:i for i,ch in enumerate(vocab) }
    encoded = lambda s: [mapping[c] for c in s]
    #final_Encoding = encoded(collective_string)
    #print('encoded: ', final_Encoding)
    data = torch.tensor(encoded(collective_string), dtype=torch.float32)
    data = data/10
    repeat_factor = 2097152 // data.numel() 
    repeat_tensor = data.repeat(repeat_factor+1)
    repeat_tensor = repeat_tensor[:-1]
    dat = torch.reshape(repeat_tensor, (1, 1, 128, 128, 128))
    #dat = dat.unsqueeze(0).unsqueeze(0)
    return dat





class attention_block_nlp(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(attention_block, self).__init__()
        #num_heads = 8
        #x, y, z, a, b  = input.size
        #embed_dim

        self.attention_ish = nn.MultiheadAttention(embed_dim, num_heads)
        self.num_heads = num_heads
        self.enncode = encoding()

    def forward(self, input_img, ):
        #print('immg dims: ', input_img.size())

        batch_size, _, depth,height, width = input_img.size()
       # embed_dim = depth
        #num_heads = 8
        #self.attention_ish = nn.MultiheadAttention(embed_dim, num_heads)
        queries = input_img.permute(2, 0, 1, 3, 4).reshape(depth, -1, input_img.size(-1))
        keys = input_img.permute(2, 0, 1, 3, 4).reshape(depth, -1, input_img.size(-1))
        values = input_img.permute(2, 0, 1, 3, 4).reshape(depth, -1, input_img.size(-1))
        output, _ = self.attention_ish(queries, queries, queries)
        output = output.view(depth, batch_size, -1, queries.size(-1), queries.size(-1))
        output = output.permute(1, 2, 0, 3, 4)
        return output


