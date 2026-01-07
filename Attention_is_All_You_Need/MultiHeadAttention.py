from torch import nn
from ScaledDotProductAttention import MyScaledProductAttention

class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):  # d_model: model dimension
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.attention = MyScaledProductAttention()

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query_input, key_input, value_input, mask=None):
        q_batch_size, q_num_tokens, dimension = query_input.size()
        k_batch_size, k_num_tokens, dimension = key_input.size()

        # projected versions
        query = self.W_q(query_input)
        key = self.W_k(key_input)
        value = self.W_v(value_input)

        # dividing into heads
        query = query.view(q_batch_size, q_num_tokens, self.num_heads, self.d_k).transpose(1,2)
        key = key.view(k_batch_size, k_num_tokens, self.num_heads, self.d_k).transpose(1,2)
        value = value.view(k_batch_size, k_num_tokens, self.num_heads, self.d_k).transpose(1,2)
        # shape: [batch_size, num_tokens, num_heads, d_k] => [batch_size, num_heads, num_tokens, d_k]
        # In pytorch, only the last two term will be matmul, others will be considered as batch_dimension

        # attention
        att_output = self.attention(query, key, value, mask)

        # concat
        # must use contiguous() after transpose in order to use view()
        att_output = att_output.transpose(1,2).contiguous().view(q_batch_size, q_num_tokens, self.d_model)

        # Final Linear Layer
        output = self.W_o(att_output)

        return output