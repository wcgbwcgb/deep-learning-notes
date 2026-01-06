from torch import nn
import ScaledDotProductAttention

class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):  # d_model: model dimension
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, num_tokens, dimension = x.size()

        # projected versions
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        # dividing into heads
        query = query.view(batch_size, num_tokens, self.num_heads, self.d_k).transpose(1,2)
        key = key.view(batch_size, num_tokens, self.num_heads, self.d_k).transpose(1,2)
        value = value.view(batch_size, num_tokens, self.num_heads, self.d_k).transpose(1,2)
        # shape: [batch_size, num_tokens, num_heads, d_k] => [batch_size, num_heads, num_tokens, d_k]
        # In pytorch, only the last two term will be matmul, others will be considered as batch_dimension

        # attention
        att_weights = self.attention(query, key, value)

        # concat
        # must use contiguous() after transpose in order to use view()
        att_weights = att_weights.transpose(1,2).contiguous().view(batch_size, num_tokens, self.d_model)

        # Final Linear Layer
        output = self.W_o(att_weights)

        return output