import torch.nn as nn
import torch

class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):

        super(Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        #activation function
        self.relu = nn.ReLU()
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):

        #TODO: your code here
        Bq = self.encoder_att(encoder_out)
        Ah = self.decoder_att(decoder_hidden)
        Ah = torch.unsqueeze(Ah, 1)

        sum = Bq + Ah
        sum = self.relu(sum)
        scores = self.full_att(sum)
        probabilities = self.softmax(scores)

        context = encoder_out * probabilities
        attention_weighted_encoding = context.sum(axis=1)

        return attention_weighted_encoding


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(
            self, attention_dim, decoder_dim, embed_dim, vocab_size,
            encoder_dim=512, dropout_rate=0.5):

        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.attention = Attention(
            encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=dropout_rate)

        self.decode_step = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # before: (batch_size, encoded_image_size*encoded_image_size, 512)
        mean_encoder_out = encoder_out.mean(dim=1)
        # (batch_size, 512)

        # transform 512 (dim image embeddings) in decoder dim
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        #c = h
        return h, c

    def forward(self, word, decoder_hidden_state, decoder_cell_state, encoder_out):
        #TODO: your code here

        embedded = self.embedding(word)
        att = self.attention(encoder_out, decoder_hidden_state)
        catted = torch.cat((embedded, att), 1)
        decoder_hidden_state, decoder_cell_state = self.decode_step(catted, (decoder_hidden_state, decoder_cell_state))
        decoder_hidden_state = self.dropout(decoder_hidden_state)
        scores = self.fc(decoder_hidden_state)

        return scores, decoder_hidden_state, decoder_cell_state
