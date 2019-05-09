from typing import Dict

import numpy as np

from overrides import overrides

import torch
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.trainer import Trainer

from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder

HIDDEN_DIM = 128
elmo_embedding_dim = 256


# Model in AllenNLP represents a model that is trained.
@Model.register("lstm_classifier")
class LstmClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 positive_label: int = 4) -> None:
        super().__init__(vocab)
        # We need the embeddings to convert word IDs to their vector representations
        self.word_embeddings = word_embeddings

        self.encoder = encoder

        # After converting a sequence of vectors to a single vector, we feed it into
        # a fully-connected linear layer to reduce the dimension to the total number of labels.
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))

        # Monitor the metrics - we use accuracy, as well as prec, rec, f1 for 4 (very positive)
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(positive_label)

        # We use the cross entropy loss because this is a classification task.
        # Note that PyTorch's CrossEntropyLoss combines softmax and log likelihood loss,
        # which makes it unnecessary to add a separate softmax layer.
        self.loss_function = torch.nn.CrossEntropyLoss()

    # Instances are fed to forward after batching.
    # Fields are passed through arguments with the same name.
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        # In deep NLP, when sequences of tensors in different lengths are batched together,
        # shorter sequences get padded with zeros to make them equal length.
        # Masking is the process to ignore extra zeros added by padding
        mask = get_text_field_mask(tokens)

        # Forward pass
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        return {'accuracy': self.accuracy.get_metric(reset),
                'precision': precision,
                'recall': recall,
                'f1_measure': f1_measure}


@Seq2VecEncoder.register("stacked_self_attention")
class TransformerSeq2VecEncoder(Seq2VecEncoder):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 use_positional_encoding: bool = True,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.1) -> None:
        super(TransformerSeq2VecEncoder, self).__init__()
        self.stacked_attention = StackedSelfAttentionEncoder(input_size, hidden_size, projection_dim, feedforward_hidden_dim,
                                                        num_layers, num_attention_heads, use_positional_encoding,
                                                        dropout_prob, residual_dropout_prob, attention_dropout_prob)
        self.input_dim = input_size
        self.output_dim = self.stacked_attention._attention_layers[-1].get_output_dim()

    @overrides
    def forward(self, inputs, mask):
        #print("aaaa")
        out = self.stacked_attention(inputs, mask)
        #print("bbbbbb")
        #print(out.shape)
        #print("====")
        out = out[:, -1, :]
        return out

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim



def main():
    elmo_token_indexer = ELMoTokenCharactersIndexer()

    reader = StanfordSentimentTreeBankDatasetReader(token_indexers={'tokens': elmo_token_indexer})

    train_dataset = reader.read('data/stanfordSentimentTreebank/trees/train.txt')
    dev_dataset = reader.read('data/stanfordSentimentTreebank/trees/dev.txt')

    # You can optionally specify the minimum count of tokens/labels.
    # `min_count={'tokens':3}` here means that any tokens that appear less than three times
    # will be ignored and not included in the vocabulary.
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                      min_count={'tokens': 3})

        # Use the 'Small' pre-trained model
    options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                    '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
    weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                   '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)

    # BasicTextFieldEmbedder takes a dict - we need an embedding just for tokens,
    # not for labels, which are used as-is as the "answer" of the sentence classification
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

    # Seq2VecEncoder is a neural network abstraction that takes a sequence of something
    # (usually a sequence of embedded word vectors), processes it, and returns a single
    # vector. Oftentimes this is an RNN-based architecture (e.g., LSTM or GRU), but
    # AllenNLP also supports CNNs and other simple architectures (for example,
    # just averaging over the input vectors).
    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(elmo_embedding_dim, HIDDEN_DIM, batch_first=True))

    model = LstmClassifier(word_embeddings, encoder, vocab)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=10,
                      num_epochs=20)

    trainer.train()



if __name__ == '__main__':
    main()
