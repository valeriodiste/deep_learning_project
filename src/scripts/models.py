
# Import PyTorch and its modules
import torch
import torch.nn as nn
from torch.nn import Transformer
# Import PyTorch Lightning
import pytorch_lightning as pl
# Import other modules
import random
import math
import os
import numpy as np
# Import the nltk library and its modules (for the preprocessing of the text)
from nltk.tokenize import word_tokenize
# Import the gensim library and its modules (for the Word2Vec model)
from gensim.models import Word2Vec
# Import custom modules
try:
    from src.scripts import datasets
    from src.scripts.utils import MODEL_TYPES, RANDOM_SEED, get_preprocessed_text
except ModuleNotFoundError:
    from deep_learning_project.src.scripts import datasets
    from deep_learning_project.src.scripts.utils import MODEL_TYPES, RANDOM_SEED, get_preprocessed_text

# Seed random number generators for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class Word2VecModel():

    ''' 
    Class for the Word2Vec Neural Network model (using the Gensim library).

    For more details: https://radimrehurek.com/gensim/models/word2vec.html
    '''

    def __init__(self, embeddings_size, words_window_size, min_word_frequency, learning_rate, max_epochs, save_path=None):
        '''
        Constructor of the Word2VecNN class.

        Args:
        - embeddings_size: int, the size of the output word vectors
        - words_window_size: int, the size of the window for the context words (maximum distance between the current and the predicted word within a sentence)
        - min_word_frequency: int, the minimum frequency of words in the corpus to be considered (words with a frequency lower than this value will be ignored)
        - learning_rate: float, the (initial) learning rate of the training algorithm
        - max_epochs: int, the maximum number of epochs for training the model
        - save_path: str, the path to save the trained model (if None, the model will not be saved)
        '''
        # Store the hyperparameters
        self.embeddings_size = embeddings_size
        self.words_window_size = words_window_size
        self.min_word_frequency = min_word_frequency
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        # Store the save path for the trained model
        self.save_path = save_path
        # Store (or load) the model and vocabulary
        if save_path is not None and os.path.exists(save_path):
            self.model = Word2Vec.load(save_path)
            self.vocabulary = self.model.wv
        else:
            self.model = None
            self.vocabulary = None
        # Initialize a flaf to indicate whether the model has been trained
        self.is_trained = False

    def train(self, corpus):
        '''
        Train the Word2Vec model using the provided corpus.

        Args:
        - corpus: list of lists of strings, the corpus to use for training the model
        '''
        # Train the Word2Vec model using the provided corpus
        self.model = Word2Vec(
            corpus,
            vector_size=self.embeddings_size,
            window=self.words_window_size,
            min_count=self.min_word_frequency,
            epochs=self.max_epochs,
            alpha=self.learning_rate,
            # Use CBOW (Continuous Bag of Words) as the training algorithm
            sg=0,
        )
        # Store the vocabulary for the model
        self.vocabulary = self.model.wv
        # Set the flag to indicate that the model has been trained
        self.is_trained = True
        # Save the model in the given save path
        if self.save_path is not None:
            self.model.save(self.save_path)

    def load(self):
        '''
        Load the Word2Vec model's checkpoint from the given save path.

        Returns:
        - bool, whether the model has been successfully loaded or not
        '''
        # Load the model from the given save path
        if self.save_path is not None and os.path.exists(self.save_path):
            self.model = Word2Vec.load(self.save_path)
            self.vocabulary = self.model.wv
            self.is_trained = True
            return True
        else:
            print("The Word2Vec model could not be loaded, the save path does not exist.")
            return False

    def get_embedding(self, sentence, print_debug=False):
        '''
        Get the embedding of a sentence using the Word2Vec model.

        Args:
        - sentence: string corresponding to the sentence to embed

        Returns:
        - torch.Tensor, the embedding of the sentence
        '''
        # If the model has not been trained, return None
        if not self.is_trained:
            print("The Word2Vec model has not been trained yet, returning None...")
            return None
        # Preprocess the sentence (remove stopwords and punctuation)
        sentence_to_tokenize = get_preprocessed_text(sentence)
        # Tokenize the sentence
        sentence_tokens = word_tokenize(sentence_to_tokenize)
        # Compute the word embeddings for each token of the sentence (ignoring tokens that are not in the vocabulary)
        word_embeddings = [self.vocabulary[token]
                           for token in sentence_tokens if token in self.vocabulary]
        # Compute the sentence embedding
        sentence_embedding = None
        if len(word_embeddings) == 0:
            # If no token is in the vocabulary, use a zero tensor
            sentence_embedding = torch.zeros(self.embeddings_size)
        else:
            # Compute the average embedding of the sentence
            sentence_embedding = torch.tensor(
                np.mean(word_embeddings, axis=0), dtype=torch.float32)
        if print_debug:
            print("sentence:", sentence)
            print("sentence_to_tokenize:", sentence_to_tokenize)
            print("sentence_tokens:", sentence_tokens)
            print("word_embeddings:", word_embeddings)
            print("sentence_embedding:", sentence_embedding)
        # Return the sentence embedding (as a list of floats)
        return sentence_embedding.tolist()

    def get_is_trained(self):
        '''
        Get the flag indicating whether the model has been trained.

        Returns:
        - bool, the flag indicating whether the model has been trained or not
        '''
        return self.is_trained


class SiameseNetwork(pl.LightningModule):

    ''' Class for the Siamese Network model with Triplet Margin Loss'''

    # Constructor of the class
    def __init__(self, input_size: int, output_size: int, learning_rate: float, margin: float, dropout: float, activation_function: str):
        '''
        Constructor of the SiameseNetwork class.

        Args:
        - input_size: int, the size of the input features
        - output_size: int, the size of the output features
        - learning_rate: float, the learning rate of the optimizer
        - margin: float, the margin for the Triplet Margin Loss
        - activation_function: str, the activation function to use, one of ["ReLU", "LeakyReLU"]
        '''
        # Initialize the PyTorch Lightning model (call the parent constructor)
        super(SiameseNetwork, self).__init__()
        # Store the model (Siamese Network with 2 hidden layers and Leaky ReLU activation functions, which avoids the "dying ReLU" problem of the standard ReLU function)
        activation = None
        if activation_function == "ReLU":
            activation = nn.ReLU()
        elif activation_function == "LeakyReLU":
            activation = nn.LeakyReLU(0.01)
        else:
            raise ValueError(
                f"Invalid activation function: {activation_function}")
        # Define the model (Siamese Network with 2 hidden layers and the specified activation function, apply dropout to avoid overfitting)
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            activation,
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(256, 128),
            activation,
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(128, output_size)
        )
        # Set the loss function (Triplet Margin Loss with the given margin)
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        # Store the margin, learning rate and dropout value
        self.margin = margin
        self.learning_rate = learning_rate
        self.dropout = dropout
        # Store the outputs for training and validation steps
        self.training_losses = []
        self.validation_losses = []
        # Set the model type
        self.model_type = MODEL_TYPES.SIAMESE_NETWORK

    # Pytorch lightning function to compute the forward pass of the model
    def forward(self, x):
        # Forward pass for input x
        return self.model(x)

    # Auxiliary function to compute the triplet loss (used for both the training and validation steps)
    def _step(self, batch):
        # Training or validation step for the model
        anchor, positive, negative = batch
        # Compute the outputs for the anchor, positive, and negative samples
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)
        # Compute the loss
        loss = self.triplet_loss(
            anchor_output, positive_output, negative_output)
        # Return the loss
        return loss

    # Pytorch lightning function for the training step
    def training_step(self, batch, batch_idx):
        # Training step for the model
        training_loss = self._step(batch)
        # Append the loss to the training losses list (for logging)
        self.training_losses.append(training_loss)
        # Return the loss
        return training_loss

    # Pytorch lightning function for the validation step
    def validation_step(self, batch, batch_idx):
        # Validation step for the model
        validation_loss = self._step(batch)
        # Append the loss to the validation losses list (for logging)
        self.validation_losses.append(validation_loss)
        # Return the loss
        return validation_loss

    # Pytorch lightning function to configure the optimizers of the model
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # Pytorch lightning function (optional) called at the very end of each validation epoch
    def on_validation_epoch_end(self):
        # Get the epoch number
        epoch_num = self.current_epoch
        print()
        if len(self.training_losses) > 0:
            avg_epoch_training_loss = torch.stack(self.training_losses).mean()
            self.log("avg_epoch_training_loss", avg_epoch_training_loss)
            print(f"Average training loss for epoch {epoch_num}: ",
                  avg_epoch_training_loss.item())
            self.training_losses.clear()
        if len(self.validation_losses) > 0:
            avg_epoch_validation_loss = torch.stack(
                self.validation_losses).mean()
            self.log("avg_epoch_val_loss", avg_epoch_validation_loss)
            print(f"Average validation loss for epoch {epoch_num}: ",
                  avg_epoch_validation_loss.item())
            self.validation_losses.clear()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        '''
        Constructor of the PositionalEncoding class (custom torch.nn.Module).

        This module implements the positional encoding module of the traditional Transformer architecture.

        For more details: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class DSITransformer(pl.LightningModule):

    # Enum class for the Transformer model types
    class TRANSFORMER_TYPES:
        ''' Enum class for the Transformer model types '''
        SCHEDULED_SAMPLING_TRANSFORMER = "Scheduled Sampling Transformer"
        ''' The Transformer model with scheduled sampling to reduce exposure bias (use the "scheduled_sampling_decay" parameter to control the linear decay of the probability of using the ground truth target's token) '''
        AUTOREGRESSIVE_TRANSFORMER = "Autoregressive Transformer"
        ''' The Transformer model with an autoregressive approach (i.e. generate the sequence token by token using the model's own predictions) '''
        TEACHER_FORCINIG_TRANSFORMER = "Teacher Forcing Transformer"
        ''' The Transformer model with teacher forcing (i.e. use the ground truth target's token for each prediction) '''

    def __init__(
            self, tokens_in_vocabulary: int,
            embeddings_size: int, target_tokens: int,
            transformer_heads: int, layers: int,
            dropout: float, learning_rate: float,
            batch_size: int,
            transformer_type: TRANSFORMER_TYPES,
            scheduled_sampling_decay: float = 0.05
    ):
        '''
        Constructor of the DSITransformer class.

        Args:
        - tokens_in_vocabulary: int, the number of tokens in the vocabulary
        - embeddings_size: int, the size of the embeddings
        - target_tokens: int, the number of possible target tokens for the output
        - transformer_heads: int, the number of multi-head attention heads
        - layers: int, the number of encoder and decoder layers
        - dropout: float, the dropout value
        - learning_rate: float, the learning rate of the optimizer
        - batch_size: int, the batch size
        - transformer_type: DSITransformer.TRANSFORMER_TYPES, the type of the Transformer model (scheduled sampling, autoregressive, or teacher forcing)
        - scheduled_sampling_decay: float (optional), the linear decay of the scheduled sampling probability when a scheduled sampling transformed is used (default is 0.05) 

        For more details: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        '''
        # Initialize the PyTorch Lightning model (call the parent constructor)
        super(DSITransformer, self).__init__()
        # PyTorch Lightning function to save the model's hyperparameters
        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "embedding_size": embeddings_size,
            "number_of_layers": layers,
            "dropout": dropout
        })
        # Store the input and output sizes
        self.input_size = embeddings_size
        self.target_tokens = target_tokens
        # Store the padding token (11 is used for the document ID padding token)
        self.doc_id_padding_token = 11
        # Store the model (Transformer model with the specified hyperparameters)
        self.model = Transformer(
            # Number of expected features in the encoder/decoder inputs
            d_model=embeddings_size,
            # Number of multi-head attention heads
            nhead=transformer_heads,
            # Number of encoder & decoder layers (symmetric for simplicity)
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            # Dimension of the feedforward network model (hidden layer size)
            dim_feedforward=embeddings_size,
            # Dropout value
            dropout=dropout
        )
        # Embedding layer for the input tokens (i.e. tokens in the vocabulary for both documents and queries)
        self.get_input_embedding = nn.Embedding(
            tokens_in_vocabulary, embeddings_size, padding_idx=0)
        # Embedding layer for the target tokens (output features, i.e. document IDs)
        self.get_target_embedding = nn.Embedding(
            target_tokens, embeddings_size, padding_idx=self.doc_id_padding_token)
        # Positional encoding layer ("custom" torch.nn.Module, implements the positional encoding module of the traditional Transformer architecture)
        self.positional_encoder = PositionalEncoding(embeddings_size, dropout)
        # Output layer of the model (linear layer, outputs the predictions for each target token, hence each digit of the document ID)
        self.output_layer = nn.Linear(embeddings_size, target_tokens)
        # Store the loss function (Cross Entropy Loss)
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            ignore_index=self.doc_id_padding_token)
        # Use scheduled sampling to avoid exposure bias (with a linear decay of the probability of using the ground truth target)
        self.transformer_type = transformer_type
        self.scheduled_sampling_decay = scheduled_sampling_decay
        self.scheduled_sampling_probability = 1.0
        # Store the outputs for training and validation steps
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
        # Store the model type
        self.model_type = MODEL_TYPES.DSI_TRANSFORMER

    # Pytorch lightning function to compute the forward pass of the model
    #   For more details: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward
    def forward(self, input, target):
        # Get the length of the input and target sequences
        input_length = input.size(0)
        target_length = target.size(0)
        # Create the masks for the input and target sequences
        input_mask = torch.zeros(
            (input_length, input_length), device=self.device).type(torch.bool)
        # target_mask = self.generate_square_subsequent_mask(
        #     target_length).type(torch.bool).to(self.device)
        target_mask = nn.Transformer.generate_square_subsequent_mask(
            target_length, device=self.device, dtype=torch.bool)
        # input == 0 is for padding_idx
        input_padding_mask = (input == 0).transpose(0, 1).type(torch.bool)
        # target == self.padding_idx is for padding_idx
        target_padding_mask = (target == self.doc_id_padding_token).transpose(
            0, 1).type(torch.bool)
        # Get the embeddings for the input and target sequences
        input = self.get_input_embedding(input)
        target = self.get_target_embedding(target)
        # Apply the positional encoding to the input and target sequences
        input = self.positional_encoder(input).to(self.device)
        target = self.positional_encoder(target).to(self.device)
        # Compute the output of the transformer model
        output = self.model(input, target, input_mask, target_mask,
                            None, input_padding_mask, target_padding_mask, input_padding_mask)
        # Return the final output of the model
        return self.output_layer(output)

    # Auxiliary function for both the training and valdiation steps (to compute the loss and accuracy)
    def _step(self, batch, force_autoregression=False):
        ''' 
        Generate the output document ID using an autoregressive approach (i.e. generate the sequence token by token using the model's own predictions) 

        Returns the loss and accuracy of the model for the given batch
        '''
        # Get the input and target sequences from the batch
        input, target = batch
        # Transpose the input and target sequences to match the Transformer's expected input format
        input = input.transpose(0, 1)
        target = target.transpose(0, 1)
        # Ground truth target (excluding the last token, i.e. the end token)
        target_in_true = target[:-1, :]
        # Compute the true output of the model (excluding the first token, i.e. the start token)
        output_true = self(input, target_in_true)
        # Initialize the output tensor
        output = torch.zeros(target.size(0) - 1, input.size(1),
                             self.target_tokens, device=input.device)
        # Start with the first token
        target_in = target[:1, :]
        # Iterate over the target sequence to generate the output sequence
        for i in range(1, target.size(0)):
            # Store the next token
            next_token = None
            # Check wheter to use teacher forcing for the next token or use the model's own prediction (autoregressive approach)
            use_teacher_forcing = \
                (self.transformer_type == DSITransformer.TRANSFORMER_TYPES.SCHEDULED_SAMPLING_TRANSFORMER and
                    random.random() < self.scheduled_sampling_probability) or \
                (self.transformer_type ==
                 DSITransformer.TRANSFORMER_TYPES.TEACHER_FORCINIG_TRANSFORMER)
            # Use scheduled sampling to avoid exposure bias
            if not force_autoregression and use_teacher_forcing:
                # Use the ground truth token for the next prediction
                ground_truth_token = output_true[i-1:i, :, :]
                output[i - 1] = ground_truth_token.squeeze(0)
                next_token = torch.argmax(ground_truth_token, dim=-1)
            else:
                # Generate the output for the input and target sequences (autoregressive approach)
                output_till_now = self(input, target_in)
                # Get the prediction for the last token and append it to the output tensor
                last_token_output = output_till_now[-1, :, :].unsqueeze(0)
                output[i - 1] = last_token_output.squeeze(0)
                # Use the last generated best token as the next token of the target_in sequence
                next_token = torch.argmax(last_token_output, dim=-1)
            # Append the next token to the target_in sequence
            target_in = torch.cat((target_in, next_token), dim=0)
        # Get the target output (excluding the first token, i.e. the start token)
        target_out = target[1:, :]
        # Ensure the target_out tensor is contiguous in memory (to efficiently compute the loss)
        target_out = target_out.contiguous()
        # Compute the loss
        reshaped_output = output.reshape(-1, self.target_tokens)
        reshaped_target_out = target_out.reshape(-1)
        loss = self.cross_entropy_loss(reshaped_output, reshaped_target_out)
        # Get the best token prediction (to compute the accuracy)
        predictions = torch.argmax(output, dim=-1)
        # Compute accuracy with masking for padding
        non_padding_mask = (target_out != self.doc_id_padding_token)
        num_correct = ((predictions == target_out) &
                       non_padding_mask).sum().item()
        num_total = non_padding_mask.sum().item()
        accuracy_value = num_correct / num_total if num_total > 0 else 0.0
        accuracy = torch.tensor(accuracy_value)
        # Return loss and accuracy (tensors)
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        # Training step for the model (compute the loss and accuracy)
        loss, accuracy = self._step(batch)
        # Append the loss to the training losses list (for logging)
        # self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.training_accuracies.append(accuracy)
        # Append the accuracy to the training accuracies list (for logging)
        # self.log('train_accuracy', accuracy.item(),
        #          on_epoch=True, prog_bar=True)
        self.training_losses.append(loss)
        # Return the loss
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step for the model (compute the loss and accuracy)
        loss, accuracy = self._step(batch, True)
        # Append the loss to the validation losses list (for logging)
        # self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.validation_losses.append(loss)
        # Append the accuracy to the validation accuracies list (for logging)
        # self.log('val_accuracy', accuracy.item(), on_epoch=True, prog_bar=True)
        self.validation_accuracies.append(accuracy)
        # Return the loss
        return loss

    # Pytorch lightning function to configure the optimizers of the model
    def configure_optimizers(self):
        # Define and return optimizer. Example: Adam
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    # PyTorch Lightning function (optional) called at the very end of each training epoch
    def on_train_epoch_end(self):
        # If the validation losses list is NOT empty, return (to avoid logging the training losses twice)
        if len(self.validation_losses) > 0:
            return
        epoch_num = self.current_epoch
        print()
        # Log the scheduled sampling probability for this epoch
        if self.transformer_type == DSITransformer.TRANSFORMER_TYPES.SCHEDULED_SAMPLING_TRANSFORMER:
            self.log('scheduled_sampling_probability',
                     self.scheduled_sampling_probability)
            print(f"Scheduled sampling probability for epoch {epoch_num}: ",
                  self.scheduled_sampling_probability)
            # Decrease the scheduled sampling probability
            self.scheduled_sampling_probability -= self.scheduled_sampling_decay
            if self.scheduled_sampling_probability < 0.0:
                self.scheduled_sampling_probability = 0.0
        # Log the average training loss for this epoch
        if not len(self.training_losses) == 0:
            avg_epoch_training_loss = torch.stack(self.training_losses).mean()
            self.log("avg_epoch_training_loss", avg_epoch_training_loss)
            print(f"Average training loss for epoch {epoch_num}: ",
                  avg_epoch_training_loss.item())
            self.training_losses.clear()
        # Log the average training accuracy for this epoch
        if not len(self.training_accuracies) == 0:
            avg_epoch_training_accuracy = torch.stack(
                self.training_accuracies).mean()
            self.log("avg_epoch_training_accuracy",
                     avg_epoch_training_accuracy)
            print(f"Average training accuracy for epoch {epoch_num}: ",
                  avg_epoch_training_accuracy.item())
            self.training_accuracies.clear()

    # Pytorch lightning function (optional) called at the very end of each validation epoch
    def on_validation_epoch_end(self):
        epoch_num = self.current_epoch
        print()
        # Log the scheduled sampling probability for this epoch
        if self.transformer_type == DSITransformer.TRANSFORMER_TYPES.SCHEDULED_SAMPLING_TRANSFORMER:
            self.log('scheduled_sampling_probability',
                     self.scheduled_sampling_probability)
            print(f"Scheduled sampling probability for epoch {epoch_num}: ",
                  self.scheduled_sampling_probability)
            # Decrease the scheduled sampling probability
            self.scheduled_sampling_probability -= self.scheduled_sampling_decay
            if self.scheduled_sampling_probability < 0.0:
                self.scheduled_sampling_probability = 0.0
        # Log the average training loss for this epoch
        if not len(self.training_losses) == 0:
            avg_epoch_training_loss = torch.stack(self.training_losses).mean()
            self.log("avg_epoch_training_loss", avg_epoch_training_loss)
            print(f"Average training loss for epoch {epoch_num}: ",
                  avg_epoch_training_loss.item())
            self.training_losses.clear()
        # Log the average validation loss for this epoch
        if not len(self.validation_losses) == 0:
            avg_epic_validation_loss = torch.stack(
                self.validation_losses).mean()
            self.log("avg_epoch_val_loss", avg_epic_validation_loss)
            print(f"Average validation loss for epoch {epoch_num}: ",
                  avg_epic_validation_loss.item())
            self.validation_losses.clear()
        # Log the average training accuracy for this epoch
        if not len(self.training_accuracies) == 0:
            avg_epoch_training_accuracy = torch.stack(
                self.training_accuracies).mean()
            self.log("avg_epoch_training_accuracy",
                     avg_epoch_training_accuracy)
            print(f"Average training accuracy for epoch {epoch_num}: ",
                  avg_epoch_training_accuracy.item())
            self.training_accuracies.clear()
        # Log the average validation accuracy for this epoch
        if not len(self.validation_accuracies) == 0:
            avg_epoch_validation_accuracy = torch.stack(
                self.validation_accuracies).mean()
            self.log("avg_epoch_val_accuracy", avg_epoch_validation_accuracy)
            print(f"Average validation accuracy for epoch {epoch_num}: ",
                  avg_epoch_validation_accuracy.item())
            self.validation_accuracies.clear()

    def reset_scheduled_sampling_probability(self):
        ''' Reset the scheduled sampling probability to 1.0 '''
        self.scheduled_sampling_probability = 1.0

    def generate_top_k_doc_ids(self, encoded_query: torch.Tensor, k: int, retrieval_dataset: datasets.TransformerRetrievalDataset):
        ''' Generate the top K document IDs for the given encoded query '''
        # Special tokens of the document IDs encoding
        doc_id_start_token = retrieval_dataset.doc_id_start_token
        doc_id_end_token = retrieval_dataset.doc_id_end_token
        doc_id_padding_token = retrieval_dataset.doc_id_padding_token
        # Max length of the document IDs
        doc_id_max_length = retrieval_dataset.doc_id_max_length
        # Repeat the query encoding for each of the k document IDs (to compute the k predictions in parallel)
        source_sequence = encoded_query.unsqueeze(1).repeat(1, k)
        # Initialize target sequence (document ID) as a tensor of the defined document ID size containing only the start token
        target_sequences = torch.full((1, k), doc_id_start_token,
                                      dtype=torch.long, device=encoded_query.device)
        # Initialize a tensor to store the top k sequences
        top_k_doc_ids_tokens = torch.zeros(doc_id_max_length, k,
                                           dtype=torch.long, device=encoded_query.device)
        # Iterate over the maximum length of the sequences (i.e. the number of tokens to generate for each document IDs)
        for i in range(doc_id_max_length):
            # Get the next tokens weights and sequences predictions from the transformer model
            outputs = self(source_sequence, target_sequences)
            # Get the next token to append to each sequence (i.e. the token with the highest probability for each of the k sequences)
            sorted_outputs, sorted_indices = torch.sort(
                outputs[-1], descending=True, dim=-1)
            # Get, for each output, the n-th highest probability token (where n depends on the sequence, so that it is 0 for the first of the k sequences, and self.target_tokens-1 for the last of the k sequences)
            min_prob_index = doc_id_max_length - 1 // 2
            indices = torch.linspace(0, min_prob_index, steps=k,
                                     device=encoded_query.device).long().unsqueeze(0)
            # Increment/Decrement some of the indices by 1 with a random probability
            indices += torch.randint_like(
                indices, 0, 2, device=encoded_query.device) * 2 - 1
            # Clamp the indices to be within the range [0, doc_id_max_length-1]
            indices = torch.clamp(indices, 0, doc_id_max_length - 1)
            # Get the tokens to append to the each of the k sequences
            tokens_to_append = sorted_indices.gather(1, indices).squeeze()
            # Append the selected tokens to the input sequence
            top_k_doc_ids_tokens[i] = tokens_to_append
            # Update the target sequence with the new tokens
            target_sequences = torch.cat(
                (target_sequences, tokens_to_append.unsqueeze(0)), dim=0)
        # Convert the top k sequences of document IDs' tokens to a list of k document IDs
        top_k_doc_ids = []
        for i in range(k):
            doc_id_tokens = top_k_doc_ids_tokens[:, i].tolist()
            doc_id = retrieval_dataset.decode_doc_id(doc_id_tokens)
            top_k_doc_ids.append(doc_id)
        # Remove duplicate document IDs
        top_k_doc_ids = list(set(top_k_doc_ids))
        # Refill the list in case of removed duplicates
        top_k_doc_ids = top_k_doc_ids + retrieval_dataset.get_similar_doc_ids(
            k - len(top_k_doc_ids), target_doc_ids=top_k_doc_ids)
        # Return the top k document IDs
        return top_k_doc_ids
