# Import PyTorch and its modules
import torch
from torch.utils.data import Dataset
# Import Hugging Face's Transformers library and its modules
from transformers import AutoTokenizer
# Import other libraries
import random
import json
import os
import sys
from torch.nn import functional
# Import custom modules
try:
    from src.scripts.utils import MODEL_TYPES, RANDOM_SEED, get_preprocessed_text
    from tqdm import tqdm
except ModuleNotFoundError:
    from deep_learning_project.src.scripts.utils import MODEL_TYPES, RANDOM_SEED, get_preprocessed_text
    from tqdm.notebook import tqdm

# Seed random number generators for reproducibility
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Constants for the special document IDs tokens of the Transformer models
DOC_ID_START_TOKEN = 12
DOC_ID_END_TOKEN = 10
DOC_ID_PADDING_TOKEN = 11


class SiameseNetworkDataset(Dataset):

    # Initialize the dataset
    def __init__(self, queries_dict: dict, documents_dict: dict, dataset_file_path: str = None, force_dataset_rebuild: bool = False):
        '''
        Constructor of the SiameseNetworkDataset class.

        Args:
        - queries_dict: dict, a dictionary containing the queries data
        - documents_dict: dict, a dictionary containing the documents data
        - save_database_file_path: str, the path of the JSON file in which the triplets data will be saved or from which it will be loaded
        '''

        # Store the queries and documents dictionaries
        self.queries = queries_dict
        self.documents = documents_dict

        def get_triplets():
            ''' Function to build or retrieve the dataset of <query (anchor), document (positive), document (negative)> triplets '''
            # Check if the triplets data is already saved in the file, if it exists, load it, otherwise, build it
            if not force_dataset_rebuild and dataset_file_path is not None and os.path.exists(dataset_file_path):
                print(
                    f"Loading the Siamese Network's triplets data from {dataset_file_path}...")
                triplets = []
                with open(dataset_file_path, 'r') as f:
                    triplets = json.load(f)
                print(
                    f"Loaded {len(triplets)} triplets from {dataset_file_path}")
                return triplets
            else:
                # Initialize the list of triplets
                triplets = []
                # Set of document IDs
                document_ids = set(documents_dict.keys())
                # Iterate over each query to build the triplets
                for query_id in tqdm(queries_dict.keys(), desc="Building SiameseNetworkDataset"):
                    # Get the list of related documents
                    positive_document_ids = set(
                        queries_dict[query_id]['relevant_docs'])
                    # Get the list of unrelated documents
                    unrelated_document_ids = document_ids - positive_document_ids
                    negative_document_ids = random.sample(
                        unrelated_document_ids, len(positive_document_ids))
                    # Add the triplets to the list
                    for positive_document, negative_document in zip(positive_document_ids, negative_document_ids):
                        triplets.append(
                            [query_id, positive_document, negative_document])
                # Save the triplets into the "triplets.json" file if a save file path is provided
                if dataset_file_path is not None:
                    print(
                        f"Saving the Siamese Network's triplets data to {dataset_file_path}...")
                    with open(dataset_file_path, 'w') as f:
                        json.dump(triplets, f)
                return triplets

        # Build or retrieve the dataset of <query (anchor), document (positive), document (negative)> triplets
        self.triplets = get_triplets()

    # Torch dataset function to get the length of the dataset
    def __len__(self):
        return len(self.triplets)

    # Torch dataset function to get a specific item from the dataset given its index
    def __getitem__(self, idx):

        # Get the anchor, positive, and negative document IDs
        anchor_query_id, positive_doc_id, negative_doc_id = self.triplets[idx]

        # Return the <query (anchor), document (positive), document (negative)> triplet as PyTorch tensors representing the query/document embeddings
        return torch.tensor(
            self.queries[anchor_query_id]["embedding"], dtype=torch.float32
        ), torch.tensor(
            self.documents[positive_doc_id]["embedding"], dtype=torch.float32
        ), torch.tensor(
            self.documents[negative_doc_id]["embedding"], dtype=torch.float32
        )


class TransformerIndexingDataset(Dataset):

    # Initialize the dataset of tuples (encoded_doc, encoded_doc_id) for the indexing phase (Transformer models learns to map documents to queries)
    def __init__(
        self,
        documents: dict,
        doc_id_max_length: int = -1,
        doc_max_length: int = 64,
        dataset_file_path: str = None,
        force_dataset_rebuild: bool = False
    ):
        '''
        Constructor of the TransformerIndexingDataset class.

        Args:
        - documents: dict, a dictionary containing the documents data
        - doc_id_max_length: int, the maximum length of the doc IDs sequence
        - doc_max_length: int, the maximum length of the input sequence
        - dataset_file_path: str, the path of the JSON file in which the <document, doc_id> pairs data will be saved or from which it will be loaded
        - force_dataset_rebuild: bool, a flag to force the rebuilding of the dataset (if false, a dataset file path is provided, and the file exists, the dataset will be loaded from the file)
        '''
        # Store the documents dictionary
        self.documents = documents
        # Store the maximum document length
        self.doc_max_length = doc_max_length
        # Store the dataset file path
        self.save_dataset_file_path = dataset_file_path
        # We use a bert tokenizer to encode the documents
        tokenizer_model = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            use_fast=True
        )
        # Define the doc IDs special tokens
        self.doc_id_start_token = DOC_ID_START_TOKEN
        self.doc_id_end_token = DOC_ID_END_TOKEN
        self.doc_id_padding_token = DOC_ID_PADDING_TOKEN
        # Set the maximum doc ID length
        if doc_id_max_length < 0:
            # Compute the maximum doc ID length (and add 1 for the start token and 1 for the end token)
            self.doc_id_max_len = max(len(str(doc_id))
                                      for doc_id in documents.keys()) + 2
        else:
            # Assign the provided doc IDs max length
            self.doc_id_max_len = doc_id_max_length + 2
        # Initialize the encoded documents and encoded doc IDs lists
        self.encoded_docs, self.encoded_doc_ids = self.get_dataset(
            force_dataset_rebuild)

    def get_dataset(self, force_dataset_rebuild=False):
        ''' Function to build or retrieve the dataset of <encoded_doc, encoded_doc_id> tuples '''
        if not force_dataset_rebuild and self.save_dataset_file_path is not None and os.path.exists(self.save_dataset_file_path):
            print(
                f"Loading the Transformer Indexing Dataset from {self.save_dataset_file_path}...")
            with open(self.save_dataset_file_path, 'r') as f:
                dataset = json.load(f)
            print(
                f"Loaded {len(dataset['encoded_docs'])} documents from {self.save_dataset_file_path}")
            encoded_docs = [torch.tensor(doc)
                            for doc in dataset['encoded_docs']]
            doc_ids = [torch.tensor(doc_id)
                       for doc_id in dataset['encoded_doc_ids']]
            return encoded_docs, doc_ids
        else:
            # Initialize the encoded documents and doc IDs lists
            doc_ids = []
            encoded_docs = []
            # For each document in the documents dictionary
            for doc_id in tqdm(self.documents.keys(), desc='Building TransformerIndexingDataset'):
                document = self.documents[doc_id]
                # Tokenize and then encode the document text
                preprocessed_text = get_preprocessed_text(document['text'])
                encoded_doc = self.tokenizer(preprocessed_text,
                                             add_special_tokens=True,
                                             max_length=self.doc_max_length,
                                             truncation=True,
                                             return_tensors='pt'
                                             )['input_ids'][0].tolist()
                # Pad the document sequence to the max encoded document length
                document_padding_length = self.doc_max_length - \
                    len(encoded_doc)
                encoded_doc = functional.pad(
                    torch.tensor(encoded_doc),
                    (0, document_padding_length),
                    value=0
                )
                # Encode the doc ID
                doc_id_padding_length = self.doc_id_max_len - len(doc_id)
                encoded_doc_id = torch.tensor(
                    # Start of sequence token
                    [self.doc_id_start_token] +
                    # Encoded document ID
                    list(map(int, doc_id)) +
                    # End of sequence token
                    [self.doc_id_end_token] +
                    # Padding tokens (if needed)
                    [self.doc_id_padding_token] * doc_id_padding_length
                )
                # Add the encoded document and doc ID to the lists
                doc_ids.append(encoded_doc_id)
                encoded_docs.append(encoded_doc)
            # Save the dataset to the file if a save file path is provided
            if self.save_dataset_file_path is not None:
                print(
                    f"Saving the Transformer Indexing Dataset to {self.save_dataset_file_path}...")
                with open(self.save_dataset_file_path, 'w') as f:
                    json.dump({
                        'encoded_docs': [doc.tolist() for doc in encoded_docs],
                        'encoded_doc_ids': [doc_id.tolist() for doc_id in doc_ids]
                    }, f)
            # Return the encoded documents and doc IDs
            return encoded_docs, doc_ids

    def __len__(self):
        return len(self.encoded_docs)

    def __getitem__(self, idx):
        return self.encoded_docs[idx], self.encoded_doc_ids[idx]


class TransformerRetrievalDataset(Dataset):

    # Initialize the dataset of tuples (encoded_query, encoded_doc_id) for the retrieval phase (Transformer models learns to map queries to documents)
    def __init__(
        self,
        documents: dict,
        queries: dict,
        doc_id_max_length: int = -1,
        query_max_length: int = 16,
        dataset_file_path: str = None,
        force_dataset_rebuild: bool = False
    ):
        '''
        Constructor of the TransformerRetrievalDataset class.

        Args:
        - documents: dict, a dictionary containing the documents data
        - queries: dict, a dictionary containing the queries data
        - doc_id_max_length: int, the maximum length of the doc IDs sequence
        - query_max_length: int, the maximum length of the input sequence
        - dataset_file_path: str, the path of the JSON file in which the <document, doc_id> pairs data will be saved or from which it will be loaded
        - force_dataset_rebuild: bool, a flag to force the rebuilding of the dataset (if false, a dataset file path is provided, and the file exists, the dataset will be loaded from the file)
        '''
        # Store the documents and queries dictionaries
        self.documents = documents
        self.queries = queries
        # Store the maximum query length
        self.query_max_length = query_max_length
        # Store the dataset file path
        self.save_dataset_file_path = dataset_file_path
        # Initialize tokenized query to query dictionary
        self.query_ids = dict()
        # We use a bert tokenizer to encode the documents
        tokenizer_model = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            use_fast=True
        )
        # Define the doc IDs special tokens
        self.doc_id_start_token = DOC_ID_START_TOKEN
        self.doc_id_end_token = DOC_ID_END_TOKEN
        self.doc_id_padding_token = DOC_ID_PADDING_TOKEN
        # Set the maximum doc ID length
        if doc_id_max_length < 0:
            # Compute the maximum doc ID length (and add 1 for the start token and 1 for the end token)
            self.doc_id_max_length = max(len(str(doc_id))
                                         for doc_id in documents.keys()) + 2
        else:
            # Assign the provided doc IDs max length
            self.doc_id_max_length = doc_id_max_length + 2
        # Initialize the encoded documents and encoded doc IDs lists
        self.encoded_queries, self.encoded_doc_ids = self.get_dataset(
            force_dataset_rebuild)

    def get_dataset(self, force_dataset_rebuild=False):
        ''' Function to build or retrieve the dataset of <encoded_query, encoded_doc_id> tuples '''
        if not force_dataset_rebuild and self.save_dataset_file_path is not None and os.path.exists(self.save_dataset_file_path):
            print(
                f"Loading the Transformer Retrieval Dataset from {self.save_dataset_file_path}...")
            with open(self.save_dataset_file_path, 'r') as f:
                dataset = json.load(f)
            print(
                f"Loaded {len(dataset['encoded_queries'])} encoded queries and document IDs from {self.save_dataset_file_path}")
            encoded_queries = [torch.tensor(query)
                               for query in dataset['encoded_queries']]
            doc_ids = [torch.tensor(doc_id)
                       for doc_id in dataset['encoded_doc_ids']]
            # Rebuild the query_ids dictionary
            query_ids_mapping = dataset['query_ids_mapping']
            for query, query_id in zip(dataset['encoded_queries'], query_ids_mapping):
                self.query_ids[str(query)] = query_id
            return encoded_queries, doc_ids
        else:
            # Initialize the encoded queries and doc IDs lists
            doc_ids = []
            encoded_queries = []
            # Store the query ids for reoading the query_ids dictionary
            query_ids_mapping = []
            # Iterate over the queries
            for query_id in tqdm(self.queries.keys(), desc='Building TransformerRetrievalDataset'):
                query = self.queries[query_id]
                # Tokenize and then encode the query text
                preprocessed_text = get_preprocessed_text(query['text'])
                encoded_query = self.tokenizer(preprocessed_text,
                                               add_special_tokens=True,
                                               max_length=self.query_max_length,
                                               truncation=True,
                                               return_tensors='pt'
                                               )['input_ids'][0].tolist()
                # Pad the query sequence to the max encoded queries length
                query_padding_length = self.query_max_length - \
                    len(encoded_query)
                encoded_query = functional.pad(
                    torch.tensor(encoded_query),
                    (0, query_padding_length),
                    value=0
                )
                # Add the tokenized query to query dictionary
                self.query_ids[str(encoded_query.tolist())] = query_id
                # For each document ID in the relevant document IDs list of the query
                for doc_id in query['relevant_docs']:
                    # Encode the doc ID
                    doc_id_padding_length = self.doc_id_max_length - \
                        len(doc_id)
                    encoded_doc_id = torch.tensor(
                        # Start of sequence token
                        [self.doc_id_start_token] +
                        # Encoded document ID
                        list(map(int, doc_id)) +
                        # End of sequence token
                        [self.doc_id_end_token] +
                        # Padding tokens (if needed)
                        [self.doc_id_padding_token] * doc_id_padding_length
                    )
                    # Add the encoded document and doc ID to the lists
                    encoded_queries.append(encoded_query)
                    doc_ids.append(encoded_doc_id)
                    query_ids_mapping.append(query_id)
            # Save the dataset to the file if a save file path is provided
            if self.save_dataset_file_path is not None:
                print(
                    f"Saving the Transformer Retrieval Dataset to {self.save_dataset_file_path}...")
                with open(self.save_dataset_file_path, 'w') as f:
                    json.dump({
                        'encoded_queries': [query.tolist() for query in encoded_queries],
                        'encoded_doc_ids': [doc_id.tolist() for doc_id in doc_ids],
                        'query_ids_mapping': query_ids_mapping
                    }, f)
            # Return the encoded queries and doc IDs
            return encoded_queries, doc_ids

    def __len__(self):
        return len(self.encoded_queries)

    def __getitem__(self, idx):
        return self.encoded_queries[idx], self.encoded_doc_ids[idx]

    def get_query_id(self, encoded_query):
        ''' Get the query ID from the encoded query (given either as a tensor or a list of integers) '''
        if isinstance(encoded_query, torch.Tensor):
            # Convert the tensor to a list
            encoded_query = encoded_query.tolist()
        # Return the query ID from the query_ids dictionary (converting all the keys of the dictionary to lists)
        return self.query_ids[str(encoded_query)]

    def decode_doc_id(self, encoded_doc_id, force_debug_output=False, recover_malformed_doc_ids=True):
        ''' 
        Decode the given encoded document ID into to a string 

        If the document ID is malformed, the output document ID will be prefixed with "M=" (for malformed) and its special tokens will be converted to letters.

        If the force_debug_output flag is set to True (and the doc id is not malformed), the output document ID will be prefixed with "D=" (for debug) and its special tokens will be converted to letters.

        Args:
        - encoded_doc_id: list or tensor, the encoded document ID (list of integers from 0 to 9 or special token integers)
        - use_debug_output: bool, wheter to return document IDs as a debug string (converting special tokens to letters) or as valid document IDs (string with the ID's digits)
        '''
        # Convert the given encoded doc id to a list if its a tensor
        if isinstance(encoded_doc_id, torch.Tensor):
            encoded_doc_id = encoded_doc_id.tolist()
        # Check if the given encoded doc id is malformed
        malformed_doc_id = \
            self.doc_id_end_token not in encoded_doc_id or \
            encoded_doc_id[0] == self.doc_id_end_token or \
            (encoded_doc_id[0] == self.doc_id_start_token
             and encoded_doc_id[1] == self.doc_id_end_token)
        # Convert the encoded doc id to a list of integers or special tokens mappings
        if not force_debug_output and not malformed_doc_id:
            # Remove the start token if it's the first character
            if encoded_doc_id[0] == self.doc_id_start_token:
                encoded_doc_id = encoded_doc_id[1:]
            # Keep only the characters before the first end token (if it exists)
            first_end_token_index = encoded_doc_id.index(
                self.doc_id_end_token)
            encoded_doc_id = encoded_doc_id[:first_end_token_index]
        else:
            # Map each special token to a letter
            special_tokens_mappings = {
                self.doc_id_start_token: 'S',
                self.doc_id_end_token: 'E',
                self.doc_id_padding_token: 'P'
            }
            doc_id_start = ""
            if malformed_doc_id and not recover_malformed_doc_ids:
                doc_id_start = "M="
            elif force_debug_output:
                doc_id_start = "D="
            converted_encoded_doc_id = [doc_id_start]
            for token in encoded_doc_id:
                if int(token) in special_tokens_mappings.keys():
                    if malformed_doc_id and recover_malformed_doc_ids:
                        # Skip the special tokens if the doc id is malformed and we want to recover it
                        continue
                    else:
                        converted_encoded_doc_id.append(
                            special_tokens_mappings[token])
                else:
                    converted_encoded_doc_id.append(str(token))
            encoded_doc_id = converted_encoded_doc_id
        # Convert the remaining tokens to string and join them
        decoded_doc_id = "".join(
            [str(token) for token in encoded_doc_id])
        # Recover the malformed doc id if needed
        if malformed_doc_id and recover_malformed_doc_ids:
            # Check if the final decoded doc id is valid
            if decoded_doc_id not in self.documents.keys():
                max_int_value = sys.maxsize
                min_int_value = -sys.maxsize - 1
                closest_doc_id = min(self.documents.keys(), key=lambda doc_id: abs(
                    int(doc_id if str.isdigit(doc_id) else max_int_value) -
                    int(decoded_doc_id if str.isdigit(decoded_doc_id) else min_int_value)))
                decoded_doc_id = str(closest_doc_id)
        # Return the decoded document ID
        return decoded_doc_id
