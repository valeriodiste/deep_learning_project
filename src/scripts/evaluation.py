# Import PyTorch and its modules
import torch
from torch.nn import functional
# Import other modules
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Import custom modules
try:
    from src.scripts import datasets
    from src.scripts.utils import MODEL_TYPES, RANDOM_SEED
    from tqdm import tqdm
except ModuleNotFoundError:
    from deep_learning_project.src.scripts import datasets
    from deep_learning_project.src.scripts.utils import MODEL_TYPES, RANDOM_SEED
    from tqdm.notebook import tqdm

# Seed random number generators for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def compute_mean_average_precision_at_k(model_type, queries_dict, docs_dict, k_documents=10, n_queries=10, print_debug=False, **kwargs):
    '''
    Computes the precision at k for the given model.

    Args:
    - model_type (MODEL_TYPES): The model to evaluate.
    - queries_dict (dict): The queries dictionary (should only contain examples that were not in the training set)
    - docs_dict (dict): The documents dictionary (should only contain examples that were not in the training set)
    - k_documents (int): The number of documents to retrieve.
    - n_queries (int): The number of queries to evaluate.
    - **kwargs: Additional arguments for the model.

    Returns:
    - dict: The evaluation results, with structure:
        {
            "mean_average_precision": float,
            "evaluated_queries": {
                query_id: precision_at_k,
                ...
            },
            "model": MODEL_TYPES,
            "k_documents": int,
            "n_queries": int,
            "infos": dict (optional)
        }

    Keyword Args:
    - tfidf_matrix (csr_matrix): The TF-IDF matrix (use for TF-IDF model)
    - vectorizer (TfidfVectorizer): The TF-IDF vectorizer (use for TF-IDF model)
    - model (SiameseNetwork | DSI_Transformer): The model to evaluate (use for Siamese Network and DSI Transformer models)
    - retrieval_dataset (datasets.TransformerRetrievalDataset): The retrieval dataset (use for DSI Transformer model)
    - retrieval_test_set (dict): The test set for the retrieval dataset (use for DSI Transformer model)

    '''

    # Results dictionary
    results = {
        "mean_average_precision": 0,
        "evaluated_queries": {},
        "model": model_type,
        "k_documents": k_documents,
        "n_queries": n_queries,
        "infos": {}
    }

    # Get all the document ids
    encoded_doc_ids = list(docs_dict.keys())

    # Get n random queries
    query_ids = random.sample(list(queries_dict.keys()), n_queries)

    # Evaluate the specified model
    print(f"Evaluating {model_type} model to compute MAP@K...")
    if model_type == MODEL_TYPES.TF_IDF:

        # Get the TF-IDF matrix and vectorizer from the kwargs
        tfidf_matrix = kwargs["tfidf_matrix"]
        vectorizer = kwargs["vectorizer"]

        # Iterate over the queries
        for query_index, encoded_query in enumerate(query_ids):

            # Get the relevant documents for the query
            relevant_docs = queries_dict[encoded_query]["relevant_docs"]

            # Compute the TF-IDF matrix
            query_tfidf = vectorizer.transform(
                [queries_dict[encoded_query]['text'].lower()])

            # Compute cosine similarities
            similarities = cosine_similarity(
                query_tfidf, tfidf_matrix).flatten()

            # Get the top K retrieved documents
            top_k_documents = [encoded_doc_ids[i] for i in np.argsort(
                similarities, axis=0)[::-1]][:k_documents]

            # Count how many of the top K retrieved documents are also in the relevant documents
            relevant_count = sum(
                [1 for doc_id in top_k_documents if doc_id in relevant_docs])

            # Compute the precision at k for the query
            results["evaluated_queries"][encoded_query] = \
                relevant_count / min(k_documents, len(relevant_docs))

    elif model_type == MODEL_TYPES.WORD2VEC:

        # Iterate over the queries
        for query_index, encoded_query in enumerate(query_ids):

            # Get the relevant documents for the query
            relevant_docs = queries_dict[encoded_query]["relevant_docs"]

            # Compute the relevance scores for the documents using the embeddings of the queries and documents
            docs_relevance = []
            for doc_id in tqdm(encoded_doc_ids, f"Computing relevance scores for MAP@K for query {query_index+1}/{n_queries}"):
                # Compute the cosine similarity between the query and document embeddings
                relevance_score = functional.cosine_similarity(
                    torch.tensor(queries_dict[encoded_query]
                                 ["embedding"], dtype=torch.float32),
                    torch.tensor(docs_dict[doc_id]
                                 ["embedding"], dtype=torch.float32),
                    dim=0
                ).item()
                # Append the relevance score for this document to the list of relevance scores for this query
                docs_relevance.append((doc_id, relevance_score))

            # Check if the relevance scores are all equal (i.e., the model is not able to differentiate between the documents)
            if print_debug:
                if len(set([relevance_score for _, relevance_score in docs_relevance])) < 5:
                    print(
                        f"  Warning: All the computed relevance scores are too similar for query {encoded_query}, model may not be working properly...")

            # Get the top K retrieved documents
            top_k_documents = [doc_id for doc_id, _ in sorted(
                docs_relevance, key=lambda x: x[1], reverse=True)][:k_documents]

            # Count how many of the top K retrieved documents are also in the relevant documents
            relevant_count = sum(
                [1 for doc_id in top_k_documents if doc_id in relevant_docs])

            # Compute the precision at k for the query
            results["evaluated_queries"][encoded_query] = relevant_count / \
                min(k_documents, len(relevant_docs))

    elif model_type == MODEL_TYPES.SIAMESE_NETWORK:

        # Get the Siamese Network model from the kwargs, used to predict the relevant documents
        siamese_network_model = kwargs["model"]

        # Set the model to evaluation mode
        siamese_network_model.eval()

        # Iterate over the queries
        for query_index, encoded_query in enumerate(query_ids):

            # Get the relevant documents for the query
            relevant_docs = queries_dict[encoded_query]["relevant_docs"]

            # Get the query embedding
            query_embedding = torch.tensor(
                queries_dict[encoded_query]["embedding"], dtype=torch.float32)

            # Compute the relevance scores for the documents using the Siamese Network model
            docs_relevance = []
            for doc_id in tqdm(encoded_doc_ids, f"Computing relevance scores for MAP@K for query {query_index+1}/{n_queries}"):
                # Get the document embedding
                doc_embedding = torch.tensor(docs_dict[doc_id]["embedding"],
                                             dtype=torch.float32)
                # Compute the cosine similarity between the query and document embeddings
                query_prediction = siamese_network_model(query_embedding)
                doc_prediction = siamese_network_model(doc_embedding)
                relevance_score = functional.cosine_similarity(
                    query_prediction.unsqueeze(0),
                    doc_prediction.unsqueeze(0)
                ).item()
                # Append the relevance score for this document to the list of relevance scores for this query
                docs_relevance.append((doc_id, relevance_score))

            # Check if the relevance scores are all equal (i.e., the model is not able to differentiate between the documents)
            if print_debug:
                if len(set([relevance_score for _, relevance_score in docs_relevance])) < 5:
                    print(
                        f"  Warning: All the computed relevance scores are too similar for query {encoded_query}, model may not be working properly...")

            # Get the top K retrieved documents
            top_k_documents = [doc_id for doc_id, _ in sorted(
                docs_relevance, key=lambda x: x[1], reverse=True)][:k_documents]

            # Count how many of the top K retrieved documents are also in the relevant documents
            relevant_count = sum(
                [1 for doc_id in top_k_documents if doc_id in relevant_docs])

            # Compute the precision at k for the query
            results["evaluated_queries"][encoded_query] = \
                relevant_count / min(k_documents, len(relevant_docs))

            if print_debug:
                print(
                    f"  Precision at {k_documents} for query {query_index+1}/{n_queries}: {results['evaluated_queries'][encoded_query]}")

    elif model_type == MODEL_TYPES.DSI_TRANSFORMER:

        # Get the DSI Transformer model and test dataset from the kwargs
        transformer_model = kwargs["model"]

        # Set the model to evaluation mode
        transformer_model.eval()

        # Get the full transformer retrieval dataset
        retrieval_dataset: datasets.TransformerRetrievalDataset = kwargs["retrieval_dataset"]

        # Get the encoded queries and encoded ids in the test dataset (a dict with keys "encoded_queries" and "doc_ids" containing, each, a list of lists representing the encoded queries and the encoded relevant document ids for each encoded query)
        retrieval_test_set = kwargs["retrieval_test_set"]

        # Get the actual random queries from the test set
        encoded_queries = retrieval_test_set["encoded_queries"]
        encoded_doc_ids = retrieval_test_set["encoded_doc_ids"]

        # Get unique queries
        unique_queries = []
        for encoded_query in encoded_queries:
            if encoded_query not in unique_queries:
                unique_queries.append(encoded_query)

        # Pick n random queries
        random_queries = random.sample(unique_queries, n_queries)

        # Set the "infos" return key to contain the transformer type
        results["infos"]["type"] = transformer_model.transformer_type

        # Iterate over the queries
        for query_index, encoded_query in enumerate(random_queries):

            # Get the actual query id
            query_id = retrieval_dataset.get_query_id(encoded_query)

            # Get the relevant documents for the query
            relevant_docs = queries_dict[query_id]["relevant_docs"]

            # Get the model's top k document IDs for the query
            top_k_doc_ids = transformer_model.generate_top_k_doc_ids(
                torch.tensor(encoded_query, dtype=torch.long),
                k_documents,
                retrieval_dataset
            )

            # Print the top k predicted document IDs for the query
            if print_debug:
                print(
                    f"Top {k_documents} (predicted) document IDs for query {query_index+1}/{n_queries}:")
                print(f"  {top_k_doc_ids}")
                print(
                    f"> Actual relevant document IDs for the query:")
                print(f"  {relevant_docs}")

            # Count how many of the top K retrieved documents are also in the relevant documents
            relevant_count = sum(
                [1 for doc_id in top_k_doc_ids if doc_id in relevant_docs])

            # Compute the precision at k for the query
            results["evaluated_queries"][query_id] = relevant_count / \
                min(k_documents, len(relevant_docs))

            if print_debug:
                print(
                    f"  Precision at {k_documents} for query {query_index+1}/{n_queries}: {results['evaluated_queries'][query_id]}")

    else:
        raise ValueError("Invalid model")

    # Compute the mean average precision
    if results["evaluated_queries"] != {}:
        results["mean_average_precision"] = np.mean(
            list(results["evaluated_queries"].values()))

    return results


def compute_recall_at_k(model_type, queries_dict, docs_dict, query_id=None, k_documents=1000, print_debug=False, **kwargs):
    '''
    Computes the recall at k for the given model.

    Args:
    - model_type (MODEL_TYPES): The model to evaluate.
    - queries_dict (dict): The queries dictionary (should only contain examples that were not in the training set)
    - docs_dict (dict): The documents dictionary (should only contain examples that were not in the training set)
    - query_id (str): The query ID of the query to evaluate (if None, a random query will be selected).
    - k_documents (int): The number of documents to retrieve.
    - **kwargs: Additional arguments for the model.

    Returns:
    - dict: The evaluation results, with structure:
        {
            "recall_at_k": float,
            "k_documents": int,
            "query_id": str,
            "model": MODEL_TYPES,
            "infos": dict (optional)
        }
    '''

    # If the query ID is not specified, select a random query (if the model is a DSI Transformer model, the query ID is aways randomly selected from the test set)
    if query_id is None and model_type != MODEL_TYPES.DSI_TRANSFORMER:
        query_id = random.choice(list(queries_dict.keys()))

    # Results dictionary
    results = {
        "recall_at_k": 0,
        "k_documents": k_documents,
        "query_id": query_id,
        "model": model_type,
        "infos": {}
    }

    # Get the relevant documents for the query
    relevant_docs = queries_dict[query_id]["relevant_docs"] if query_id is not None else None

    # Get all the document ids
    encoded_doc_ids = list(docs_dict.keys())

    # Evaluate the specified model
    if print_debug:
        print(f"Evaluating {model_type} model to compute Recall@K...")
    if model_type == MODEL_TYPES.TF_IDF:

        # Get the TF-IDF matrix and vectorizer from the kwargs
        tfidf_matrix = kwargs["tfidf_matrix"]
        vectorizer = kwargs["vectorizer"]

        # Compute the TF-IDF matrix
        query_tfidf = vectorizer.transform(
            [queries_dict[query_id]['text'].lower()])

        # Compute cosine similarities
        similarities = cosine_similarity(
            query_tfidf, tfidf_matrix).flatten()

        # Get the top K retrieved documents
        top_k_documents = [encoded_doc_ids[i] for i in np.argsort(
            similarities, axis=0)[::-1]][:k_documents]

        # Count how many of the top K retrieved documents are also in the relevant documents
        relevant_count = sum(
            [1 for doc_id in top_k_documents if doc_id in relevant_docs])

        # Compute the recall at k for the query
        results["recall_at_k"] = relevant_count / len(relevant_docs)

    elif model_type == MODEL_TYPES.WORD2VEC:

        # Compute the relevance scores for the documents using the embeddings of the queries and documents
        docs_relevance = []
        for doc_id in tqdm(encoded_doc_ids, "Computing relevance scores for Recall@K..."):
            # Compute the cosine similarity between the query and document embeddings
            relevance_score = functional.cosine_similarity(
                torch.tensor(queries_dict[query_id]
                             ["embedding"], dtype=torch.float32),
                torch.tensor(docs_dict[doc_id]
                             ["embedding"], dtype=torch.float32),
                dim=0
            ).item()
            # Append the relevance score for this document to the list of relevance scores for this query
            docs_relevance.append((doc_id, relevance_score))

        # Get the top K retrieved documents
        top_k_documents = [doc_id for doc_id, _ in sorted(
            docs_relevance, key=lambda x: x[1], reverse=True)][:k_documents]

        # Count how many of the top K retrieved documents are also in the relevant documents
        relevant_count = sum(
            [1 for doc_id in top_k_documents if doc_id in relevant_docs])

        # Compute the recall at k for the query
        results["recall_at_k"] = relevant_count / len(relevant_docs)

    elif model_type == MODEL_TYPES.SIAMESE_NETWORK:

        # Get the Siamese Network model from the kwargs, used to predict the relevant documents
        siamese_network_model = kwargs["model"]

        # Set the model to evaluation mode
        siamese_network_model.eval()

        # Get the query embedding
        query_embedding = torch.tensor(
            queries_dict[query_id]["embedding"], dtype=torch.float32)

        # Compute the relevance scores for the documents using the Siamese Network model
        docs_relevance = []
        for doc_id in tqdm(encoded_doc_ids, "Computing relevance scores for Recall@K..."):
            # Get the document embedding
            doc_embedding = torch.tensor(docs_dict[doc_id]["embedding"],
                                         dtype=torch.float32)
            # Compute the cosine similarity between the query and document embeddings
            query_prediction = siamese_network_model(query_embedding)
            doc_prediction = siamese_network_model(doc_embedding)
            relevance_score = functional.cosine_similarity(
                query_prediction.unsqueeze(0),
                doc_prediction.unsqueeze(0)
            ).item()
            # Append the relevance score for this document to the list of relevance scores for this query
            docs_relevance.append((doc_id, relevance_score))

        # Get the top K retrieved documents
        top_k_documents = [doc_id for doc_id, _ in sorted(
            docs_relevance, key=lambda x: x[1], reverse=True)][:k_documents]

        # Count how many of the top K retrieved documents are also in the relevant documents
        relevant_count = sum(
            [1 for doc_id in top_k_documents if doc_id in relevant_docs])

        # Compute the recall at k for the query
        results["recall_at_k"] = relevant_count / len(relevant_docs)

    elif model_type == MODEL_TYPES.DSI_TRANSFORMER:

        # Get the DSI Transformer model and test dataset from the kwargs
        transformer_model = kwargs["model"]

        # Set the model to evaluation mode
        transformer_model.eval()

        # Get the full transformer retrieval dataset
        retrieval_dataset: datasets.TransformerRetrievalDataset = kwargs["retrieval_dataset"]

        # Get the encoded queries and encoded ids in the test dataset (a dict with keys "encoded_queries" and "doc_ids" containing, each, a list of lists representing the encoded queries and the encoded relevant document ids for each encoded query)
        retrieval_test_set = kwargs["retrieval_test_set"]

        # Get the actual random queries from the test set
        encoded_queries = retrieval_test_set["encoded_queries"]
        encoded_doc_ids = retrieval_test_set["encoded_doc_ids"]

        # Get unique queries
        unique_queries = []
        for encoded_query in encoded_queries:
            if encoded_query not in unique_queries:
                unique_queries.append(encoded_query)

        # Pick a random query
        encoded_query = random.choice(unique_queries)

        # Get the actual query id
        query_id = retrieval_dataset.get_query_id(encoded_query)

        # Update the results dictionary
        results["query_id"] = query_id

        # Set the "infos" return key to contain the transformer type
        results["infos"]["type"] = transformer_model.transformer_type

        # Get the relevant documents for the query
        relevant_docs = queries_dict[query_id]["relevant_docs"]

        # Get the model's top k document IDs for the query
        top_k_doc_ids = transformer_model.generate_top_k_doc_ids(
            torch.tensor(encoded_query, dtype=torch.long),
            k_documents,
            retrieval_dataset
        )

        # Print the top k predicted document IDs for the query
        if print_debug:
            print(
                f"Top {k_documents} (predicted) document IDs for query {query_id}:")
            print(f"  {top_k_doc_ids}")
            print(f"> Actual relevant document IDs for the query:")
            print(f"  {relevant_docs}")

        # Count how many of the top K retrieved documents are also in the relevant documents
        relevant_count = sum(
            [1 for doc_id in top_k_doc_ids if doc_id in relevant_docs])

        # Compute the recall at k for the query
        results["recall_at_k"] = relevant_count / len(relevant_docs)

    else:
        raise ValueError("Invalid model")

    return results
