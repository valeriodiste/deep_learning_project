import json
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Random seed for reproducibility
RANDOM_SEED = 14


class MODEL_TYPES:
    ''' Enum class with the model types '''
    TF_IDF: str = "TF-IDF"
    WORD2VEC: str = "Word2Vec"
    SIAMESE_NETWORK: str = "Siamese Network with Triplet Loss"
    DSI_TRANSFORMER: str = "DSI Transformer"


# Dictionary with the model checkpoints files for each model type
MODEL_CHECKPOINTS_FILES = {
    MODEL_TYPES.TF_IDF: None,
    MODEL_TYPES.WORD2VEC: "word2vec.model",
    MODEL_TYPES.SIAMESE_NETWORK: "siamese_network.ckpt",
    MODEL_TYPES.DSI_TRANSFORMER: "transformer.ckpt"
}


def print_json(json_obj, indent_all_by_tabs=0, compact=True):
    ''' Print the given JSON object in a readable format '''
    json_lines = json.dumps(json_obj, indent=2).split("\n")
    if compact:
        # Remove first and last lines if they contain only brackets
        if json_lines[0].strip() == "{":
            json_lines = json_lines[1:]
        if json_lines[-1].strip() == "}":
            json_lines = json_lines[:-1]
    for line in json_lines:
        print(" " * indent_all_by_tabs + line)


def get_preprocessed_text(text, print_debug=False):
    ''' Preprocess the text by removing stopwords and punctuation '''
    if print_debug:
        print("text:", text)
    # Define the stopwords to remove
    words_to_remove = set(string.punctuation).union(
        set(stopwords.words('english')))
    if print_debug:
        print("words_to_remove:", words_to_remove)
    # Tokenize the text
    tokenized_text = word_tokenize(text)
    if print_debug:
        print("tokenized_text:", tokenized_text)
    # Remove english stopwords and punctuation and convert tokens to lowercase
    tokens = [token.lower() for token in tokenized_text
              if token.lower() not in words_to_remove]
    if print_debug:
        print("tokens:", tokens)
    # Return the preprocessed text
    return " ".join(tokens)


def print_model_evaluation_results(map_k_evaluation_results=None, recall_k_evaluation_results=None):
    ''' Print the MAP@K and Recall@K evaluation results for the model '''
    # Print the MAP@K evaluation results
    if map_k_evaluation_results is not None:
        additional_info = ""
        if "info" in map_k_evaluation_results and map_k_evaluation_results["info"] is not None and len(map_k_evaluation_results["info"].keys()) > 0:
            additional_info = " (" + ", ".join(
                [f"{key}: {value}" for key, value in map_k_evaluation_results["info"].items()]) + ")"
        print(
            f"MAP@{map_k_evaluation_results['k_documents']} for the {map_k_evaluation_results['model']} model{additional_info}:")
        print(f"  > {map_k_evaluation_results['mean_average_precision']}")
        print(f"  Computed on {map_k_evaluation_results['n_queries']} queries")
        print(f"  Single queries precision:")
        for query_id in map_k_evaluation_results['evaluated_queries'].keys():
            print(
                f"    Query {query_id}: {map_k_evaluation_results['evaluated_queries'][query_id]}")
    else:
        print(f"No MAP@K evaluation results for the model")
    # Print the Recall@K evaluation results
    if recall_k_evaluation_results is not None:
        additional_info = ""
        if "info" in map_k_evaluation_results and recall_k_evaluation_results["info"] is not None and len(recall_k_evaluation_results["info"].keys()) > 0:
            additional_info = " (" + ", ".join(
                [f"{key}: {value}" for key, value in recall_k_evaluation_results["info"].items()]) + ")"
        print(
            f"Recall@{recall_k_evaluation_results['k_documents']} results for the {recall_k_evaluation_results['model']} model{additional_info}:")
        for i in range(len(recall_k_evaluation_results['recall_at_k_results'])):
            print(
                f"  > {recall_k_evaluation_results['recall_at_k_results'][i]}")
            print(
                f"    Computed for query {recall_k_evaluation_results['query_ids'][i]}")
    else:
        print(f"No Recall@K evaluation results for the model")
