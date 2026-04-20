import asyncio
import csv
import heapq
import json
import logging
import os
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
# from gen_openai_emb import generate_openai_embeddings
from sentence_transformers import util
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    NanoBEIREvaluator,
    TripletEvaluator,
)
from sentence_transformers.evaluation.NanoBEIREvaluator import (
    DatasetNameType,
    dataset_name_to_id,
)
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import (
    is_datasets_available,
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
)
from torch import Tensor
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


def _embedding_info(emb, model) -> dict:
    # Accept torch.Tensor or numpy
    import numpy as np
    import torch
    if isinstance(emb, torch.Tensor):
        arr = emb.detach().cpu().numpy()
        dtype = str(emb.dtype)
    else:
        arr = np.asarray(emb)
        dtype = str(arr.dtype)

    n = int(arr.shape[0])
    dim = int(arr.shape[1])


    # Map dtype to element size in bits
    dtype_bits = {
        "float32": 32, "torch.float32": 32, "float": 32,
        "float16": 16, "torch.float16": 16, "half": 16,
        "bfloat16": 16, "torch.bfloat16": 16,
        "uint8": 8, "torch.uint8": 8,
        "int8": 8, "torch.int8": 8,
    }
    element_size_bit = dtype_bits.get(dtype, 32)
    # breakpoint()
    return {"n": n, "dimension": dim, "element_size_bit": element_size_bit}


class QuantizedSentenceTransformer(SentenceTransformer):
    def __init__(
        self,
        model_name_or_path=None,
        quant_bits: int = 8,
        calibration_embeddings=None,
        calibration_texts=None,
        calibration_sample_size: int = 1000,
        *args,
        **kwargs,
    ):
        """
        Wrapper that pre-computes calibration embeddings (stored as numpy)
        and uses sentence_transformers.quantization.quantize_embeddings in encode().
        """
        if model_name_or_path is not None:
            super().__init__(model_name_or_path, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

        self.quant_bits = quant_bits
        self.precision_map = {8: "int8", 4: "int4", 1: "ubinary"}

        # 1. Prioritize pre-computed embeddings (convert to numpy if needed)
        if calibration_embeddings is not None:
            if isinstance(calibration_embeddings, torch.Tensor):
                self.calibration_embeddings = calibration_embeddings.detach().cpu().numpy()
            else:
                self.calibration_embeddings = np.asarray(calibration_embeddings)

        # 2. Otherwise, compute from texts (store as numpy)
        elif calibration_texts is not None:
            import random

            if len(calibration_texts) > calibration_sample_size:
                print(
                    f"Sampling {calibration_sample_size} texts from {len(calibration_texts)} for calibration..."
                )
                calibration_texts = random.sample(
                    calibration_texts, calibration_sample_size
                )

            print(f"Computing calibration embeddings from {len(calibration_texts)} texts...")

            # Encode without tensor conversion to get numpy directly
            calib_embs = super().encode(
                calibration_texts,
                show_progress_bar=True,
                convert_to_tensor=False,
                batch_size=32,
            )
            # Ensure numpy format
            if isinstance(calib_embs, torch.Tensor):
                calib_embs = calib_embs.detach().cpu().numpy()
            self.calibration_embeddings = np.asarray(calib_embs)
        else:
            self.calibration_embeddings = None
            print(
                f"WARNING: QuantizedSentenceTransformer initialized with {quant_bits} bits but NO calibration data."
            )

    def set_calibration_embeddings(self, embeddings):
        if isinstance(embeddings, torch.Tensor):
            self.calibration_embeddings = embeddings.detach().cpu().numpy()
        else:
            self.calibration_embeddings = np.asarray(embeddings)

    def encode(self, sentences, *args, **kwargs) -> Tensor:
        """
        Encode sentences and optionally quantize.
        Always returns torch tensor (convert_to_tensor is forced to True).
        """
        kwargs["convert_to_tensor"] = True
        
        # 1. Compute embeddings (will be torch tensor)
        embeddings = super().encode(sentences, *args, **kwargs)

        # 2. Quantize if calibration data is available
        if self.quant_bits and self.calibration_embeddings is not None:
            precision = self.precision_map.get(self.quant_bits, "int8")
            
            # CHECK: If embeddings are already quantized (int types), skip quantization
            if isinstance(embeddings, torch.Tensor) and not embeddings.is_floating_point():
                logger.info(
                    f"Embeddings already quantized to {embeddings.dtype}. Skipping re-quantization."
                )
                return embeddings
            
            try:
                # Convert embeddings to numpy for quantize_embeddings
                if isinstance(embeddings, torch.Tensor):
                    embeddings_np = embeddings.detach().cpu().numpy()
                else:
                    embeddings_np = np.asarray(embeddings)
                
                # Quantize
                embeddings_q = quantize_embeddings(
                    embeddings_np,
                    precision=precision,
                    calibration_embeddings=self.calibration_embeddings,
                )
                
                # Convert back to torch tensor
                if not isinstance(embeddings_q, torch.Tensor):
                    embeddings = torch.from_numpy(np.asarray(embeddings_q))
                else:
                    embeddings = embeddings_q
            except Exception as e:
                logger.warning(f"Quantization failed (precision={precision}): {e}. Returning embeddings as-is.")
        
        return embeddings
    

class UBinarySentenceTransformer(SentenceTransformer):
    """
    A SentenceTransformer model that always outputs binary embeddings.
    """

    def encode(self, sentences, *args, **kwargs) -> Tensor:
        """
        Overrides the default encode method to enforce binary precision.
        """
        # Set the desired arguments for binary embeddings
        kwargs["precision"] = "ubinary"
        kwargs["convert_to_tensor"] = True

        # Call the original encode method from the parent class (SentenceTransformer)
        return super().encode(sentences, *args, **kwargs)


class PQSentenceTransformer(SentenceTransformer):
    """
    Encodes with float32, trains a faiss PQ index on the corpus,
    then returns approximate PQ-reconstructed embeddings for retrieval.
    """
    def __init__(self, model_name_or_path=None, M=96, nbits=8, *args, **kwargs):
        # M = number of subspaces (dim must be divisible by M)
        # nbits = bits per subspace (8 → 256 codewords)
        if model_name_or_path is not None:
            super().__init__(model_name_or_path, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self.M = M
        self.nbits = nbits
        self.index = None  # faiss PQ index, built after corpus encode

    def fit_pq(self, corpus_embeddings: np.ndarray):
        import faiss
        corpus_embeddings = np.asarray(corpus_embeddings, dtype=np.float32)
        d = corpus_embeddings.shape[1]
        assert d % self.M == 0, f"Embedding dim {d} must be divisible by M={self.M}"
        self.index = faiss.IndexPQ(d, self.M, self.nbits)
        self.index.train(corpus_embeddings)
        self.index.add(corpus_embeddings)

    def get_pq_reconstructed(self, corpus_embeddings: np.ndarray) -> torch.Tensor:
        """Encode to PQ codes then decode back to float — approximated vectors."""
        corpus_embeddings = np.asarray(corpus_embeddings, dtype=np.float32)
        codes = self.index.sa_encode(corpus_embeddings)
        reconstructed = self.index.sa_decode(codes)
        return torch.from_numpy(reconstructed)


class TurboQuantSentenceTransformer(SentenceTransformer):
    """
    Wraps a SentenceTransformer with TurboQuantMSE vector quantization.
    Embeddings are encoded normally (including multi-GPU pool), then
    quantized and reconstructed to float32 via apply_turboquant(), which
    is called by compute_metrices after the pool encode completes — the
    same pattern PQ uses for fit_pq / get_pq_reconstructed.
    Standard cosine/dot similarity works downstream without a special distance fn.
    The tq_bits attribute is read by the evaluator to override element_size_bit
    in the emb_info JSON (since the reconstructed tensor is float32, not N-bit).
    """
    def __init__(self, model_name_or_path=None, tq_bits=4, *args, **kwargs):
        if model_name_or_path is not None:
            super().__init__(model_name_or_path, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self.tq_bits = tq_bits
        self._tq = None  # lazy init: TurboQuantMSE needs dim, known after first call

    def apply_turboquant(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize embeddings. Initialises TurboQuantMSE lazily."""
        embeddings = embeddings.cpu().float()
        dim = embeddings.shape[-1]
        if self._tq is None:
            from turboquant import TurboQuantMSE
            self._tq = TurboQuantMSE(dim=dim, bits=self.tq_bits, device="cpu")
        indices, norms = self._tq.quantize(embeddings)
        return self._tq.dequantize(indices, norms)


# Define a dummy placeholder for the model card data attribute
class DummyModelCardData:
    def set_evaluation_metrics(self, *args, **kwargs):
        """This is a dummy method. It does nothing."""
        pass


# Define the complete, self-contained dummy model
class DummyModel:
    """
    A standalone placeholder model that mimics all necessary attributes
    and methods for the InformationRetrievalEvaluator when using
    pre-computed embeddings.
    """

    def __init__(self, similarity=util.cos_sim, similarity_fn_name="cosine"):
        self.similarity = similarity
        self.similarity_fn_name = similarity_fn_name
        self.model_card_data = DummyModelCardData()

    def start_multi_process_pool(self, *args, **kwargs):
        """Dummy method. Returns an empty dict."""
        return {}

    def stop_multi_process_pool(self, pool):
        """Dummy method. Does nothing."""
        pass


class MultiGPUInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    def __init__(self, encode_batch_size=128, encode_chunk_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encode_batch_size = encode_batch_size
        self.encode_chunk_size = encode_chunk_size

    # Override the compute_metrices method to customize the evaluation process
    def compute_metrices(
        self,
        model: SentenceTransformer,
        corpus_model=None,
        corpus_embeddings: Tensor | None = None,
        output_path: str | None = None,
        corpus_df: pd.DataFrame | None = None,  # new parameter
        query_df: pd.DataFrame | None = None,  # new parameter
        query_prompt_str: str | None = None,  # new parameter
    ) -> dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        pool = model.start_multi_process_pool()
        try:
            # Compute embedding for the queries
            if query_df is None:
                logger.info("Computing embeddings for queries")
                print("Computing query embeddings")
                query_embeddings = model.encode(
                    self.queries,
                    pool=pool,
                    batch_size=self.encode_batch_size,
                    chunk_size=self.encode_chunk_size,
                    show_progress_bar=True,
                    prompt_name=self.query_prompt_name,
                    prompt=query_prompt_str
                    if query_prompt_str is not None
                    else self.query_prompt,
                )
            else:
                # filter and reorder the embeddings to only include the queries we have
                query_df = query_df.set_index("id").loc[self.queries_ids].reset_index()
                query_embeddings = torch.tensor(query_df["embeddings"].values.tolist())

            # get info about query embeddings
            query_emb_info = _embedding_info(query_embeddings, model)

            queries_result_list = {}
            for name in self.score_functions:
                queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

            if (corpus_embeddings is None) and (corpus_df is None):
                logger.info("Computing embeddings for corpus")
                print("Computing corpus embeddings")
                corpus_embeddings = model.encode(
                    self.corpus,
                    pool=pool,
                    batch_size=self.encode_batch_size,
                    chunk_size=self.encode_chunk_size,
                    show_progress_bar=True,
                    prompt_name=self.corpus_prompt_name,
                    prompt=self.corpus_prompt,
                )
            elif corpus_df is not None:
                corpus_embeddings = torch.tensor(corpus_df["embeddings"].values.tolist())
        finally:
            model.stop_multi_process_pool(pool)

        # If PQ model: fit codebook on corpus, then replace corpus embeddings
        # with PQ reconstructions. Queries stay float32 (asymmetric PQ).
        if hasattr(model, 'fit_pq'):
            corpus_np = corpus_embeddings.detach().cpu().numpy() \
                if isinstance(corpus_embeddings, torch.Tensor) else np.asarray(corpus_embeddings)
            model.fit_pq(corpus_np)
            corpus_embeddings = model.get_pq_reconstructed(corpus_np)

        # If TurboQuant model: quantize+dequantize both query and corpus embeddings.
        # encode() returns numpy by default, so convert before applying.
        if hasattr(model, 'apply_turboquant'):
            if not isinstance(query_embeddings, torch.Tensor):
                query_embeddings = torch.from_numpy(np.asarray(query_embeddings, dtype=np.float32))
            query_embeddings = model.apply_turboquant(query_embeddings)
            if not isinstance(corpus_embeddings, torch.Tensor):
                corpus_embeddings = torch.from_numpy(np.asarray(corpus_embeddings, dtype=np.float32))
            corpus_embeddings = model.apply_turboquant(corpus_embeddings)

        # get info about corpus embeddings
        corpus_emb_info = _embedding_info(corpus_embeddings, model)
        # For PQ, the actual stored corpus is M*nbits bits/vector, not float32.
        # Set dimension=M and element_size_bit=nbits so that
        # n * dimension * element_size_bit = n * M * nbits (correct PQ index size).
        if hasattr(model, 'fit_pq'):
            corpus_emb_info["dimension"] = model.M
            corpus_emb_info["element_size_bit"] = model.nbits

        # For TurboQuant: reconstructed tensor is float32 but logical storage is tq_bits.
        # Override element_size_bit on both query and corpus so emb_info reflects true size.
        if hasattr(model, 'tq_bits'):
            corpus_emb_info["element_size_bit"] = model.tq_bits
            query_emb_info["element_size_bit"] = model.tq_bits

        # Store for caller to consume (tuple of dicts)
        self.last_embedding_info = {
            "queries": query_emb_info,
            "corpus": corpus_emb_info,
        }


        # Iterate over chunks of the corpus
        for corpus_start_idx in trange(
            0,
            len(self.corpus),
            self.corpus_chunk_size,
            desc="Corpus Chunks",
            disable=not self.show_progress_bar,
        ):
            corpus_end_idx = min(
                corpus_start_idx + self.corpus_chunk_size,
                len(self.corpus),
            )

            # Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = self.embed_inputs(
                    corpus_model,
                    self.corpus[corpus_start_idx:corpus_end_idx],
                    encode_fn_name="document",
                    prompt_name=self.corpus_prompt_name,
                    prompt=self.corpus_prompt,
                )
            else:
                sub_corpus_embeddings = corpus_embeddings[
                    corpus_start_idx:corpus_end_idx
                ]

            # Prepare tensors for scoring: q_embs = query embeddings, c_embs = corpus chunk embeddings
            q_embs = query_embeddings
            c_embs = sub_corpus_embeddings

            # If numpy arrays, convert to torch tensors
            if isinstance(q_embs, np.ndarray):
                q_embs = torch.from_numpy(q_embs)
            if isinstance(c_embs, np.ndarray):
                c_embs = torch.from_numpy(c_embs)

            # Ensure tensors are float for standard similarity math (cosine/dot)
            # Handles int8/int4/ubinary quantized outputs by casting them to float
            if torch.is_tensor(q_embs) and not torch.is_floating_point(q_embs):
                q_embs = q_embs.float()
            if torch.is_tensor(c_embs) and not torch.is_floating_point(c_embs):
                c_embs = c_embs.float()


            # Compute cosine similarites
            for name, score_function in self.score_functions.items():
                # pair_scores = score_function(query_embeddings, sub_corpus_embeddings)
                pair_scores = score_function(q_embs, c_embs)

                # Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores,
                    min(max_k, len(pair_scores[0])),
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(
                        pair_scores_top_k_idx[query_itr],
                        pair_scores_top_k_values[query_itr],
                    ):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        # NOTE: TREC/BEIR/MTEB skips cases where the corpus_id is the same as the query_id, e.g.:
                        # if corpus_id == self.queries_ids[query_itr]:
                        #     continue
                        # This is not done here, as this might be unexpected behaviour if the user just uses
                        # sets of integers from 0 as query_ids and corpus_ids.
                        if len(queries_result_list[name][query_itr]) < max_k:
                            # heaqp tracks the quantity of the first element in the tuple
                            heapq.heappush(
                                queries_result_list[name][query_itr],
                                (score, corpus_id),
                            )
                        else:
                            heapq.heappushpop(
                                queries_result_list[name][query_itr],
                                (score, corpus_id),
                            )

        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {
                        "corpus_id": corpus_id,
                        "score": score,
                    }

        if self.write_predictions and output_path is not None:
            for name in queries_result_list:
                base_filename = self.predictions_file.replace(
                    ".jsonl",
                    f"_{name}.jsonl",
                )
                json_path = os.path.join(output_path, base_filename)
                mode = "w"  # Always create a new file for each score function

                with open(json_path, mode=mode, encoding="utf-8") as fOut:
                    for query_itr in range(len(queries_result_list[name])):
                        query_id = self.queries_ids[query_itr]
                        query_text = self.queries[query_itr]
                        results = queries_result_list[name][query_itr]

                        # Sort results by score in descending order
                        results = sorted(
                            results,
                            key=lambda x: x["score"],
                            reverse=True,
                        )

                        prediction = {
                            "query_id": query_id,
                            "query": query_text,
                            "results": results,
                        }

                        fOut.write(json.dumps(prediction) + "\n")

        logger.info(f"Queries: {len(self.queries)}")
        logger.info(f"Corpus: {len(self.corpus)}\n")

        # # Compute scores
        # for query_itr in range(len(queries_result_list["cosine"])):
        #     try:
        #         query_id = self.queries_ids[query_itr]
        #     except IndexError:
        #         print("query_itr: ", query_itr)

        scores = {
            name: self.compute_metrics(queries_result_list[name])
            for name in self.score_functions
        }

        # Output
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores


class MultiGPUTripletEvaluator(TripletEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # overrding the method
    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(
            f"TripletEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:",
        )

        pool = model.start_multi_process_pool()
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(
            self.truncate_dim,
        ):
            embeddings_anchors = model.encode(
                self.anchors,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                pool=pool,
            )
            embeddings_positives = model.encode(
                self.positives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                pool=pool,
            )
            embeddings_negatives = model.encode(
                self.negatives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                pool=pool,
            )
        model.stop_multi_process_pool(pool)
        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_functions = {
            "cosine": lambda anchors, positives, negatives: (
                pairwise_cos_sim(anchors, positives),
                pairwise_cos_sim(anchors, negatives),
            ),
            "dot": lambda anchors, positives, negatives: (
                pairwise_dot_score(anchors, positives),
                pairwise_dot_score(anchors, negatives),
            ),
            "manhattan": lambda anchors, positives, negatives: (
                pairwise_manhattan_sim(anchors, positives),
                pairwise_manhattan_sim(anchors, negatives),
            ),
            "euclidean": lambda anchors, positives, negatives: (
                pairwise_euclidean_sim(anchors, positives),
                pairwise_euclidean_sim(anchors, negatives),
            ),
        }

        metrics = {}
        for fn_name in self.similarity_fn_names:
            if fn_name in similarity_functions:
                positive_scores, negative_scores = similarity_functions[fn_name](
                    embeddings_anchors,
                    embeddings_positives,
                    embeddings_negatives,
                )
                accuracy = (
                    (positive_scores > negative_scores + self.margin[fn_name])
                    .float()
                    .mean()
                    .item()
                )
                metrics[f"{fn_name}_accuracy"] = accuracy
                logger.info(
                    f"Accuracy {fn_name.capitalize()} Similarity:\t{accuracy:.2%}",
                )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps] + list(metrics.values()))

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps] + list(metrics.values()))

        if len(self.similarity_fn_names) > 1:
            metrics["max_accuracy"] = max(metrics.values())

        if self.main_similarity_function:
            self.primary_metric = {
                SimilarityFunction.COSINE: "cosine_accuracy",
                SimilarityFunction.DOT_PRODUCT: "dot_accuracy",
                SimilarityFunction.EUCLIDEAN: "euclidean_accuracy",
                SimilarityFunction.MANHATTAN: "manhattan_accuracy",
            }.get(self.main_similarity_function)
        else:
            if len(self.similarity_fn_names) > 1:
                self.primary_metric = "max_accuracy"
            else:
                self.primary_metric = f"{self.similarity_fn_names[0]}_accuracy"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics


class MultiGPUNanoBEIREvaluator(NanoBEIREvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.last_embedding_info = {}

    # overrdinnf this method to replace the default InformationRetrievalEvaluator with MultiGPUInformationRetrievalEvaluator
    def _load_dataset(
        self,
        dataset_name: DatasetNameType,
        **ir_evaluator_kwargs,
    ) -> InformationRetrievalEvaluator:
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the NanoBEIREvaluator via `pip install datasets`.",
            )
        from datasets import load_dataset

        dataset_path = dataset_name_to_id[dataset_name.lower()]
        corpus = load_dataset(dataset_path, "corpus", split="train")
        queries = load_dataset(dataset_path, "queries", split="train")
        qrels = load_dataset(dataset_path, "qrels", split="train")
        corpus_dict = {
            sample["_id"]: sample["text"]
            for sample in corpus
            if len(sample["text"]) > 0
        }
        queries_dict = {
            sample["_id"]: sample["text"]
            for sample in queries
            if len(sample["text"]) > 0
        }
        qrels_dict = {}
        for sample in qrels:
            if sample["query-id"] not in qrels_dict:
                qrels_dict[sample["query-id"]] = set()
            qrels_dict[sample["query-id"]].add(sample["corpus-id"])

        if self.query_prompts is not None:
            ir_evaluator_kwargs["query_prompt"] = self.query_prompts.get(
                dataset_name,
                None,
            )
        if self.corpus_prompts is not None:
            ir_evaluator_kwargs["corpus_prompt"] = self.corpus_prompts.get(
                dataset_name,
                None,
            )
        human_readable_name = self._get_human_readable_name(dataset_name)
        return MultiGPUInformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=f"nanobeir__{human_readable_name}____evaluator",
            **ir_evaluator_kwargs,
        )

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
        corpus_dfs: dict = None,
        query_dfs: dict = None,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        per_metric_results = {}
        per_dataset_results = {}
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"
        logger.info(
            f"NanoBEIR Evaluation of the model on {self.dataset_names} dataset{out_txt}:",
        )

        if self.score_functions is None:
            self.score_functions = {model.similarity_fn_name: model.similarity}
            self.score_function_names = [model.similarity_fn_name]
            self._append_csv_headers(self.score_function_names)

        for evaluator in tqdm(
            self.evaluators,
            desc="Evaluating datasets",
            disable=not self.show_progress_bar,
        ):
            print(f"Evaluating {evaluator.name}")
            if corpus_dfs and query_dfs:
                evaluation = evaluator(
                    model,
                    output_path,
                    epoch,
                    steps,
                    corpus_df=corpus_dfs.get(evaluator.name.split("__")[1])
                    if corpus_dfs
                    else None,
                    query_df=query_dfs.get(evaluator.name.split("__")[1])
                    if query_dfs
                    else None,
                )
            else:
                evaluation = evaluator(model, output_path, epoch, steps)
            for k in evaluation:
                if self.truncate_dim:
                    # dataset, _, metric = k.split("_", maxsplit=2)
                    nanobeir_name, dataset_name, subset_name, rest_part = k.split("__")
                    evaluator_name, _, metric = rest_part.split("_", maxsplit=2)
                else:
                    # lets parse the key correctly
                    nanobeir_name, dataset_name, subset_name, rest_part = k.split("__")
                    evaluator_name, metric = rest_part.split("_", maxsplit=1)

                if metric not in per_metric_results:
                    per_metric_results[metric] = []
                # per_dataset_results[dataset + "_" + metric] = evaluation[k]
                per_dataset_results[
                    f"{nanobeir_name}__{dataset_name}__{subset_name}__{evaluator_name}_{metric}"
                ] = evaluation[k]
                per_metric_results[metric].append(evaluation[k])

            # update last embedding info
            self.last_embedding_info[evaluator.name] = evaluator.last_embedding_info
        agg_results = {}
        for metric in per_metric_results:
            agg_results[metric] = self.aggregate_fn(per_metric_results[metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(agg_results[f"{name}_accuracy@{k}"])

                for k in self.precision_recall_at_k:
                    output_data.append(agg_results[f"{name}_precision@{k}"])
                    output_data.append(agg_results[f"{name}_recall@{k}"])

                for k in self.mrr_at_k:
                    output_data.append(agg_results[f"{name}_mrr@{k}"])

                for k in self.ndcg_at_k:
                    output_data.append(agg_results[f"{name}_ndcg@{k}"])

                for k in self.map_at_k:
                    output_data.append(agg_results[f"{name}_map@{k}"])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [
                        (name, agg_results[f"{name}_ndcg@{max(self.ndcg_at_k)}"])
                        for name in self.score_function_names
                    ],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = (
                    f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"
                )

        avg_queries = np.mean([len(evaluator.queries) for evaluator in self.evaluators])
        avg_corpus = np.mean([len(evaluator.corpus) for evaluator in self.evaluators])
        logger.info(f"Average Queries: {avg_queries}")
        logger.info(f"Average Corpus: {avg_corpus}\n")

        for name in self.score_function_names:
            logger.info(f"Aggregated for Score Function: {name}")
            for k in self.accuracy_at_k:
                logger.info(
                    "Accuracy@{}: {:.2f}%".format(
                        k,
                        agg_results[f"{name}_accuracy@{k}"] * 100,
                    ),
                )

            for k in self.precision_recall_at_k:
                logger.info(
                    "Precision@{}: {:.2f}%".format(
                        k,
                        agg_results[f"{name}_precision@{k}"] * 100,
                    ),
                )
                logger.info(
                    "Recall@{}: {:.2f}%".format(
                        k,
                        agg_results[f"{name}_recall@{k}"] * 100,
                    ),
                )

            for k in self.mrr_at_k:
                logger.info("MRR@{}: {:.4f}".format(k, agg_results[f"{name}_mrr@{k}"]))

            for k in self.ndcg_at_k:
                logger.info(
                    "NDCG@{}: {:.4f}".format(k, agg_results[f"{name}_ndcg@{k}"]),
                )

        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        self.store_metrics_in_model_card_data(model, agg_results, epoch, steps)

        per_dataset_results.update(agg_results)

        return per_dataset_results


def get_embedding_for_dataset(
    dataset_config,
    embedding_path,
    dataset_name,
    model_name,
    subset=None,
    data_file=None,
):
    """
    Function to get embeddings for a specific dataset.
    If the embeddings do not exist, it generates them.
    """

    base_path = (
        os.path.join(embedding_path, dataset_name, subset)
        if subset is not None
        else os.path.join(embedding_path, dataset_name)
    )
    # if it is nanobeir then subset is None

    print(f"Base Path: {base_path}")
    corpus_path = os.path.join(base_path, "corpus_embeddings.parquet")
    queries_path = os.path.join(base_path, "queries_embeddings.parquet")

    if (not os.path.exists(corpus_path)) or (not os.path.exists(queries_path)):
        print("Could not find existing embeddings. Generating new ones...")
        dataset_input_path = None

        # Either path is not None eg. nasa sde v1, nasa sde v2, nasa smd ir (which is hf path)
        if dataset_config.get("path") is not None:
            dataset_input_path = dataset_config["path"]
        # either paths is not None like Nanobeir but path is non
        elif dataset_config.get("paths") is not None:
            # need to be updated
            dataset_input_path = dataset_config[
                "paths"
            ]  # there are multiple paths for different subsets
        # either path needs to be local like beir and also has subsets
        elif dataset_config.get("dataset_cache_path") is not None:
            dataset_input_path = os.path.join(
                dataset_config["dataset_cache_path"],
                subset,
            )

        if isinstance(dataset_input_path, dict):
            # when there is paths
            # genererate embedding for all subsets save it all and return dfs inside a dict
            corpus_dfs = {}
            queries_dfs = {}

            for name, path in dataset_input_path.items():
                corpus_path = os.path.join(base_path, name, "corpus_embeddings.parquet")
                queries_path = os.path.join(
                    base_path,
                    name,
                    "queries_embeddings.parquet",
                )

                if (os.path.exists(corpus_path)) and (os.path.exists(queries_path)):
                    print("Found existing embeddings for", name)
                    corpus_df = pd.read_parquet(corpus_path)
                    queries_df = pd.read_parquet(queries_path)

                else:
                    corpus_df, queries_df = asyncio.run(
                        generate_openai_embeddings(
                            path,
                            os.path.join(base_path, name),
                            model=model_name,
                        ),
                    )

                corpus_dfs[name] = corpus_df
                queries_dfs[name] = queries_df
            return corpus_dfs, queries_dfs
        else:
            corpus_df, queries_df = asyncio.run(
                generate_openai_embeddings(
                    dataset_input_path,
                    base_path,
                    model=model_name,
                ),
            )

            return corpus_df, queries_df

    else:
        print("Found existing embeddings. Loading them...")
        corpus_df = pd.read_parquet(corpus_path)
        queries_df = pd.read_parquet(queries_path)
        return corpus_df, queries_df


def hamming_similarity_from_distance(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes Hamming similarity based on Hamming distance for packed binary tensors.
    Memory-efficient version using PyTorch operations and chunking.
    """

    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    # Ensure tensors are on the CPU and are of type uint8 for bitwise operations
    a_cpu = a.cpu().to(torch.uint8)
    b_cpu = b.cpu().to(torch.uint8)

    # Get dimensions
    num_queries = a_cpu.shape[0]
    num_corpus = b_cpu.shape[0]
    embedding_dim_bytes = a_cpu.shape[1]

    # The total number of bits
    vector_length = embedding_dim_bytes * 8

    # Process in smaller chunks to manage memory
    chunk_size = min(500, num_corpus)  # Reduce chunk size for safety

    # Initialize result tensor
    result = torch.zeros((num_queries, num_corpus), dtype=torch.float32)

    # Precompute bit count lookup table for efficiency
    bit_count_table = torch.tensor(
        [bin(i).count("1") for i in range(256)],
        dtype=torch.int32,
    )

    for start_idx in range(0, num_corpus, chunk_size):
        end_idx = min(start_idx + chunk_size, num_corpus)

        # Get chunk of corpus embeddings
        b_chunk = b_cpu[start_idx:end_idx]

        # Compute XOR for this chunk using broadcasting
        # Shape: (num_queries, chunk_size, embedding_dim_bytes)
        differences = a_cpu.unsqueeze(1) ^ b_chunk.unsqueeze(0)

        # Count bits using lookup table (more memory efficient)
        # Flatten the last dimension for vectorized lookup
        flat_diffs = differences.view(-1)
        bit_counts = bit_count_table[flat_diffs.long()]

        # Reshape back and sum over the embedding dimension
        bit_counts = bit_counts.view(
            num_queries,
            end_idx - start_idx,
            embedding_dim_bytes,
        )
        hamming_distances = bit_counts.sum(dim=2)

        # Convert to similarity
        chunk_similarities = vector_length - hamming_distances.float()
        result[:, start_idx:end_idx] = chunk_similarities

    return result
