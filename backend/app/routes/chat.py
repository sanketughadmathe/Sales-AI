# from typing import List, Optional

# import numpy as np
# import torch
# from colpali_engine.models import ColQwen2, ColQwen2Processor
# from fastapi import APIRouter, HTTPException, status
# from pydantic import BaseModel
# from torch.utils.data import DataLoader
# from vespa.deployment import VespaCloud
# from vespa.package import (
#     HNSW,
#     ApplicationPackage,
#     Document,
#     Field,
#     FieldSet,
#     Schema,
# )

# router = APIRouter()

# # Initialize the model and processor
# model = ColQwen2.from_pretrained(
#     "vidore/colqwen2-v0.1",
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
# model = model.eval()

# # Define Vespa schema
# finance_schema = Schema(
#     name="finance_data_schema",
#     document=Document(
#         fields=[
#             Field(
#                 name="id", type="string", indexing=["summary", "index"], match=["word"]
#             ),
#             Field(name="url", type="string", indexing=["summary", "index"]),
#             Field(
#                 name="title",
#                 type="string",
#                 indexing=["summary", "index"],
#                 match=["text"],
#             ),
#             Field(name="page_number", type="int", indexing=["summary", "attribute"]),
#             Field(name="image", type="raw", indexing=["summary"]),
#             Field(
#                 name="embedding",
#                 type="tensor<int8>(patch{}, v[16])",
#                 indexing=["attribute", "index"],
#                 ann=HNSW(
#                     distance_metric="hamming",
#                     max_links_per_node=32,
#                     neighbors_to_explore_at_insert=400,
#                 ),
#             ),
#         ]
#     ),
#     fieldsets=[FieldSet(name="default", fields=["title"])],
# )

# # Initialize Vespa application package
# app_name = "capstone"
# application_package = ApplicationPackage(name=app_name, schema=[finance_schema])

# # Initialize Vespa Cloud client
# tenant_name = "prayogg"
# vespa_cloud = VespaCloud(
#     tenant=tenant_name,
#     application=app_name,
#     application_package=application_package,  # Add this line
# )

# try:
#     vespa_app = vespa_cloud.deploy()
# except Exception as e:
#     print(f"Error deploying Vespa: {str(e)}")
#     # For development, you might want to use a mock or disable Vespa functionality
#     vespa_app = None


# # Define request and response models
# class ChatRequest(BaseModel):
#     query: str


# class ChatResult(BaseModel):
#     text: str
#     image: Optional[str]
#     confidence: float
#     page_number: int
#     document_title: str


# class ChatResponse(BaseModel):
#     results: List[ChatResult]
#     query_time: float


# # Utility functions
# async def process_query(query: str):
#     # Generate query embeddings
#     dataloader = DataLoader(
#         [query],
#         batch_size=1,
#         shuffle=False,
#         collate_fn=lambda x: processor.process_queries(x),
#     )

#     query_embeddings = []
#     for batch_query in dataloader:
#         with torch.no_grad():
#             batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
#             embeddings_query = model(**batch_query)
#             query_embeddings.extend(list(torch.unbind(embeddings_query.to("cpu"))))

#     return query_embeddings[0]


# async def search_documents(query_embedding, target_hits=3):
#     float_query_embedding = {k: v.tolist() for k, v in enumerate(query_embedding)}
#     binary_query_embeddings = dict()

#     for k, v in float_query_embedding.items():
#         binary_query_embeddings[k] = (
#             np.packbits(np.where(np.array(v) > 0, 1, 0)).astype(np.int8).tolist()
#         )

#     query_tensors = {
#         "input.query(qtb)": binary_query_embeddings,
#         "input.query(qt)": float_query_embedding,
#     }

#     # Add query tensors for nearest neighbor search
#     for i in range(len(binary_query_embeddings)):
#         query_tensors[f"input.query(rq{i})"] = binary_query_embeddings[i]

#     # Construct nearest neighbor query
#     nn = [
#         f"({{targetHits:{target_hits}}}nearestNeighbor(embedding,rq{i}))"
#         for i in range(len(binary_query_embeddings))
#     ]
#     nn = " OR ".join(nn)

#     async with vespa_app.asyncio() as session:
#         response = await session.query(
#             yql=f"select title, url, image, page_number from finance_data_schema where {nn}",
#             ranking="retrieval-and-rerank",
#             timeout=120,
#             hits=target_hits,
#             body={**query_tensors, "presentation.timing": True},
#         )

#     return response


# @router.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(request: ChatRequest):
#     try:
#         # Process the query
#         query_embedding = await process_query(request.query)

#         # Search documents
#         search_response = await search_documents(query_embedding)

#         if not search_response.is_successful():
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Search failed",
#             )

#         # Process results
#         results = []
#         for hit in search_response.hits:
#             results.append(
#                 ChatResult(
#                     text=f"Found relevant information in {hit['fields']['title']}, page {hit['fields']['page_number'] + 1}",
#                     image=hit["fields"]["image"],
#                     confidence=hit["relevance"],
#                     page_number=hit["fields"]["page_number"] + 1,
#                     document_title=hit["fields"]["title"],
#                 )
#             )

#         query_time = search_response.json.get("timing", {}).get("searchtime", -1)

#         return ChatResponse(results=results, query_time=round(query_time, 2))

#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
#         )


import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import byaldi
import requests
from byaldi import RAGMultiModalModel
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pdf2image import convert_from_path
from pydantic import BaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize the RAG model
try:
    logger.info("Loading RAG model...")
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)

    # Create docs directory if it doesn't exist
    Path("./docs").mkdir(exist_ok=True)

    # Index the PDFs
    pdf_files = [
        {
            "name": "Tesla finance",
            "path": "./docs/tesla.pdf",
            "url": "https://ir.tesla.com/_flysystem/s3/sec/000162828024043432/tsla-20241023-gen.pdf",
        },
        {
            "name": "Basic Finance",
            "path": "./docs/finance.pdf",
            "url": "https://www.pwc.com/jm/en/research-publications/pdf/basic-understanding-of-a-companys-financials.pdf",
        },
    ]

    # Download and index PDFs
    for pdf in pdf_files:
        # Download PDF
        response = requests.get(pdf["url"])
        if response.status_code == 200:
            with open(pdf["path"], "wb") as f:
                f.write(response.content)

            # Index the PDF
            RAG.index(
                input_path=pdf["path"],
                index_name=f"./docs/{pdf['name']}",
                store_collection_with_index=True,
                overwrite=True,
            )
            logger.info(f"Indexed {pdf['name']}")

    logger.info("RAG model and indexing completed successfully!")
except Exception as e:
    logger.error(f"Error initializing RAG model: {str(e)}")
    raise

# Initialize Together API client
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    logger.warning("TOGETHER_API_KEY not found in environment variables")


# Define request and response models
class ChatRequest(BaseModel):
    query: str


class ChatResult(BaseModel):
    text: str
    image: Optional[str]
    confidence: float
    document_title: str


class ChatResponse(BaseModel):
    results: List[ChatResult]
    query_time: float


async def process_query_with_llama(query: str, image_b64: str) -> str:
    """Process query using Llama model via Together API"""
    try:
        data = {
            "model": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "stream": False,
            "logprobs": False,
        }

        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            "https://api.together.xyz/v1/chat/completions", headers=headers, json=data
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error processing with Llama model",
            )

    except Exception as e:
        logger.error(f"Error in Llama processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process with Llama model",
        )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """Handle chat requests"""
    try:
        # Search using RAG
        logger.info(f"Processing query: {request.query}")
        results = RAG.search(request.query, k=2)  # Get top 2 results

        # Process results
        chat_results = []
        for result in results:
            # Get Llama's interpretation of the image and query
            llama_response = await process_query_with_llama(
                request.query, result.base64
            )

            chat_results.append(
                ChatResult(
                    text=llama_response,
                    image=result.base64,
                    confidence=result.similarity_score,
                    document_title=result.metadata.get("source", "Unknown Document"),
                )
            )

        return ChatResponse(
            results=chat_results,
            query_time=0.1,  # You can implement actual timing if needed
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# Health check endpoint
@router.get("/health")
async def health_check():
    """Check if the service is healthy"""
    return {
        "status": "healthy",
        "rag_model_loaded": RAG is not None,
        "together_api_key_configured": bool(TOGETHER_API_KEY),
    }
