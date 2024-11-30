# import base64
# import json
# import logging
# import os
# import tempfile
# from io import BytesIO
# from pathlib import Path
# from typing import List, Optional

# import anthropic
# import numpy as np
# import torch
# from colpali_engine.models import ColQwen2, ColQwen2Processor
# from dotenv import load_dotenv
# from fastapi import APIRouter, HTTPException
# from PIL import Image
# from pydantic import BaseModel
# from torch.utils.data import DataLoader
# from vespa.application import Vespa


# def create_temp_cert_files(cert_content, key_content):
#     """Create temporary files for certificates and return their paths"""
#     try:
#         # Create a temporary directory
#         temp_dir = Path(tempfile.mkdtemp())
#         logger.info(f"Created temporary directory: {temp_dir}")

#         # Create cert file
#         cert_path = temp_dir / "cert.pem"
#         with open(cert_path, "w") as f:
#             f.write(cert_content)
#         logger.info(f"Created certificate file: {cert_path}")

#         # Create key file
#         key_path = temp_dir / "key.pem"
#         with open(key_path, "w") as f:
#             f.write(key_content)
#         logger.info(f"Created key file: {key_path}")

#         return str(cert_path), str(key_path)
#     except Exception as e:
#         logger.error(f"Error creating temporary certificate files: {str(e)}")
#         raise


# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()
# logger.info("Environment variables loaded")

# # Load environment variables
# load_dotenv()

# router = APIRouter()

# # Get environment variables
# VESPA_URL = os.getenv("VESPA_URL")
# VESPA_CERT_PATH = os.getenv("VESPA_CERT_PATH")
# VESPA_KEY_PATH = os.getenv("VESPA_KEY_PATH")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# logger.info(f"VESPA_URL: {VESPA_URL}")
# logger.info(f"VESPA_CERT_PATH: {VESPA_CERT_PATH}")
# logger.info(f"VESPA_KEY_PATH: {VESPA_KEY_PATH}")
# logger.info(f"ANTHROPIC_API_KEY exists: {bool(ANTHROPIC_API_KEY)}")

# # Validate required environment variables
# if not all([VESPA_URL, VESPA_CERT_PATH, VESPA_KEY_PATH, ANTHROPIC_API_KEY]):
#     raise ValueError("Missing required environment variables. Check your .env file.")

# # Initialize Anthropic client
# claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
# logger.info("Anthropic client initialized")

# # Initialize model and processor
# logger.info("Initializing ColQwen2 model and processor...")
# try:
#     model = ColQwen2.from_pretrained(
#         "vidore/colqwen2-v0.1",
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#     )
#     processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
#     model = model.eval()
#     logger.info("Model and processor initialized successfully")
# except Exception as e:
#     logger.error(f"Error initializing model: {str(e)}")
#     raise

# cert_key = """-----BEGIN CERTIFICATE-----
# MIIBNjCB3qADAgECAhB9Q1DuM00vBVfQSwy6NCR1MAoGCCqGSM49BAMCMB4xHDAa
# BgNVBAMTE2Nsb3VkLnZlc3BhLmV4YW1wbGUwHhcNMjQxMTI5MjIxNzE1WhcNMzQx
# MTI3MjIxNzE1WjAeMRwwGgYDVQQDExNjbG91ZC52ZXNwYS5leGFtcGxlMFkwEwYH
# KoZIzj0CAQYIKoZIzj0DAQcDQgAEGSo1C9mYABg+aUHlXzHTXi8d1oROXQr7dgXs
# gEEMurv6ldFi7xLyhipvCS4KysjSWHHms2oEVvDmLU0nfA1ClzAKBggqhkjOPQQD
# AgNHADBEAiAwvT7alE0Xr700wiDXXZOTlPq9TDCIv0ggS7jpgMITDgIgRyeoFqUz
# YpcVoziR0NRuIY/2na0bSA7t6Go3VOAxUpI=
# -----END CERTIFICATE-----
# """
# private_key = """-----BEGIN PRIVATE KEY-----
# MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgFFzKYGhc3dkdz8lZ
# QTBfrTbbsEkjSdVoAE0rIRCaaFKhRANCAAQZKjUL2ZgAGD5pQeVfMdNeLx3WhE5d
# Cvt2BeyAQQy6u/qV0WLvEvKGKm8JLgrKyNJYceazagRW8OYtTSd8DUKX
# -----END PRIVATE KEY-----
# """

# if cert_key is not None:
#     cert_key_new = cert_key.replace(r"\n", "\n")

# if private_key is not None:
#     private_key_new = private_key.replace(r"\n", "\n")


# # # Initialize Vespa connection with certificates
# # vespa_app = Vespa(
# #     url=VESPA_URL,
# #     cert=cert_key,
# #     key=private_key_new,
# # )

# try:
#     # Create temporary certificate files
#     logger.info("Creating temporary certificate files...")
#     cert_path, key_path = create_temp_cert_files(cert_key, private_key)

#     # Initialize Vespa connection
#     logger.info(
#         f"Initializing Vespa connection with cert: {cert_path} and key: {key_path}"
#     )
#     vespa_app = Vespa(url=VESPA_URL, cert=cert_path, key=key_path)
#     logger.info("Vespa connection initialized successfully")

# except Exception as e:
#     logger.error(f"Error setting up Vespa connection: {str(e)}")
#     raise


# # # Initialize Vespa connection
# # logger.info("Initializing Vespa connection...")
# # try:
# #     vespa_app = Vespa(url=VESPA_URL, cert=cert_key, key=private_key_new)
# #     logger.info("Vespa connection initialized successfully")
# # except Exception as e:
# #     logger.error(f"Error initializing Vespa connection: {str(e)}")
# #     raise


# class ChatQuery(BaseModel):
#     message: str


# class ChatResponse(BaseModel):
#     response: str
#     images: Optional[List[str]] = None


# @router.post("/chat", response_model=ChatResponse)
# async def chat_with_rag(query: ChatQuery):
#     logger.info(f"Received chat query: {query.message}")
#     try:
#         response = await process_rag_query(query.message)
#         return response
#     except Exception as e:
#         logger.error(f"Error in chat endpoint: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# async def process_rag_query(query_text: str):
#     try:
#         logger.info(f"query_text_type = {type(query_text)}\n query_text = {query_text}")

#         # Generate embeddings for the query
#         logger.info("Preparing DataLoader")
#         dataloader = DataLoader(
#             [query_text],
#             batch_size=1,
#             shuffle=False,
#             collate_fn=lambda x: processor.process_queries(x),
#         )

#         logger.info("Generating embeddings")
#         qs = []
#         for batch_query in dataloader:
#             with torch.no_grad():
#                 batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
#                 embeddings_query = model(**batch_query)
#                 qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

#         logger.info("Embeddings generated successfully")
#         logger.debug(f"Embeddings shape: {[q.shape for q in qs]}")

#         # Query preparation
#         float_query_embedding = {k: v.tolist() for k, v in enumerate(qs[0])}
#         binary_query_embeddings = {}
#         for k, v in float_query_embedding.items():
#             binary_query_embeddings[k] = (
#                 np.packbits(np.where(np.array(v) > 0, 1, 0)).astype(np.int8).tolist()
#             )

#         query_tensors = {
#             "input.query(qtb)": binary_query_embeddings,
#             "input.query(qt)": float_query_embedding,
#         }

#         # print(f"query_tensors = {len(query_tensors)}\n")

#         # Add query tensors for nearest neighbor search
#         target_hits_per_query_tensor = 5
#         logger.info("Adding nearest neighbor tensors")
#         for i in range(len(binary_query_embeddings)):
#             query_tensors[f"input.query(rq{i})"] = binary_query_embeddings[i]

#         # Construct nearest neighbor query
#         logger.info("Constructing nearest neighbor query")
#         nn = [
#             f"({{targetHits:{target_hits_per_query_tensor}}}nearestNeighbor(embedding,rq{i}))"
#             for i in range(len(binary_query_embeddings))
#         ]
#         nn = " OR ".join(nn)
#         logger.debug(f"NN query: {nn}")

#         # Query Vespa
#         logger.info("Querying Vespa")
#         async with vespa_app.asyncio(connections=1, timeout=180) as session:
#             logger.debug("Making Vespa query")
#             response = await session.query(
#                 yql=f"select title, url, image, page_number from finance_data_schema where {nn}",
#                 ranking="retrieval-and-rerank",
#                 timeout=180,
#                 hits=3,
#                 body={**query_tensors, "presentation.timing": True},
#             )

#             if not response.is_successful():
#                 logger.error(f"Vespa query failed: {response.status_code}")
#                 logger.error(f"Response body: {response.json()}")
#                 raise HTTPException(
#                     status_code=500, detail="Failed to get response from Vespa"
#                 )
#         # print(f"response = ***{response}***")

#         # Get the best matching result
#         logger.info("Processing Vespa response")
#         best_hit = response.hits[0]
#         image_data = best_hit["fields"]["image"]
#         logger.debug(f"Best hit title: {best_hit['fields']['title']}")
#         logger.debug(f"Best hit page: {best_hit['fields']['page_number']}")

#         # Create prompt for Claude
#         logger.info("Preparing Claude prompt")
#         prompt = f"""
#         User Query: {query_text}

#         Context: Looking at a financial document page from {best_hit['fields']['title']},
#         page {best_hit['fields']['page_number'] + 1}.

#         Please analyze the image and provide a detailed response to the user's query.
#         Focus on relevant financial information and data points visible in the document.
#         """

#         # Call Claude API
#         logger.info("Calling Claude API")
#         try:
#             message = claude_client.messages.create(
#                 model="claude-3-opus-20240229",
#                 max_tokens=1000,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": prompt},
#                             {
#                                 "type": "image",
#                                 "source": {
#                                     "type": "base64",
#                                     "media_type": "image/jpeg",
#                                     "data": image_data,
#                                 },
#                             },
#                         ],
#                     }
#                 ],
#             )
#             logger.info("Claude API response received")
#         except Exception as e:
#             logger.error(f"Error calling Claude API: {str(e)}")
#             raise

#         logger.info("Preparing final response")
#         return ChatResponse(response=message.content[0].text, images=[image_data])

#     except Exception as e:
#         logger.error(f"Error in process_rag_query: {str(e)}")
#         logger.error(f"Full error details: {repr(e)}")
#         if "query_tensors" in locals():
#             # logger.error(f"Query tensors: {json.dumps(query_tensors, indent=2)}")
#             logger.error(f"Query tensors: {len(query_tensors)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.get("/test-vespa")
# async def test_vespa_connection():
#     logger.info("Testing Vespa connection")
#     try:
#         status = vespa_app.get_application_status()
#         logger.info("Vespa connection test successful")
#         return {"status": "connected", "details": status}
#     except Exception as e:
#         logger.error(f"Vespa connection test failed: {str(e)}")
#         return {"status": "error", "message": str(e)}


import base64
import json
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import anthropic
import numpy as np
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import DataLoader
from vespa.application import Vespa

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

router = APIRouter()

# Get environment variables
VESPA_URL = os.getenv("VESPA_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

logger.info(f"VESPA_URL: {VESPA_URL}")
logger.info(f"ANTHROPIC_API_KEY exists: {bool(ANTHROPIC_API_KEY)}")

# Initialize Anthropic client
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
logger.info("Anthropic client initialized")

# Initialize model and processor
logger.info("Initializing ColQwen2 model and processor...")
try:
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
    model = model.eval()
    logger.info("Model and processor initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    raise


def create_temp_cert_files(cert_content, key_content):
    """Create temporary files for certificates and return their paths"""
    try:
        # Create a temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Created temporary directory: {temp_dir}")

        # Create cert file
        cert_path = temp_dir / "cert.pem"
        with open(cert_path, "w") as f:
            f.write(cert_content)
        logger.info(f"Created certificate file: {cert_path}")

        # Create key file
        key_path = temp_dir / "key.pem"
        with open(key_path, "w") as f:
            f.write(key_content)
        logger.info(f"Created key file: {key_path}")

        return str(cert_path), str(key_path)
    except Exception as e:
        logger.error(f"Error creating temporary certificate files: {str(e)}")
        raise


# # Your certificate strings
# cert_key = """-----BEGIN CERTIFICATE-----
# MIIBNjCB3qADAgECAhB9Q1DuM00vBVfQSwy6NCR1MAoGCCqGSM49BAMCMB4xHDAa
# BgNVBAMTE2Nsb3VkLnZlc3BhLmV4YW1wbGUwHhcNMjQxMTI5MjIxNzE1WhcNMzQx
# MTI3MjIxNzE1WjAeMRwwGgYDVQQDExNjbG91ZC52ZXNwYS5leGFtcGxlMFkwEwYH
# KoZIzj0CAQYIKoZIzj0DAQcDQgAEGSo1C9mYABg+aUHlXzHTXi8d1oROXQr7dgXs
# gEEMurv6ldFi7xLyhipvCS4KysjSWHHms2oEVvDmLU0nfA1ClzAKBggqhkjOPQQD
# AgNHADBEAiAwvT7alE0Xr700wiDXXZOTlPq9TDCIv0ggS7jpgMITDgIgRyeoFqUz
# YpcVoziR0NRuIY/2na0bSA7t6Go3VOAxUpI=
# -----END CERTIFICATE-----"""

# private_key = """-----BEGIN PRIVATE KEY-----
# MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgFFzKYGhc3dkdz8lZ
# QTBfrTbbsEkjSdVoAE0rIRCaaFKhRANCAAQZKjUL2ZgAGD5pQeVfMdNeLx3WhE5d
# Cvt2BeyAQQy6u/qV0WLvEvKGKm8JLgrKyNJYceazagRW8OYtTSd8DUKX
# -----END PRIVATE KEY-----"""

cert_key = """-----BEGIN CERTIFICATE-----
MIIBNjCB3qADAgECAhB9Q1DuM00vBVfQSwy6NCR1MAoGCCqGSM49BAMCMB4xHDAa
BgNVBAMTE2Nsb3VkLnZlc3BhLmV4YW1wbGUwHhcNMjQxMTI5MjIxNzE1WhcNMzQx
MTI3MjIxNzE1WjAeMRwwGgYDVQQDExNjbG91ZC52ZXNwYS5leGFtcGxlMFkwEwYH
KoZIzj0CAQYIKoZIzj0DAQcDQgAEGSo1C9mYABg+aUHlXzHTXi8d1oROXQr7dgXs
gEEMurv6ldFi7xLyhipvCS4KysjSWHHms2oEVvDmLU0nfA1ClzAKBggqhkjOPQQD
AgNHADBEAiAwvT7alE0Xr700wiDXXZOTlPq9TDCIv0ggS7jpgMITDgIgRyeoFqUz
YpcVoziR0NRuIY/2na0bSA7t6Go3VOAxUpI=
-----END CERTIFICATE-----
"""
private_key = """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgFFzKYGhc3dkdz8lZ
QTBfrTbbsEkjSdVoAE0rIRCaaFKhRANCAAQZKjUL2ZgAGD5pQeVfMdNeLx3WhE5d
Cvt2BeyAQQy6u/qV0WLvEvKGKm8JLgrKyNJYceazagRW8OYtTSd8DUKX
-----END PRIVATE KEY-----
"""

try:
    # Create temporary certificate files
    logger.info("Creating temporary certificate files...")
    cert_path, key_path = create_temp_cert_files(cert_key, private_key)

    # Initialize Vespa connection
    logger.info(
        f"Initializing Vespa connection with cert: {cert_path} and key: {key_path}"
    )
    vespa_app = Vespa(url=VESPA_URL, cert=cert_path, key=key_path)
    logger.info("Vespa connection initialized successfully")

except Exception as e:
    logger.error(f"Error setting up Vespa connection: {str(e)}")
    raise


def cleanup_temp_files():
    try:
        if "cert_path" in globals():
            os.remove(cert_path)
            os.remove(key_path)
            os.rmdir(os.path.dirname(cert_path))
            logger.info("Temporary certificate files cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")


# Register cleanup function
import atexit  # noqa: E402

atexit.register(cleanup_temp_files)


class ChatQuery(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    images: Optional[List[str]] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_with_rag(query: ChatQuery):
    logger.info(f"Received chat query: {query.message}")
    try:
        response = await process_rag_query(query.message)
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_rag_query(query_text: str):
    try:
        logger.info("Starting query processing")

        # Generate embeddings for the query
        logger.info("Preparing DataLoader")
        dataloader = DataLoader(
            [query_text],
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: processor.process_queries(x),
        )

        logger.info("Generating embeddings")
        qs = []
        for batch_query in dataloader:
            logger.debug(f"Processing batch, keys: {batch_query.keys()}")
            with torch.no_grad():
                batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
                embeddings_query = model(**batch_query)
                qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        logger.info("Embeddings generated successfully")
        logger.debug(f"Embeddings shape: {[q.shape for q in qs]}")

        # Query preparation
        logger.info("Preparing query tensors")
        float_query_embedding = {k: v.tolist() for k, v in enumerate(qs[0])}
        binary_query_embeddings = {}

        logger.debug("Converting to binary embeddings")
        for k, v in float_query_embedding.items():
            binary_query_embeddings[k] = (
                np.packbits(np.where(np.array(v) > 0, 1, 0)).astype(np.int8).tolist()
            )

        query_tensors = {
            "input.query(qtb)": binary_query_embeddings,
            "input.query(qt)": float_query_embedding,
        }

        # Add query tensors for nearest neighbor search
        target_hits_per_query_tensor = 5
        logger.info("Adding nearest neighbor tensors")
        for i in range(len(binary_query_embeddings)):
            query_tensors[f"input.query(rq{i})"] = binary_query_embeddings[i]

        # Construct nearest neighbor query
        logger.info("Constructing nearest neighbor query")
        nn = [
            f"({{targetHits:{target_hits_per_query_tensor}}}nearestNeighbor(embedding,rq{i}))"
            for i in range(len(binary_query_embeddings))
        ]
        nn = " OR ".join(nn)
        logger.debug(f"NN query: {nn}")

        # Query Vespa
        logger.info("Querying Vespa")
        try:
            async with vespa_app.asyncio(connections=1, timeout=180) as session:
                logger.debug("Making Vespa query")
                response = await session.query(
                    yql=f"select title, url, image, page_number from finance_data_schema where {nn}",
                    ranking="retrieval-and-rerank",
                    timeout=180,
                    hits=3,
                    body={**query_tensors, "presentation.timing": True},
                )

                if not response.is_successful():
                    logger.error(f"Vespa query failed: {response.status_code}")
                    logger.error(f"Response body: {response.json()}")
                    raise HTTPException(
                        status_code=500, detail="Failed to get response from Vespa"
                    )

                logger.info("Vespa query successful")
                logger.debug(f"Response hits: {len(response.hits)}")
        except Exception as vespa_error:
            logger.error(f"Error during Vespa query: {str(vespa_error)}")
            raise

        logger.info("Processing Vespa response")
        best_hit = response.hits[0]
        image_data = best_hit["fields"]["image"]
        logger.debug(f"Best hit title: {best_hit['fields']['title']}")
        logger.debug(f"Best hit page: {best_hit['fields']['page_number']}")

        # Create prompt for Claude
        logger.info("Preparing Claude prompt")
        prompt = f"""
        User Query: {query_text}
        
        Context: Looking at a financial document page from {best_hit['fields']['title']}, 
        page {best_hit['fields']['page_number'] + 1}.
        
        Please analyze the image and provide a detailed response to the user's query.
        Focus on relevant financial information and data points visible in the document.
        """

        # Call Claude API
        logger.info("Calling Claude API")
        try:
            message = claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                        ],
                    }
                ],
            )
            logger.info("Claude API response received")
        except Exception as claude_error:
            logger.error(f"Error calling Claude API: {str(claude_error)}")
            raise

        logger.info("Preparing final response")
        return ChatResponse(response=message.content[0].text, images=[image_data])

    except Exception as e:
        logger.error(f"Error in process_rag_query: {str(e)}")
        logger.error(f"Full error details: {repr(e)}")
        if "query_tensors" in locals():
            logger.error(f"Query tensors: {len(str(query_tensors))}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-vespa")
async def test_vespa_connection():
    logger.info("Testing Vespa connection")
    try:
        status = vespa_app.get_application_status()
        logger.info("Vespa connection test successful")
        return {"status": "connected", "details": status}
    except Exception as e:
        logger.error(f"Vespa connection test failed: {str(e)}")
        return {"status": "error", "message": str(e)}
