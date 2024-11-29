# -*- coding: utf-8 -*-
"""Copy of ColQwen_pdf_retrieval_and_interpretability_with_Vespa_on_finance_data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mj5B6MFccDBG5ceZo2svQHe1kprSUQWo

**Install dependencies and packages**
"""

!pip install -q transformers --upgrade
!pip install -q "colpali-engine>=0.3.2,<0.4.0" --upgrade
!apt-get install poppler-utils
!pip install -q pdf2image pypdf
!pip install -q openai
!pip install -q qwen_vl_utils
!pip install -q pyvespa vespacli

"""**Install packages**"""

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
    plot_similarity_map,
)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path
import base64
import io
from io import BytesIO
from IPython.display import display, HTML
import requests
from pypdf import PdfReader
import os
import json

from vespa.deployment import VespaCloud
from vespa.application import Vespa
from vespa.package import ApplicationPackage, Schema, Document, Field, FieldSet, HNSW
from vespa.package import RankProfile, Function, FirstPhaseRanking, SecondPhaseRanking
from vespa.io import VespaResponse, VespaQueryResponse

"""**Storing url for pdf in List**"""

# We will be working with these two pdfs
pdf_lists = [
    {"header": "Tesla finance", "url": "https://ir.tesla.com/_flysystem/s3/sec/000162828024043432/tsla-20241023-gen.pdf"},
    {"header": "Basic understanding of company finance", "url": "https://www.pwc.com/jm/en/research-publications/pdf/basic-understanding-of-a-companys-financials.pdf"}
]

"""**Loading the model**"""

model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.bfloat16, # switch to float16 if gpu does not support it
        device_map="auto",
    )
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
model = model.eval()

"""**Function for downloading pdf, converting to images every page**"""

# Utility function for downloading PDF from URL
def download_pdf(url):
    """returns: BytesIO: In-memory file object containing the PDF content."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return BytesIO(response.content)
    except Exception as e:
        raise Exception(f"Failed to download PDF: Status code {e}")


# Processes a single PDF to images of each page
def get_pdf_images(pdf_url):
    pdf_file = download_pdf(pdf_url)
    pdf_file.seek(0)  # Reset file pointer for image conversion

    # gets images of each page from the PDF
    temp_file_path = "test.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(pdf_file.read())
    pdf_images = convert_from_path(temp_file_path)

    return pdf_images

"""**Encode every image into base 64 & Create vector embeddings of image created from each page**"""

# Encodes images in base64
def encode_images_base64(images):
    """returns: list of str: Base64-encoded strings for each image."""
    base64_images = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        base64_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
    return base64_images

# Converts images to embeddings using a model and DataLoader
def generate_image_embeddings(images, model, processor, batch_size=2):  # adjust batch_size according to vram
    embeddings = []
    dataloader = DataLoader(
        images, batch_size=batch_size, shuffle=False,
        collate_fn=lambda x: processor.process_images(x)
    )
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            batch_embeddings = model(**batch).to("cpu")    # after the embeddings are generated, moving them to the CPU avoids clogging up GPU memory
            embeddings.extend(list(batch_embeddings))
    return embeddings

"""**Processing all the pdfs in the pdf_list, download the pdfs, convert each page to images and generate embeddings of each page using colqwen model.**"""

for pdf in pdf_lists:
    header = pdf.get("header", "Unnamed Document")
    url = pdf["url"]
    print(f"Process for {header} pdf started")
    #header = pdf.get("header", "Unnamed Document") - check later can be removed
    try:
        pdf_page_images = get_pdf_images(url)
        pdf["images"] = pdf_page_images
        pdf_embeddings = generate_image_embeddings(pdf_page_images, model, processor, batch_size=2)
        pdf["embeddings"] = pdf_embeddings
    except Exception as e:
        print(f"Error processing {header}: {e}")

# utility function to resize the images to a standard size
def resize_image(images, max_height=800):
    resized_images = []
    for image in images:
        width, height = image.size
        if height > max_height:
            ratio = max_height / height
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            resized_images.append(image.resize((new_width, new_height)))
        else:
            resized_images.append(image)
    return resized_images

"""Preparing tha data which will be inserted into Vespa vector database. The multivector embeddings were already generated above for the images. Here, we are doing slight modification -> binary quantization of vectors (i.e., float values will be converted to 0, 1), and images are converted to base64 encoding for easily storing the encoded string of the image to vespa.

Binary quantization: Say we have a vector [0.8, -0.5, 1.2, 0.3], upon binary quantization the vector is transformed to [1, 0, 1, 1]. Values less than equal to zero get 0, and more than 0 gets 1.
"""

vespa_feed = []
for pdf in pdf_lists:
    url = pdf["url"]
    title = pdf["header"]
    base64_images = encode_images_base64(resize_image(pdf["images"]))

    for page_number, (embedding, image, base64_img) in enumerate(zip(pdf["embeddings"], pdf["images"], base64_images)):
        embedding_dict = dict()
        for idx, patch_embedding in enumerate(embedding):
            binary_vector = (
                np.packbits(np.where(patch_embedding > 0, 1, 0))   # binary quantization of vectors
                .astype(np.int8)
                .tobytes()
                .hex()
            )
            embedding_dict[idx] = binary_vector
        page = {
            "id": hash(url + str(page_number)),
            "url": url,
            "title": title,
            "page_number": page_number,
            "image": base64_img,
            "embedding": embedding_dict,
        }
        vespa_feed.append(page)

len(vespa_feed)  # total numbers pages -> including all pdfs

vespa_feed[0].keys()

"""Here, we define the Vespa schema including fields, data types, indexing, and which fields will be available for search matching. This is pretty common in every database to define a schema before inserting data (like elastic search, weaviate, etc)."""

colpali_schema = Schema(
    name="finance_data_schema",   # name it according to your choice
    document=Document(
        fields=[
            Field(
                name="id",
                type="string",
                indexing=["summary", "index"],
                match=["word"]
            ),
            Field(
                name="url",
                type="string",
                indexing=["summary", "index"]
            ),
            Field(
                name="title",
                type="string",
                indexing=["summary", "index"],
                match=["text"],
                index="enable-bm25",
            ),
            Field(
                name="page_number",
                type="int",
                indexing=["summary", "attribute"]
            ),
            Field(
                name="image",
                type="raw",
                indexing=["summary"]
            ),
            Field(
                name="embedding",
                type="tensor<int8>(patch{}, v[16])",
                indexing=[
                    "attribute",
                    "index",
                ],  # adds HNSW index for candidate retrieval.
                ann=HNSW(
                    distance_metric="hamming",    # we are using hamming distance metric (hamming works the best with bits value, concept is to count the mismatch of bits numbers between two vectors)
                    max_links_per_node=32,
                    neighbors_to_explore_at_insert=400,
                ),
            )
        ]
    ),
    fieldsets=[
        FieldSet(name="default", fields=["title"])
    ]
)

"""Next, we define a ranking profile in Vespa that uses both Hamming distance (for binary similarity) and MaxSim (for float-based late interaction similarity) to rank documents. This profile enables a two-phase ranking: a first phase for initial filtering using binary similarity, and a second phase for re-ranking with continuous (float) similarity scores."""

input_query_tensors = []
MAX_QUERY_TERMS = 64
for i in range(MAX_QUERY_TERMS):
    input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])"))

input_query_tensors.append(("query(qt)", "tensor<float>(querytoken{}, v[128])"))
input_query_tensors.append(("query(qtb)", "tensor<int8>(querytoken{}, v[16])"))

colpali_retrieval_profile = RankProfile(
    name="retrieval-and-rerank",
    inputs=input_query_tensors,
    functions=[
        # Computes binary similarity using Hamming distance for query(qtb) and embedding.
        Function(
            name="max_sim_binary",
            expression="""
                sum(
                  reduce(
                    1/(1 + sum(
                        hamming(query(qtb), attribute(embedding)) ,v)
                    ),
                    max,
                    patch
                  ),
                  querytoken
                )
            """,
        ),
        # Computes similarity between the query(qt) tensor and the document’s embedding attribute.
        Function(
            name="max_sim",
            # query(qt) * unpack_bits(attribute(embedding)) : calculates the element-wise product of the query and document embeddings for similarity.
            # reduce(..., max, patch) : selects the maximum similarity value across patches
            # sum(..., querytoken) : aggregates the similarity score across all tokens.
            expression="""
                sum(
                    reduce(
                        sum(
                            query(qt) * unpack_bits(attribute(embedding)) , v
                        ),
                        max, patch
                    ),
                    querytoken
                )
            """,
        )
    ],
    first_phase=FirstPhaseRanking(expression="max_sim_binary"),   # first filtering of image patches using hamming distance (binary similarity)
    second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=5),  # reranking of filtered top 5 image patches using maxsim
)
colpali_schema.add_rank_profile(colpali_retrieval_profile)

"""With the configured application, we can now deploy it to Vespa Cloud.

To deploy the application to Vespa Cloud we need to create an account (free trial works) and then a tenant in the https://console.vespa-cloud.com/.

For this step tenant_name and app_name are required which you can setup in Vespa cloud after creating your account. Create an application in Vespa Cloud after tenant is created, and paste the name here below. Since we are not giving key below, you need to login interactively from here.
"""

app_name = "capstone2"   # name this according to your choice
vespa_application_package = ApplicationPackage(
    name=app_name, schema=[colpali_schema]
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Replace with your tenant name from the Vespa Cloud Console
tenant_name = "prayogg"

vespa_cloud = VespaCloud(
    tenant=tenant_name,
    application=app_name,
    application_package=vespa_application_package,
)

app: Vespa = vespa_cloud.deploy()

"""Here, we are inserting the full vespa_feed of 70 rows that we prepared before in the database."""

async with app.asyncio(connections=1, timeout=180) as session:
    for page in tqdm(vespa_feed):
        response: VespaResponse = await session.feed_data_point(
            data_id=page["id"], fields=page, schema="finance_data_schema"
        )
        if not response.is_successful():
            print(response.json())

"""Time for testing the retrieval with a given query"""

queries = [
    "current assets for 2018",
]

"""Generate the embeddings of the query using the same colqwen model"""

dataloader = DataLoader(
    queries,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: processor.process_queries(x),
)
qs = []
for batch_query in dataloader:
    with torch.no_grad():
        batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

"""Utility function to display the top retrieved results images in presentable format. Since we saved bse64 format image in vespa, first we need to decode that and then display."""

# Display query results images
#def display_query_results(query, response, hits=5):
    # query_time = response.json.get("timing", {}).get("searchtime", -1)
    # query_time = round(query_time, 2)
    # count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
    # html_content = f"<h3>Query text: '{query}', query time {query_time}s, count={count}, top results:</h3>"

    # for i, hit in enumerate(response.hits[:hits]):
    #     title = hit["fields"]["title"]
    #     url = hit["fields"]["url"]
    #     page = hit["fields"]["page_number"]
    #     image = hit["fields"]["image"]
    #     score = hit["relevance"]

    #     html_content += f"<h4>PDF Result {i + 1}</h4>"
    #     html_content += f'<p><strong>Title:</strong> <a href="{url}">{title}</a>, page {page+1} with score {score:.2f}</p>'
    #     html_content += (
    #         f'<img src="data:image/png;base64,{image}" style="max-width:100%;">'
    #     )

    # display(HTML(html_content))#

"""Now we will retrieve top 5 results (this number was defined in ranking process above rerank_count=5) of the given query and see the results."""

target_hits_per_query_tensor = (
    5  # this is a hyperparameter that can be tuned for speed versus accuracy
)
async with app.asyncio(connections=1, timeout=180) as session:
    for idx, query in enumerate(queries):
        try:
            float_query_embedding = {k: v.tolist() for k, v in enumerate(qs[idx])}
            binary_query_embeddings = dict()
            for k, v in float_query_embedding.items():
                binary_query_embeddings[k] = (
                    np.packbits(np.where(np.array(v) > 0, 1, 0)).astype(np.int8).tolist()
                )

            query_tensors = {
                "input.query(qtb)": binary_query_embeddings,
                "input.query(qt)": float_query_embedding,
            }

            # Add query tensors for nearest neighbor search
            for i in range(len(binary_query_embeddings)):
                query_tensors[f"input.query(rq{i})"] = binary_query_embeddings[i]

            # Construct nearest neighbor query
            nn = [f"({{targetHits:{target_hits_per_query_tensor}}}nearestNeighbor(embedding,rq{i}))"
                  for i in range(len(binary_query_embeddings))]
            nn = " OR ".join(nn)

            # Make query with error logging
            response = await session.query(
                yql=f"select title, url, image, page_number from finance_data_schema where {nn}",
                ranking="retrieval-and-rerank",
                timeout=120,
                hits=3,
                body={**query_tensors, "presentation.timing": True},
            )

            if not response.is_successful():
                print(f"Error in response: {response.status_code}")
                print(f"Response body: {response.json()}")
                continue

            #display_query_results(query, response)

        except Exception as e:
            print(f"Error processing query {idx}: {str(e)}")
            print(f"Query tensors: {json.dumps(query_tensors, indent=2)}")

# best_hit = response.hits[0]
# score = best_hit["relevance"]
# display(score)
# display(best_hit)

best_hit = response.hits[0]
pdf_url = best_hit["fields"]["url"]
pdf_title = best_hit["fields"]["title"]
score = best_hit["relevance"]
images = best_hit["fields"]["image"]
image_data = base64.b64decode(images)
image = Image.open(BytesIO(image_data))
display(image)

!pip install claudette
from google.colab import userdata
import base64
import os
os.environ["ANTHROPIC_API_KEY"]= userdata.get('ANTHROPIC_API_KEY')
from claudette import *
chat = Chat(models[1])

chat ([image_data, queries[0]])

"""The top image (idx 11) has the tokens in the query "balance", "1 July 2017", "equity". So the retrieval was pretty accurate. The rest of the extracted images might not have much context considering the data is limited (70 pages only in total), and there is not much overlap of data to extract."""