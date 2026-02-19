# Chapter 4: Multimodal Modular RAG for Drone Technology

## 1. What is Multimodal Modular RAG?

**Multimodal**: Combines different data forms (text, images, audio, video)

**Modular**: Uses distinct specialized modules for different data types and tasks
- Text module: LLM processing
- Image module: Computer vision, object detection
- Each module optimized for its data type

**Multisource**: Retrieves from multiple datasets simultaneously

**Key Advantage**: Comprehensive responses leveraging diverse information sources

---

## 2. System Architecture

### Workflow Components:

**D4 - Data Retrieval**:
- Load LLM textual dataset (Chapter 3)
- Load multimodal VisDrone dataset (images + labels)
- Initialize query engines for both datasets

**G1 - User Input**:
- Single baseline query for both modules
- Example: "How do drones identify a truck?"

**G2/G4 - Generation & Retrieval** (Overlapping):
- LlamaIndex seamlessly retrieves and generates
- OpenAI models for text processing
- Image retrieval via metadata and labels

**E - Evaluation**:
- Text performance: Cosine similarity
- Image performance: GPT-4o vision analysis
- Combined multimodal metric

---

## 3. Drone Technology Use Cases

**Industries**:
- Aerial photography
- Agricultural monitoring
- Search and rescue operations
- Wildlife tracking
- Commercial deliveries
- Infrastructure inspections
- Environmental research
- Traffic management
- Firefighting
- Law enforcement surveillance

**Computer Vision**: Identify objects (pedestrians, cars, trucks, bicycles, etc.)

---

## 4. Dataset Structure

### LLM Dataset (from Chapter 3):
```
Tensors:
- id: Unique string identifier
- text: Document content
- metadata: Source information
- embedding: Vector representation
```

### VisDrone Multimodal Dataset:
```
Dataset: hub://activeloop/visdrone-det-train
Tensors:
- images: (6471, 360:1500, 480:2000, 3) uint8, JPEG format
- boxes: (6471, 1:914, 4) float32, bounding box coordinates [x, y, w, h]
- labels: (6471, 1:914) uint32, class labels

Labels: 12 classes
['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 
 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
```

---

## 5. Implementation: LLM Module

### Load Dataset:
```python
import deeplake

dataset_path_llm = "hub://user/drone_v2"
ds_llm = deeplake.load(dataset_path_llm)

# Convert to DataFrame
data_llm = {}
for tensor_name in ds_llm.tensors:
    tensor_data = ds_llm[tensor_name].numpy()
    if tensor_data.ndim > 1:
        data_llm[tensor_name] = [np.array(e).flatten().tolist() for e in tensor_data]
    else:
        if tensor_name == "text":
            data_llm[tensor_name] = [t.tobytes().decode('utf-8') if isinstance(t, bytes) else t for t in tensor_data]
        else:
            data_llm[tensor_name] = tensor_data.tolist()

df_llm = pd.DataFrame(data_llm)
```

### Initialize Query Engine:
```python
from llama_index.core import VectorStoreIndex

# Create vector index
vector_store_index_llm = VectorStoreIndex.from_documents(documents_llm)

# Set as query engine
vector_query_engine_llm = vector_store_index_llm.as_query_engine(
    similarity_top_k=3,
    temperature=0.1,
    num_output=1024
)
```

### Query:
```python
import time

user_input = "How do drones identify a truck?"

start_time = time.time()
llm_response = vector_query_engine_llm.query(user_input)
elapsed_time = time.time() - start_time

print(f"Query execution time: {elapsed_time:.4f} seconds")
print(llm_response)
```

**Example Output**:
```
Query execution time: 1.5489 seconds
Drones can identify a truck using visual detection and tracking methods.
```

---

## 6. Implementation: Multimodal Module

### Load VisDrone Dataset:
```python
dataset_path = 'hub://activeloop/visdrone-det-train'
ds = deeplake.load(dataset_path)

# View structure
ds.summary()

# Visualize
ds.visualize()
```

### Create DataFrame:
```python
import pandas as pd

df = pd.DataFrame(columns=['image', 'boxes', 'labels'])

for i, sample in enumerate(ds):
    df.loc[i, 'image'] = sample.images.tobytes()  # JPEG bytes
    boxes_list = sample.boxes.numpy(aslist=True)
    df.loc[i, 'boxes'] = [box.tolist() for box in boxes_list]
    label_data = sample.labels.data()
    df.loc[i, 'labels'] = label_data['text']
```

---

## 7. Image Processing

### Display Image:
```python
from PIL import Image
import cv2

ind = 0  # Select first image
image = ds.images[ind].numpy()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = Image.fromarray(image_rgb)
display(img)
```

### Add Bounding Boxes:
```python
import io
from PIL import ImageDraw

def display_image_with_bboxes(image_data, bboxes, labels, label_name):
    image_bytes = io.BytesIO(image_data)
    img = Image.open(image_bytes)
    
    # Get class names
    class_names = ds.labels[ind].info['class_names']
    
    # Filter for specific label
    try:
        label_index = class_names.index(label_name)
        relevant_indices = np.where(labels == label_index)[0]
    except ValueError:
        relevant_indices = range(len(labels))
    
    # Draw boxes
    draw = ImageDraw.Draw(img)
    for idx, box in enumerate(bboxes):
        if idx in relevant_indices:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), label_name, fill="red")
    
    # Save and display
    img.save("boxed_image.jpg")
    display(img)

# Usage
labels = ds.labels[ind].data()['value']
image_data = ds.images[ind].tobytes()
bboxes = ds.boxes[ind].numpy()
display_image_with_bboxes(image_data, bboxes, labels, "truck")
```

---

## 8. Multimodal Query Engine

### Create Vector Index:
```python
from llama_index.core import Document, GPTVectorStoreIndex

# Add unique IDs
df['doc_id'] = df.index.astype(str)

# Create documents from labels
documents = []
for _, row in df.iterrows():
    text_labels = row['labels']
    text = " ".join(text_labels)
    document = Document(text=text, doc_id=row['doc_id'])
    documents.append(document)

# Create index
vector_store_index = GPTVectorStoreIndex.from_documents(documents)

# Set as query engine
vector_query_engine = vector_store_index.as_query_engine(
    similarity_top_k=3,
    temperature=0.1,
    num_output=1024
)
```

### Query Multimodal Dataset:
```python
start_time = time.time()
response = vector_query_engine.query(user_input)
elapsed_time = time.time() - start_time

print(f"Query execution time: {elapsed_time:.4f} seconds")
print(response)
```

**Example Output**:
```
Query execution time: 1.8461 seconds
Drones use various sensors such as cameras, LiDAR, and GPS to identify and track objects like trucks.
```

---

## 9. Response Processing

### Extract Unique Keywords:
```python
from itertools import groupby

def get_unique_words(text):
    text = text.lower().strip()
    words = text.split()
    unique_words = [word for word, _ in groupby(sorted(words))]
    return unique_words

for node in response.source_nodes:
    print(node.node_id)
    node_text = node.get_text()
    unique_words = get_unique_words(node_text)
    print("Unique Words:", unique_words)
```

**Example Output**:
```
1af106df-c5a6-4f48-ac17-f953dffd2402
Unique Words: ['truck']
```

### Retrieve Source Image:
```python
def process_and_display(response, df, ds, unique_words):
    for node in response.source_nodes:
        doc_id = node.node.ref_doc_id
        row_index = int(doc_id)
        
        for i, sample in enumerate(ds):
            if i == row_index:
                image_data = ds.images[i].tobytes()
                labels = ds.labels[i].data()['value']
                bboxes = ds.boxes[i].numpy()
                ibox = unique_words[0]
                display_image_with_bboxes(image_data, bboxes, labels, ibox)
                break

process_and_display(response, df, ds, unique_words)
```

---

## 10. Multimodal Summary Output

### Display Combined Response:
```python
def display_source_image(image_path):
    img = Image.open(image_path)
    display(img)

# 1. User input
print(user_input)

# 2. LLM response
print(textwrap.fill(str(llm_response), 100))

# 3. Multimodal response (image)
image_path = "/content/boxed_image.jpg"
display_source_image(image_path)
```

**Output**:
```
How do drones identify a truck?
Drones can identify a truck using visual detection and tracking methods.
[Image displayed with truck bounding boxes]
```

---

## 11. Performance Metrics

### LLM Performance (Cosine Similarity):
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity_with_embeddings(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    return cosine_similarity([embeddings1], [embeddings2])[0][0]

llm_similarity_score = calculate_cosine_similarity_with_embeddings(
    user_input, 
    str(llm_response)
)
print(f"LLM Cosine Similarity: {llm_similarity_score:.3f}")
```

**Example Output**: `0.691`

---

## 12. Multimodal Performance (GPT-4o Vision)

### Encode Image:
```python
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("/content/boxed_image.jpg")
```

### Analyze with GPT-4o:
```python
from openai import OpenAI

client = OpenAI(api_key=openai.api_key)
u_word = unique_words[0]  # e.g., "truck"

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": f"You are a helpful assistant analyzing images for {u_word}."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Analyze the image and describe all {u_word}s."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }}
        ]}
    ],
    temperature=0.0
)

response_image = response.choices[0].message.content
print(response_image)
```

**Example Output**:
```
The image contains two trucks within the bounding boxes:
1. First Truck: Flatbed truck loaded with construction materials
2. Second Truck: Flatbed truck carrying similar materials
Both trucks are in a construction/industrial area.
```

### Calculate Similarity:
```python
resp = u_word + "s"  # "trucks"
multimodal_similarity_score = calculate_cosine_similarity_with_embeddings(
    resp, 
    response_image
)
print(f"Multimodal Cosine Similarity: {multimodal_similarity_score:.3f}")
```

**Example Output**: `0.505`

---

## 13. Combined Performance Metric

### Overall Score:
```python
score = (llm_similarity_score + multimodal_similarity_score) / 2
print(f"Multimodal Modular Score: {score:.3f}")
```

**Example Output**: `0.598`

**Interpretation**:
- Human observers may find results satisfactory
- Automated image evaluation remains challenging
- Complex images difficult to assess with precision
- Explains why top AI agents (ChatGPT, Gemini) use human feedback (thumbs up/down)

---

## 14. Key Technical Insights

### Seamless Integration:
- LlamaIndex handles text and multimodal indexing
- Deep Lake manages both text and image datasets
- OpenAI provides LLM and vision capabilities
- Single query interface for multiple data types

### Modular Design Benefits:
1. **Specialization**: Each module optimized for data type
2. **Scalability**: Add new modules without disrupting existing ones
3. **Flexibility**: Mix and match data sources
4. **Transparency**: Trace responses to exact sources

### Image Recognition Challenges:
- Complex scenes with multiple objects
- Bounding box accuracy
- Label precision
- Automated evaluation difficulty
- Need for human feedback loops

---

## 15. Workflow Summary

**Step 1**: Define baseline user query
**Step 2**: Query LLM textual dataset → Text response
**Step 3**: Load multimodal VisDrone dataset
**Step 4**: Index multimodal data in memory
**Step 5**: Query multimodal dataset → Text response + keywords
**Step 6**: Parse response for unique keywords
**Step 7**: Trace to source image via node IDs
**Step 8**: Add bounding boxes to image
**Step 9**: Merge text + image responses
**Step 10**: Evaluate with cosine similarity (text) + GPT-4o vision (image)

---

## 16. Best Practices

1. **Baseline Queries**: Use consistent input for both modules
2. **Unique IDs**: Ensure traceability from response to source
3. **Automated Chunking**: Let LlamaIndex optimize
4. **Bounding Boxes**: Visual confirmation of object detection
5. **GPT-4o Vision**: Leverage for image analysis
6. **Human Feedback**: Essential for complex image evaluation
7. **Modular Architecture**: Separate concerns for maintainability
8. **Performance Metrics**: Combine text and image scores
9. **Base64 Encoding**: Simplify image transmission
10. **Error Handling**: Graceful degradation when labels not found

---

## 17. Key Takeaways

1. **Multimodal RAG combines text and images** for richer responses
2. **Modular design** enables specialized processing per data type
3. **LlamaIndex + Deep Lake + OpenAI** = seamless multimodal integration
4. **VisDrone dataset** provides real-world drone imagery with labels
5. **Bounding boxes** enable precise object identification
6. **GPT-4o vision** analyzes images for automated evaluation
7. **Traceability** from response to source image via node IDs
8. **Image evaluation challenging** compared to text
9. **Human feedback essential** for complex multimodal systems
10. **Drone technology** demonstrates practical multimodal applications

---

## 18. Formulas & Concepts

### Cosine Similarity (Text):
```
similarity = (A · B) / (||A|| × ||B||)
Range: [-1, 1]
```

### Multimodal Score:
```
score = (LLM_similarity + Multimodal_similarity) / 2
```

### Bounding Box Format:
```
[x, y, width, height]
x, y: Top-left corner coordinates
width, height: Box dimensions
```

### Base64 Encoding:
```
Binary data → ASCII characters
Enables text-based transmission (HTTP)
Avoids data corruption
```

---

## Interview-Ready Q&A

**Q: What is multimodal modular RAG?**
System using distinct modules for different data types (text, images) with specialized processing for each, retrieving from multiple sources.

**Q: Why use modular architecture?**
Specialization, scalability, flexibility, transparency—each module optimized independently without disrupting others.

**Q: How to evaluate multimodal responses?**
Text: Cosine similarity. Images: GPT-4o vision analysis + cosine similarity on description. Combined: Average both scores.

**Q: What's in VisDrone dataset?**
6,471 drone images with bounding boxes and 12 class labels (pedestrian, car, truck, bicycle, etc.).

**Q: How to trace response to source image?**
Parse response for keywords → Find node ID → Match to DataFrame doc_id → Retrieve image from dataset.

**Q: Why is image evaluation challenging?**
Complex scenes, multiple objects, precision requirements, automated assessment difficulty—requires human feedback.

**Q: How does GPT-4o help with images?**
Multimodal model analyzes images, describes objects, enables text-based evaluation of visual content.

**Q: Bounding box purpose?**
Visual confirmation of object detection, precise localization, label verification.

---

## Tools & Technologies

**Frameworks**: LlamaIndex, Deep Lake, OpenAI
**Models**: GPT-4o (multimodal), all-MiniLM-L6-v2 (embeddings)
**Datasets**: Custom LLM dataset (Chapter 3), VisDrone (Activeloop)
**Image Processing**: PIL, OpenCV, Base64 encoding
**Evaluation**: Sentence Transformers, scikit-learn, GPT-4o vision
**Data Structures**: Pandas DataFrames, NumPy arrays


---

## Yes/No Questions with Answers

**Q1: Does multimodal modular RAG handle different types of data, such as text and images?**
Yes, it processes multiple data types such as text and images.

**Q2: Are drones used solely for agricultural monitoring and aerial photography?**
No, drones are also used for rescue, traffic, and infrastructure inspections.

**Q3: Is the Deep Lake VisDrone dataset used in this chapter for textual data only?**
No, it contains labeled drone images, not just text.

**Q4: Can bounding boxes be added to drone images to identify objects such as trucks and pedestrians?**
Yes, bounding boxes are used to mark objects within images.

**Q5: Does the modular system retrieve both text and image data for query responses?**
Yes, it retrieves and generates responses from both textual and image datasets.

**Q6: Is building a vector index necessary for querying the multimodal VisDrone dataset?**
Yes, a vector index is created for efficient multimodal data retrieval.

**Q7: Are the retrieved images processed without adding any labels or bounding boxes?**
No, images are processed with labels and bounding boxes.

**Q8: Is the multimodal modular RAG performance metric based only on textual responses?**
No, it also evaluates the accuracy of image analysis.

**Q9: Can a multimodal system such as the one described in this chapter handle only drone-related data?**
No, it can be adapted for other industries and domains.

**Q10: Is evaluating images as easy as evaluating text in multimodal RAG?**
No, image evaluation is more complex and requires specialized metrics.
