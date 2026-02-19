# Chapter 10: RAG for Video Stock Production with Pinecone and OpenAI

## 1. Core Concept: AI-Driven Video Stock Production

**Human Creativity vs Generative AI**:
- **Human**: Breaks habits, invents new ways, goes beyond well-known patterns
- **Generative AI**: Replicates established patterns, automates tasks efficiently, doesn't truly "create" but "generates"

**Chapter Goal**: Build AI-driven downloadable stock of online videos with team of AI agents working together to create commented and labeled videos.

**Use Case**: Start-up building automated video stock production system from raw footage to videos with descriptions and labels.

**Key Principle**: "Create" = "Generate" (practical term for AI automation, not true creativity)

## 2. Three-Pipeline Architecture

**Pipeline 1: The Generator and the Commentator**
- **Generator**: Creates world simulations using OpenAI Sora (text-to-video model)
- **inVideo AI**: Powered by Sora, transforms ideas into videos ("ideation")
- **Commentator**: Splits AI-generated videos into frames, generates technical comments with OpenAI vision model

**Pipeline 2: The Vector Store Administrator**
- Manages Pinecone vector store
- Embeds technical video comments from Commentator
- Upserts vectorized comments to Pinecone
- Queries vector store to verify system functionality

**Pipeline 3: The Video Expert**
- Processes user inputs
- Queries vector store, retrieves relevant video frames
- Augments user inputs with raw query output
- Activates OpenAI GPT-4o to analyze comments, detect imperfections, reformulate efficiently, provide labels

## 3. Complete Workflow: Raw to Labeled Videos

**Automated Process Steps**:

1. **Generate raw videos** automatically and download
2. **Split videos into frames** (images)
3. **Analyze sample of frames** with vision model
4. **Activate OpenAI LLM** to generate technical comments
5. **Save technical comments** with unique index, comment, frame number, video file name
6. **Upsert data** in Pinecone index vector store
7. **Query Pinecone** with user inputs
8. **Retrieve specific frame** most similar to technical comment
9. **Augment user input** with technical comment of retrieved frame
10. **Ask OpenAI LLM** to analyze logic, detect contradictions/imperfections, produce dynamic tailored description
11. **Display selected video** with frame number and file name
12. **Evaluate outputs** and apply metric calculations

**Example Query**: "Find a basketball player that is scoring with a dunk" → System finds frame within unlabeled video, selects video, displays it, generates tailored comment dynamically.

## 4. Industry Applications

**Relevant Industries**:
- **Media**: Automated content production and management
- **Marketing**: Scalable video content creation
- **Entertainment**: Efficient video generation and labeling
- **Education**: Automated educational video production
- **Firefighting**: Emergency response video analysis
- **Medical Imagery**: Diagnostic video processing

**Business Value**: Businesses and creators continuously seek efficient ways to produce and manage content that scales with growing demand.

## 5. Sora: Text-to-Video Diffusion Transformer

**Model Overview** (Liu et al., 2024):
- **Released**: February 2024 by OpenAI
- **Type**: Text-to-video diffusion transformer model
- **Access**: https://ai.invideo.io/ (inVideo AI application)
- **Copyright**: Flexible terms for free videos (https://invideo.io/terms-and-conditions/)

**Core Architecture**: Diffusion transformer operates between encoder and decoder

**Five-Step Process**:

1. **Visual Encoder**: Transforms image datasets into lower-dimensional latent space
2. **Patch Splitting**: Encoder splits latent space into patches (like words in sentence)
3. **Text-Patch Association**: Diffusion transformer associates user text input with patch dictionary
4. **Iterative Refinement**: Transformer refines noisy image representations to produce coherent frames
5. **Visual Decoder**: Reconstructs refined latent representations into high-fidelity video frames aligned with user instructions

**Key Components**:
- **Encoder**: Compresses input data (images/videos) into lower-dimensional latent space, preserving crucial information
- **Lower-Dimensional Latent Space**: Compressed representation of high-dimensional data (e.g., 1024x1024x3 image → 1000-value vector)
- **Decoder**: Reconstructs original data from latent representation, transforms low-dimensional back to high-dimensional pixel space
- **Technology Stack**: Vision transformers (CLIP), LLMs (GPT-4), other OpenAI components

**Advantages**:
- Mainstream video generation in few clicks
- Fast video dataset creation without manual filming
- Teams don't spend time finding videos that fit needs
- Quick video generation from prompt (idea in few words)

**Risks & Ethics**:
- Job displacement in filmmaking and related areas
- Deep fakes and misinformation potential
- Ethical considerations mandatory for constructive, realistic content

## 6. Environment Setup: Four Notebooks

**GitHub Directory**: Chapter10 with four notebooks:
1. **Videos_dataset_visualization.ipynb**: Dataset exploration
2. **Pipeline_1_The_Generator_and_the_Commentator.ipynb**: Video generation and commenting
3. **Pipeline_2_The_Vector_Store_Administrator.ipynb**: Vector store management
4. **Pipeline_3_The_Video_Expert.ipynb**: Expert analysis and labeling

**Common Environment Sections** (identical across all notebooks):
- Importing modules and libraries
- GitHub download functions
- Video download and display functions
- OpenAI setup
- Pinecone configuration

**Resource Requirements**:
- **CPU only** (no GPU required)
- **Limited memory**
- **Limited disk space**
- **Scalable**: Process one video at a time indefinitely

## 7. Core Modules and Libraries

| Code | Comment |
|------|---------|
| `from IPython.display import HTML` | Display videos |
| `import base64` | Encode videos as base64 |
| `from base64 import b64encode` | Encode videos as base64 |
| `import os` | Interact with operating system |
| `import subprocess` | Run commands |
| `import time` | Measure execution time |
| `import csv` | Save comments |
| `import uuid` | Generate unique IDs |
| `import cv2` | Split videos (OpenCV) |
| `from PIL import Image` | Display videos |
| `import pandas as pd` | Display comments |
| `import numpy as np` | Numerical Python |
| `from io import BytesIO` | Binary stream in memory |

**GitHub Download Function**:
```python
def download(directory, filename):
    base_url = 'https://raw.githubusercontent.com/Denis2054/RAG-Driven-Generative-AI/main/'
    file_url = f"{base_url}{directory}/{filename}"
    curl_command = f'curl -o {filename} {file_url}'
    subprocess.run(curl_command, check=True, shell=True)
```

**OpenAI Setup**:
```python
from google.colab import drive
drive.mount('/content/drive')
f = open("drive/MyDrive/files/api_key.txt", "r")
API_KEY = f.readline()
f.close()

!pip install openai==1.45.0
import openai
os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
```

**Pinecone Setup** (Pipeline 2 & 3 only):
```python
!pip install pinecone-client==4.1.1
import pinecone
f = open("drive/MyDrive/files/pinecone.txt", "r")
PINECONE_API_KEY = f.readline()
f.close()
```

## 8. Pipeline 1: Generator and Commentator

**Generator Tasks**:
1. Generate text-to-video inVideo dataset based on team's text input (sports videos)
2. Run scaled process selecting one video at a time
3. Split video into frames (images)

**Commentator Tasks**:
4. Sample frames and comment with OpenAI LLM
5. Save each commented frame with:
   - **Unique ID**: UUID
   - **Comment**: Technical description
   - **Frame**: Frame number
   - **Video file name**: Source video

**Video Display Function**:
```python
def display_video(file_name):
    with open(file_name, 'rb') as file:
        video_data = file.read()
    video_url = b64encode(video_data).decode()
    html = f'''
    <video width="640" height="480" controls>
      <source src="data:video/mp4;base64,{video_url}" type="video/mp4">
    </video>
    '''
    return HTML(html)
```

**Frame Display Function**:
```python
def display_video_frame(file_name, frame_number, size):
    cap = cv2.VideoCapture(file_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
    img = Image.fromarray(frame)
    img = img.resize(size, Image.LANCZOS)
    
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    html_str = f'<img src="data:image/jpeg;base64,{img_str}" width="{size[0]}" height="{size[1]}">'
    return HTML(html_str)
```

**Split Video into Frames**:
```python
def split_file(file_name):
    video_path = file_name
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"frame_{frame_number}.jpg", frame)
        frame_number += 1
        print(f"Frame {frame_number} saved.")
    
    cap.release()
```

**Generate Comments with OpenAI Vision**:
```python
def generate_openai_comments(filename):
    video_folder = "/content"
    total_frames = len([file for file in os.listdir(video_folder) if file.startswith('frame_')])
    
    nb = 3  # Sample frequency
    counter = 0
    
    for frame_number in range(total_frames):
        counter += 1
        if counter == nb and counter < total_frames:
            counter = 0
            image_path = os.path.join(video_folder, f"frame_{frame_number}.jpg")
            
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is happening in this image?"},
                            {"type": "image", "image_url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode()}"}
                        ]
                    }],
                    max_tokens=150
                )
            
            comment = generate_comment(response)
            save_comment(comment, frame_number, file_name)
```

**Extract Comment**:
```python
def generate_comment(response_data):
    try:
        caption = response_data.choices[0].message.content
        return caption
    except (KeyError, AttributeError):
        return "No caption available."
```

**Save Comment to CSV**:
```python
def save_comment(comment, frame_number, file_name):
    path = f"{file_name}.csv"
    write_header = not os.path.exists(path)
    
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writerow(['ID', 'FrameNumber', 'Comment', 'FileName'])
        
        unique_id = str(uuid.uuid4())
        writer.writerow([unique_id, frame_number, comment, file_name])
```

**CSV Format** (for Pinecone upsert):
- **ID**: Unique UUID string
- **FrameNumber**: Frame number of commented JPEG
- **Comment**: OpenAI vision model description
- **FileName**: Source video file name

## 9. Pipeline 1 Controller

**Controller Workflow**:

**Step 1: Collect, Download, Display Video**
```python
session_time = time.time()
file_name = "skiing1.mp4"
directory = "Chapter10/videos"
download(directory, file_name)
display_video(file_name)
```

**Step 2: Split Video into Frames**
```python
split_file(file_name)
```

**Step 3: Comment on Frames**
```python
start_time = time.time()
generate_openai_comments(file_name)
response_time = time.time() - session_time
```

**Display Results**:
```python
video_folder = "/content"
total_frames = len([file for file in os.listdir(video_folder) if file.endswith('.jpg')])
print(total_frames)

display_comments(file_name)
total_time = time.time() - start_time
print(f"Response Time: {response_time:.2f} seconds")
print(f"Total Time: {total_time:.2f} seconds")
```

**Save Comments and Frames**:
```python
save = True  # Save comments
save_frames = True  # Save frames

if save == True:
    cpath = f"{file_name}.csv"
    !cp {cpath} /content/drive/MyDrive/files/comments/{cpath}

if save_frames == True:
    root_name, extension = os.path.splitext(file_name)
    root_name = root_name + extension.strip('.')
    target_directory = f'/content/drive/MyDrive/files/comments/{root_name}'
    os.makedirs(target_directory, exist_ok=True)
    
    for file in os.listdir(os.getcwd()):
        if file.endswith('.jpg'):
            shutil.copy(os.path.join(os.getcwd(), file), target_directory)
```

**Delete Files** (for loop processing):
```python
delf = False
if delf == True:
    !rm -f *.mp4  # Video files
    !rm -f *.jpg  # Frames
    !rm -f *.csv  # Comments
```

**Scalability**: Process unlimited videos one by one with constant disk space.

## 10. Pipeline 2: Vector Store Administrator

**Four-Step Workflow**:

1. **Processing Video Comments**: Load and prepare comments for chunking (Chapter 6 approach)
2. **Chunking and Embedding**: Dataset columns ('ID', 'FrameNumber', 'Comment', 'FileName') prepared by Commentator, chunk and embed
3. **Pinecone Index**: Create index and upsert data (Chapter 6 approach)
4. **Querying Vector Store**: Hybrid retrieval using Pinecone + separate file system for videos/frames

**Hybrid Approach**: Query Pinecone for comments + retrieve media files from separate storage (GitHub, cloud storage, etc.)

**Storage Decision**: Store images in vector store OR separate location depends on project needs and cost-effectiveness.

## 11. Querying Pinecone Index

**Query Setup**:
```python
k = 1  # Number of top-k results
query_text = "Find a basketball player that is scoring with a dunk"
```

**Embed Query**:
```python
start_time = time.time()
query_embedding = get_embedding(query_text, model=embedding_model)
```

**Similarity Search**:
```python
query_results = index.query(vector=query_embedding, top_k=k, include_metadata=True)

for match in query_results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
    
    if 'metadata' in match:
        metadata = match['metadata']
        text = metadata.get('text', "No text metadata available.")
        frame_number = metadata.get('frame_number', "No frame number available.")
        file_name = metadata.get('file_name', "No file name available.")
        
        print(f"Text: {text}")
        print(f"Frame Number: {frame_number}")
        print(f"File Name: {file_name}")

response_time = time.time() - start_time
print(f"Querying response time: {response_time:.2f} seconds")
```

**Example Output**:
```
ID: f104138b-0be8-4f4c-bf99-86d0eb34f7ee, Score: 0.866656184
Text: In this image, there is a person who appears to be in the process of performing a slam dunk...
Frame Number: 191
File Name: basketball3.mp4
Querying response time: 0.57 seconds
```

**Display Video**:
```python
directory = "Chapter10/videos"
download(directory, file_name)
display_video(file_name)
```

**Display Specific Frame**:
```python
file_name_root = file_name.split('.')[0]
frame = f"frame_{frame_number}.jpg"
file_path = os.path.join('/content/', frame)

if os.path.exists(file_path):
    file_size = os.path.getsize(file_path)
    if file_size > 1000:  # Logical size threshold
        display(Image(filename=file_path))
```

**Result**: Exact frame corresponding to user input displayed.

## 12. Pipeline 3: Video Expert with GPT-4o

**Video Expert Role**:
- Analyze Commentator's technical comments
- Point out cognitive dissonances (contradictions, discrepancies)
- Rewrite comments in logical, engaging style
- Provide labels for video classification

**Workflow**:
1. **Connect to Pinecone Index** (no upserting, only querying)
2. **Define RAG Functions** (from Pipeline 1 & 2)
3. **Query Vector Store** (Pinecone query)
4. **Retrieval Augmented Generation**: GPT-4o analyzes and improves responses

**GPT-4o System Instructions**:
```python
{
    "role": "system",
    "content": "You will be provided with comments of an image frame taken from a video. 1. Point out the cognitive dissonances. 2. Rewrite the comment in a logical engaging style. 3. Provide a label for this image such as Label: basketball, football, soccer or other label."
}
```

**Three-Part Task**:
1. **Point out cognitive dissonances**: Find contradictions/discrepancies in comment (from AI-generated video production or description)
2. **Rewrite comment**: Transform technical comment to engaging description
3. **Provide label**: Classify video (basketball, football, soccer, etc.)

**Example Query Result** (from Pipeline 2):
```
ID: f104138b-0be8-4f4c-bf99-86d0eb34f7ee
Score: 0.866193652
Text: In this image, there is a person who appears to be in the process of performing a slam dunk...
Frame Number: 191
File Name: basketball3.mp4
```

**Video Expert Analysis**:
```python
prompt = text  # Use query result as prompt
response_content = get_openai_response(prompt)
print(response_content)
```

**GPT-4o Output**:
```
1. Cognitive Dissonances:
   - The comment redundantly describes the action of dunking multiple times
   - The mention of "the word 'dunk' is superimposed on the image" is unclear
   - The background details about clear skies and modern building may be irrelevant

2. Rewritten Comment:
   In this image, a basketball player is captured mid-air, executing a powerful slam dunk...

3. Label: Basketball
```

**Output Variability**: May vary run-to-run due to stochastic "creative" nature of Generative AI.

## 13. Evaluation and Metrics

**Evaluator Section**: Runs 10 examples using same process as basketball request.

**Each Example Contains**:
- User prompt
- Comment returned by vector store query
- Enhanced comment by GPT-4o model
- Human evaluator suggested content (ground truth)

**Evaluation Process** (Chapter 7 approach):

**Human Feedback (Ground Truth)**:
```python
text1 = "This image shows soccer players on a field dribbling and passing the ball..."
```

**Extract Rewritten Comment**:
```python
text2 = extract_rewritten_comment(response_content)
```

**Display Comments**:
```python
print(f"Human Feedback Comment: {text1}")
print(f"Rewritten Comment: {text2}")
```

**Calculate Cosine Similarity**:
```python
similarity_score3 = calculate_cosine_similarity_with_embeddings(text1, text2)
print(f"Cosine Similarity Score with sentence transformer: {similarity_score3}")
scores.append(similarity_score3)
```

**Track Original Query Score**:
```python
rscores.append(score)
```

**Example Output**:
```
Human Feedback Comment: This image shows soccer players on a field dribbling and passing the ball...
Rewritten Comment: "A group of people are engaged in a casual game of soccer..."
Cosine Similarity Score with sentence transformer: 0.621
```

**Metrics Calculations** (Chapter 7 metrics):
- Mean, Median, Standard Deviation, Variance
- Min, Max, Range
- Percentiles (25th, 50th, 75th)
- IQR (Interquartile Range)

**Accuracy Result**:
```
Mean: 0.65
```

**Interpretation**: Room for progress, challenging requests require further work.

## 14. System Improvement Strategies

**Quality Enhancement**:
1. **Check video quality and content**: Verify AI-generated videos for coherence
2. **Check comments**: Modify with human feedback (Chapter 5 approach)
3. **Fine-tune model**: Use images and text (Chapter 9 approach)
4. **Cosine similarity quality control**: Apply metrics from Chapter 5
5. **Design constructive ideas**: Video production team brainstorming

**Production Requirements**:
- RAG-driven Generative AI systems effective in production
- Road from design to production requires hard human effort
- AI technology requires humans to design, develop, implement

**Key Insight**: Despite tremendous AI progress, human oversight mandatory for production systems.

## 15. Scalability and Optimization

**Resource Efficiency**:
- **CPU only**: No GPU required
- **Limited memory**: Minimal RAM usage
- **Limited disk space**: Process one video at a time
- **Constant disk space**: Delete processed files in loop

**Indefinite Scaling**: Process unlimited videos one by one, leveraging Pinecone's storage capacity.

**Pre-Production Policy**: Common installation across all notebooks, focus on pipeline content once environment ready.

**Production Considerations**:
- Hard work required for real-life implementation
- Technology available, video production undergoing historical evolution
- Automated video production concept effectively demonstrated

**Time Measurements**: Encapsulated in key functions throughout ecosystem for performance monitoring.

## Interview-Ready Q&A

**Q1: What is the difference between human creativity and Generative AI?**
A: Humans break habits and invent new ways; Generative AI replicates established patterns and automates tasks without truly "creating."

**Q2: What are the three pipelines in the video production system?**
A: (1) Generator and Commentator (video generation, frame splitting, commenting), (2) Vector Store Administrator (embedding, upserting, querying), (3) Video Expert (analysis, improvement, labeling).

**Q3: What is OpenAI Sora?**
A: Text-to-video diffusion transformer model released February 2024, creates world simulations from text prompts, accessible via inVideo AI.

**Q4: How does a diffusion transformer work?**
A: Five steps: (1) Visual encoder transforms images to lower-dimensional latent space, (2) Splits into patches, (3) Associates text with patches, (4) Iteratively refines noisy representations, (5) Decoder reconstructs high-fidelity video frames.

**Q5: What are the resource requirements for this system?**
A: CPU only (no GPU), limited memory, limited disk space; scalable by processing one video at a time indefinitely.

**Q6: What does the Generator agent do?**
A: Generates text-to-video content, runs scaled process selecting one video at a time, splits videos into frames.

**Q7: What does the Commentator agent do?**
A: Samples frames, comments with OpenAI vision model ("What is happening in this image?"), saves comments with UUID, frame number, comment, file name.

**Q8: What is the CSV format for comments?**
A: Four columns: ID (UUID), FrameNumber (frame number), Comment (technical description), FileName (source video).

**Q9: What is hybrid retrieval in Pipeline 2?**
A: Query Pinecone for comments + retrieve media files (videos, frames) from separate storage location (GitHub, cloud, etc.).

**Q10: What is the Video Expert's role?**
A: Analyze Commentator's comments, point out cognitive dissonances (contradictions), rewrite in logical engaging style, provide classification labels.

**Q11: What are GPT-4o's three tasks in Pipeline 3?**
A: (1) Point out cognitive dissonances, (2) Rewrite comment logically, (3) Provide label (basketball, football, soccer, etc.).

**Q12: How is evaluation performed?**
A: 10 examples with user prompt, query result, GPT-4o enhanced comment, human ground truth; cosine similarity calculated between human and LLM comments.

**Q13: What was the accuracy result?**
A: Mean: 0.65, indicating room for progress with challenging requests requiring further work.

**Q14: What are system improvement strategies?**
A: Check video quality, modify comments with human feedback (Chapter 5), fine-tune models (Chapter 9), apply quality control metrics, team brainstorming.

**Q15: What industries benefit from this system?**
A: Media, marketing, entertainment, education, firefighting, medical imagery - any domain requiring automated video production and management.

## Tools & Technologies

**Video Generation**: OpenAI Sora (text-to-video diffusion transformer), inVideo AI

**LLM Models**: GPT-4o (Video Expert), gpt-4-vision-preview (Commentator)

**Vector Database**: Pinecone (embedding, upserting, querying)

**Computer Vision**: OpenCV (cv2) for video splitting, PIL (Image) for display

**Data Processing**: pandas (DataFrames), numpy (numerical operations), csv (comment storage)

**Encoding**: base64, b64encode (video/image encoding for HTML display)

**Unique IDs**: uuid (UUID generation for comments)

**Environment**: Google Colab (CPU only, no GPU required)

**Storage**: Google Drive (API keys, comments, frames), GitHub (video dataset)

**Display**: IPython.display (HTML video/image display)

**Utilities**: os (file operations), subprocess (GitHub downloads), time (performance measurement), BytesIO (binary streams)

**Evaluation**: Cosine similarity (sentence transformers), metrics from Chapter 7

**Notebooks**: 4 notebooks (visualization, Pipeline 1, Pipeline 2, Pipeline 3)

**Dataset**: AI-generated sports videos (jogging, skiing, basketball, football, hockey)

**Scalability**: One video at a time processing, indefinite scaling with constant disk space

**Hybrid Retrieval**: Pinecone (comments) + separate file system (videos/frames)


---

## Yes/No Questions with Answers

**Q1: Can AI now automatically comment and label videos?**
Yes, we now create video stocks automatically to a certain extent.

**Q2: Does video processing involve splitting a video into frames?**
Yes, we can split a video into frames before analyzing the frames.

**Q3: Can the programs in this chapter create a 200-minute movie?**
No, for the moment, this cannot be done directly. We would have to create many videos and then stitch them together with a video editor.

**Q4: Do the programs in this chapter require a GPU?**
No, only a CPU is required, which is cost-effective because the processing times are reasonable, and the programs mostly rely on API calls.

**Q5: Are the embedded vectors of the video content stored on disk?**
No, the embedded vectors are upserted in a Pinecone vector database.

**Q6: Do the scripts involve querying a database for retrieving data?**
Yes, the scripts query the Pinecone vector database for data retrieval.

**Q7: Is there functionality for displaying images in the scripts?**
Yes, the programs include code to display images after downloading them.

**Q8: Is it useful to have functions specifically checking file existence and size in any of the scripts?**
Yes, this avoids trying to display files that don't exist or that are empty.

**Q9: Is there a focus on multimodal data in these scripts?**
Yes, all scripts focus on handling and processing multimodal data (text, image, and video).

**Q10: Do any of the scripts mention applications of AI in real-world scenarios?**
Yes, these scripts deal with multimodal data retrieval and processing, which makes them applicable in AI-driven content management, search, and retrieval systems.
