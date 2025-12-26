# Deployment and Serving Large Language Models

## 🎯 Overview

Deploying and serving Large Language Models in production requires careful consideration of performance, scalability, cost, and reliability. This involves optimizing models for inference, managing resources efficiently, and monitoring system health.

## 📦 Model Serialization

### Core Concepts

**Model Serialization**: Converting trained models into formats suitable for deployment and inference.

**Format Considerations**:
- **Storage efficiency**: Compressed representations
- **Loading speed**: Fast deserialization
- **Compatibility**: Cross-platform support
- **Version control**: Model versioning and updates

### Common Serialization Formats

**1. PyTorch (.pt, .pth)**
```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

**2. ONNX (Open Neural Network Exchange)**
```python
# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)

# Load ONNX model
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
```

**3. TensorRT (NVIDIA)**
```python
# Convert to TensorRT
import tensorrt as trt

# Build TensorRT engine
builder = trt.Builder(logger)
network = builder.create_network()
# ... build network
engine = builder.build_cuda_engine(network)

# Save engine
with open("model.trt", "wb") as f:
    f.write(engine.serialize())
```

**4. Hugging Face Format**
```python
# Save model and tokenizer
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Load model and tokenizer
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
```

### Model Compression Techniques

**1. Quantization**
```python
# Post-training quantization
import torch.quantization as quant

# Prepare model for quantization
model.qconfig = quant.get_default_qconfig('fbgemm')
model_prepared = quant.prepare(model)

# Calibrate with sample data
with torch.no_grad():
    for data in calibration_loader:
        model_prepared(data)

# Convert to quantized model
model_quantized = quant.convert(model_prepared)
```

**2. Pruning**
```python
import torch.nn.utils.prune as prune

# Structured pruning
prune.ln_structured(
    model.linear_layer, 
    name="weight", 
    amount=0.3, 
    n=2, 
    dim=0
)

# Unstructured pruning
prune.l1_unstructured(
    model.linear_layer,
    name="weight",
    amount=0.2
)
```

**3. Knowledge Distillation**
```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        
    def forward(self, student_logits, teacher_logits, labels):
        # Distillation loss
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        distill_loss = self.kl_div(student_log_probs, teacher_probs)
        
        # Task loss
        task_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        return self.alpha * distill_loss + (1 - self.alpha) * task_loss
```

## ⚡ Inference Optimization

### Performance Optimization Strategies

**1. Model-Level Optimizations**

**Mixed Precision Inference**
```python
# Enable automatic mixed precision
model = model.half()  # Convert to FP16

# Or use autocast for dynamic precision
with torch.autocast(device_type='cuda'):
    outputs = model(inputs)
```

**Operator Fusion**
```python
# Fuse operations for better performance
model = torch.jit.script(model)  # TorchScript compilation
model = torch.jit.optimize_for_inference(model)
```

**2. Hardware-Specific Optimizations**

**CUDA Optimizations**
```python
# Optimize CUDA kernels
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Use CUDA graphs for repeated inference
def create_cuda_graph(model, input_shape):
    # Warmup
    for _ in range(10):
        with torch.cuda.stream(torch.cuda.Stream()):
            _ = model(torch.randn(input_shape, device='cuda'))
    
    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        output = model(torch.randn(input_shape, device='cuda'))
    
    return g, output
```

**CPU Optimizations**
```python
# Set optimal thread count
torch.set_num_threads(4)

# Use Intel MKL-DNN
torch.backends.mkldnn.enabled = True
```

**3. Memory Optimizations**

**Gradient Checkpointing**
```python
# Reduce memory usage during inference
from torch.utils.checkpoint import checkpoint

class OptimizedTransformerBlock(nn.Module):
    def forward(self, x):
        if self.training:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)
```

**KV Cache Management**
```python
class KVCache:
    def __init__(self, max_seq_len, num_heads, head_dim, dtype=torch.float16):
        self.max_seq_len = max_seq_len
        self.cache_k = torch.zeros(max_seq_len, num_heads, head_dim, dtype=dtype)
        self.cache_v = torch.zeros(max_seq_len, num_heads, head_dim, dtype=dtype)
        self.seq_len = 0
    
    def update(self, new_k, new_v):
        self.cache_k[self.seq_len] = new_k
        self.cache_v[self.seq_len] = new_v
        self.seq_len += 1
        return self.cache_k[:self.seq_len], self.cache_v[:self.seq_len]
    
    def reset(self):
        self.seq_len = 0
```

## 🔄 Batch Inference

### Batching Strategies

**1. Static Batching**
```python
class StaticBatchInference:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        
    def process_batch(self, inputs):
        """Process a fixed-size batch."""
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs
    
    def process_requests(self, requests):
        """Process multiple requests in batches."""
        results = []
        
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            
            # Pad batch if necessary
            if len(batch) < self.batch_size:
                padding_needed = self.batch_size - len(batch)
                batch.extend([batch[-1]] * padding_needed)
            
            batch_tensor = torch.stack(batch)
            batch_outputs = self.process_batch(batch_tensor)
            
            # Remove padding from results
            actual_outputs = batch_outputs[:len(requests[i:i + self.batch_size])]
            results.extend(actual_outputs)
        
        return results
```

**2. Dynamic Batching**
```python
import asyncio
from collections import deque

class DynamicBatchInference:
    def __init__(self, model, max_batch_size=32, max_wait_time=0.01):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = deque()
        self.processing = False
        
    async def add_request(self, input_data):
        """Add request to queue and return future for result."""
        future = asyncio.Future()
        self.request_queue.append((input_data, future))
        
        if not self.processing:
            asyncio.create_task(self._process_queue())
        
        return await future
    
    async def _process_queue(self):
        """Process queued requests in batches."""
        self.processing = True
        
        while self.request_queue:
            # Collect batch
            batch_inputs = []
            batch_futures = []
            
            # Wait for requests or timeout
            start_time = time.time()
            while (len(batch_inputs) < self.max_batch_size and 
                   self.request_queue and 
                   (time.time() - start_time) < self.max_wait_time):
                
                if self.request_queue:
                    input_data, future = self.request_queue.popleft()
                    batch_inputs.append(input_data)
                    batch_futures.append(future)
                else:
                    await asyncio.sleep(0.001)
            
            if batch_inputs:
                # Process batch
                batch_tensor = torch.stack(batch_inputs)
                with torch.no_grad():
                    batch_outputs = self.model(batch_tensor)
                
                # Return results
                for i, future in enumerate(batch_futures):
                    future.set_result(batch_outputs[i])
        
        self.processing = False
```

**3. Continuous Batching**
```python
class ContinuousBatchInference:
    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.active_sequences = {}
        self.kv_caches = {}
        
    def add_sequence(self, seq_id, initial_tokens):
        """Add new sequence to batch."""
        self.active_sequences[seq_id] = {
            'tokens': initial_tokens,
            'position': 0,
            'finished': False
        }
        self.kv_caches[seq_id] = KVCache(max_seq_len=2048, num_heads=32, head_dim=64)
    
    def generate_step(self):
        """Generate one token for all active sequences."""
        if not self.active_sequences:
            return
        
        # Prepare batch
        batch_tokens = []
        batch_positions = []
        seq_ids = []
        
        for seq_id, seq_data in self.active_sequences.items():
            if not seq_data['finished']:
                current_token = seq_data['tokens'][-1]
                batch_tokens.append(current_token)
                batch_positions.append(seq_data['position'])
                seq_ids.append(seq_id)
        
        if not batch_tokens:
            return
        
        # Run inference
        batch_input = torch.tensor(batch_tokens).unsqueeze(1)
        with torch.no_grad():
            outputs = self.model(batch_input)
            next_tokens = torch.argmax(outputs.logits, dim=-1)
        
        # Update sequences
        for i, seq_id in enumerate(seq_ids):
            next_token = next_tokens[i].item()
            self.active_sequences[seq_id]['tokens'].append(next_token)
            self.active_sequences[seq_id]['position'] += 1
            
            # Check for completion
            if next_token == self.model.config.eos_token_id:
                self.active_sequences[seq_id]['finished'] = True
    
    def remove_finished_sequences(self):
        """Remove completed sequences from batch."""
        finished_ids = [
            seq_id for seq_id, seq_data in self.active_sequences.items()
            if seq_data['finished']
        ]
        
        for seq_id in finished_ids:
            del self.active_sequences[seq_id]
            del self.kv_caches[seq_id]
```

## 🌊 Streaming Inference

### Real-Time Generation

**1. Token Streaming**
```python
class StreamingGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_stream(self, prompt, max_length=100, temperature=0.7):
        """Generate tokens one by one."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Yield decoded token
                token_text = self.tokenizer.decode(next_token, skip_special_tokens=True)
                yield token_text
                
                # Update input
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Check for end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
```

**2. WebSocket Streaming**
```python
import websockets
import json

class WebSocketStreamer:
    def __init__(self, model, tokenizer, host='localhost', port=8765):
        self.model = model
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        try:
            async for message in websocket:
                data = json.loads(message)
                prompt = data.get('prompt', '')
                
                # Stream generation
                generator = StreamingGenerator(self.model, self.tokenizer)
                
                for token in generator.generate_stream(prompt):
                    response = {
                        'type': 'token',
                        'content': token,
                        'finished': False
                    }
                    await websocket.send(json.dumps(response))
                
                # Send completion signal
                final_response = {
                    'type': 'completion',
                    'finished': True
                }
                await websocket.send(json.dumps(final_response))
                
        except websockets.exceptions.ConnectionClosed:
            pass
    
    def start_server(self):
        """Start WebSocket server."""
        return websockets.serve(self.handle_client, self.host, self.port)
```

**3. Server-Sent Events (SSE)**
```python
from flask import Flask, Response, request
import json

app = Flask(__name__)

class SSEStreamer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @app.route('/generate', methods=['POST'])
    def generate_sse(self):
        """Generate text using Server-Sent Events."""
        prompt = request.json.get('prompt', '')
        
        def event_stream():
            generator = StreamingGenerator(self.model, self.tokenizer)
            
            for token in generator.generate_stream(prompt):
                data = {
                    'token': token,
                    'finished': False
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            # Send completion
            final_data = {'finished': True}
            yield f"data: {json.dumps(final_data)}\n\n"
        
        return Response(
            event_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
```

## ⏱️ Latency and Throughput

### Performance Metrics

**1. Latency Measurements**
```python
import time
from collections import defaultdict

class LatencyTracker:
    def __init__(self):
        self.measurements = defaultdict(list)
    
    def measure_latency(self, operation_name):
        """Context manager for measuring operation latency."""
        class LatencyContext:
            def __init__(self, tracker, name):
                self.tracker = tracker
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.perf_counter()
                latency = (end_time - self.start_time) * 1000  # Convert to ms
                self.tracker.measurements[self.name].append(latency)
        
        return LatencyContext(self, operation_name)
    
    def get_stats(self, operation_name):
        """Get latency statistics for an operation."""
        measurements = self.measurements[operation_name]
        if not measurements:
            return None
        
        return {
            'count': len(measurements),
            'mean': sum(measurements) / len(measurements),
            'min': min(measurements),
            'max': max(measurements),
            'p50': sorted(measurements)[len(measurements) // 2],
            'p95': sorted(measurements)[int(len(measurements) * 0.95)],
            'p99': sorted(measurements)[int(len(measurements) * 0.99)]
        }

# Usage example
tracker = LatencyTracker()

with tracker.measure_latency('inference'):
    outputs = model(inputs)

print(tracker.get_stats('inference'))
```

**2. Throughput Optimization**
```python
class ThroughputOptimizer:
    def __init__(self, model, target_latency_ms=100):
        self.model = model
        self.target_latency_ms = target_latency_ms
        self.optimal_batch_size = 1
        
    def find_optimal_batch_size(self, sample_input, max_batch_size=64):
        """Find optimal batch size for target latency."""
        best_throughput = 0
        
        for batch_size in range(1, max_batch_size + 1):
            # Create batch
            batch_input = sample_input.repeat(batch_size, 1)
            
            # Measure latency
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = self.model(batch_input)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            
            # Check if within target latency
            if latency_ms <= self.target_latency_ms:
                throughput = batch_size / (latency_ms / 1000)  # requests/second
                if throughput > best_throughput:
                    best_throughput = throughput
                    self.optimal_batch_size = batch_size
            else:
                break  # Latency exceeded, stop searching
        
        return self.optimal_batch_size, best_throughput
```

**3. Performance Profiling**
```python
import torch.profiler

class ModelProfiler:
    def __init__(self, model):
        self.model = model
        
    def profile_inference(self, sample_input, num_iterations=100):
        """Profile model inference performance."""
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            
            for i in range(num_iterations):
                with torch.no_grad():
                    _ = self.model(sample_input)
                prof.step()
        
        # Print summary
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        return prof
```

## 📊 Model Monitoring

### Health Monitoring

**1. System Metrics**
```python
import psutil
import GPUtil
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    gpu_utilization: List[float]
    gpu_memory_percent: List[float]
    disk_usage_percent: float
    network_io: Dict[str, int]

class SystemMonitor:
    def __init__(self):
        self.metrics_history = []
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpus = GPUtil.getGPUs()
        gpu_utilization = [gpu.load * 100 for gpu in gpus]
        gpu_memory_percent = [gpu.memoryUtil * 100 for gpu in gpus]
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv
        }
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_utilization=gpu_utilization,
            gpu_memory_percent=gpu_memory_percent,
            disk_usage_percent=disk_usage_percent,
            network_io=network_io
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def check_health(self, metrics: SystemMetrics) -> Dict[str, bool]:
        """Check if system is healthy based on metrics."""
        health_status = {
            'cpu_healthy': metrics.cpu_percent < 80,
            'memory_healthy': metrics.memory_percent < 85,
            'gpu_healthy': all(util < 90 for util in metrics.gpu_utilization),
            'gpu_memory_healthy': all(mem < 90 for mem in metrics.gpu_memory_percent),
            'disk_healthy': metrics.disk_usage_percent < 90
        }
        
        health_status['overall_healthy'] = all(health_status.values())
        return health_status
```

**2. Model Performance Monitoring**
```python
class ModelPerformanceMonitor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.request_count = 0
        self.error_count = 0
        self.latency_history = []
        self.throughput_history = []
        
    def log_request(self, latency_ms: float, success: bool = True):
        """Log a model inference request."""
        self.request_count += 1
        self.latency_history.append(latency_ms)
        
        if not success:
            self.error_count += 1
    
    def calculate_throughput(self, time_window_seconds: int = 60):
        """Calculate requests per second in time window."""
        current_time = time.time()
        recent_requests = [
            req for req in self.latency_history 
            if (current_time - req) <= time_window_seconds
        ]
        
        throughput = len(recent_requests) / time_window_seconds
        self.throughput_history.append(throughput)
        return throughput
    
    def get_performance_summary(self):
        """Get performance summary statistics."""
        if not self.latency_history:
            return None
        
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        
        return {
            'model_name': self.model_name,
            'total_requests': self.request_count,
            'error_rate': error_rate,
            'avg_latency_ms': sum(self.latency_history) / len(self.latency_history),
            'p95_latency_ms': sorted(self.latency_history)[int(len(self.latency_history) * 0.95)],
            'current_throughput_rps': self.throughput_history[-1] if self.throughput_history else 0
        }
```

**3. Alerting System**
```python
import smtplib
from email.mime.text import MIMEText
from abc import ABC, abstractmethod

class AlertChannel(ABC):
    @abstractmethod
    def send_alert(self, message: str, severity: str):
        pass

class EmailAlertChannel(AlertChannel):
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, recipients: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
    
    def send_alert(self, message: str, severity: str):
        """Send email alert."""
        subject = f"[{severity.upper()}] Model Serving Alert"
        
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = self.username
        msg['To'] = ', '.join(self.recipients)
        
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)

class AlertManager:
    def __init__(self):
        self.channels: List[AlertChannel] = []
        self.alert_rules = {}
    
    def add_channel(self, channel: AlertChannel):
        """Add alert channel."""
        self.channels.append(channel)
    
    def add_rule(self, rule_name: str, condition_func, message_template: str, severity: str = 'warning'):
        """Add alerting rule."""
        self.alert_rules[rule_name] = {
            'condition': condition_func,
            'message_template': message_template,
            'severity': severity
        }
    
    def check_alerts(self, metrics: dict):
        """Check all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            if rule['condition'](metrics):
                message = rule['message_template'].format(**metrics)
                
                for channel in self.channels:
                    channel.send_alert(message, rule['severity'])

# Usage example
alert_manager = AlertManager()

# Add email channel
email_channel = EmailAlertChannel(
    smtp_server='smtp.gmail.com',
    smtp_port=587,
    username='alerts@company.com',
    password='password',
    recipients=['admin@company.com']
)
alert_manager.add_channel(email_channel)

# Add alert rules
alert_manager.add_rule(
    'high_latency',
    lambda m: m.get('avg_latency_ms', 0) > 1000,
    'High latency detected: {avg_latency_ms:.2f}ms',
    'critical'
)

alert_manager.add_rule(
    'high_error_rate',
    lambda m: m.get('error_rate', 0) > 0.05,
    'High error rate detected: {error_rate:.2%}',
    'warning'
)
```

## 🏗️ Deployment Architectures

### Deployment Patterns

**1. Single Model Serving**
```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

class ModelServer:
    def __init__(self, model_path: str):
        self.model = torch.load(model_path)
        self.model.eval()
        
    @app.route('/predict', methods=['POST'])
    def predict(self):
        try:
            data = request.json
            inputs = torch.tensor(data['inputs'])
            
            with torch.no_grad():
                outputs = self.model(inputs)
            
            return jsonify({
                'predictions': outputs.tolist(),
                'status': 'success'
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

# Start server
if __name__ == '__main__':
    server = ModelServer('model.pth')
    app.run(host='0.0.0.0', port=8080)
```

**2. Multi-Model Serving**
```python
class MultiModelServer:
    def __init__(self):
        self.models = {}
        self.model_configs = {}
    
    def load_model(self, model_name: str, model_path: str, config: dict):
        """Load a model into the server."""
        model = torch.load(model_path)
        model.eval()
        
        self.models[model_name] = model
        self.model_configs[model_name] = config
    
    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.models:
            del self.models[model_name]
            del self.model_configs[model_name]
    
    @app.route('/predict/<model_name>', methods=['POST'])
    def predict(self, model_name: str):
        if model_name not in self.models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        try:
            data = request.json
            inputs = torch.tensor(data['inputs'])
            
            model = self.models[model_name]
            with torch.no_grad():
                outputs = model(inputs)
            
            return jsonify({
                'predictions': outputs.tolist(),
                'model': model_name,
                'status': 'success'
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500
    
    @app.route('/models', methods=['GET'])
    def list_models(self):
        """List available models."""
        return jsonify({
            'models': list(self.models.keys()),
            'count': len(self.models)
        })
```

**3. Load Balancing**
```python
import random
from typing import List

class LoadBalancer:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.current_index = 0
        
    def round_robin(self) -> str:
        """Round-robin load balancing."""
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
    
    def random_selection(self) -> str:
        """Random server selection."""
        return random.choice(self.servers)
    
    def weighted_selection(self, weights: List[float]) -> str:
        """Weighted random selection."""
        return random.choices(self.servers, weights=weights)[0]

# Usage with requests
import requests

class ModelClient:
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
    
    def predict(self, inputs, model_name: str = None):
        """Make prediction request with load balancing."""
        server = self.load_balancer.round_robin()
        
        url = f"http://{server}/predict"
        if model_name:
            url += f"/{model_name}"
        
        response = requests.post(url, json={'inputs': inputs})
        return response.json()
```

## 🔧 Production Considerations

### Scalability Patterns

**1. Horizontal Scaling**
```yaml
# Kubernetes deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-serving
  template:
    metadata:
      labels:
        app: llm-serving
    spec:
      containers:
      - name: llm-server
        image: llm-serving:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-serving
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

**2. Auto-scaling**
```python
class AutoScaler:
    def __init__(self, min_replicas=1, max_replicas=10, target_cpu_percent=70):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_cpu_percent = target_cpu_percent
        self.current_replicas = min_replicas
        
    def should_scale_up(self, current_cpu_percent: float, current_latency_ms: float) -> bool:
        """Determine if scaling up is needed."""
        return (
            current_cpu_percent > self.target_cpu_percent or
            current_latency_ms > 1000  # 1 second threshold
        ) and self.current_replicas < self.max_replicas
    
    def should_scale_down(self, current_cpu_percent: float, current_latency_ms: float) -> bool:
        """Determine if scaling down is possible."""
        return (
            current_cpu_percent < self.target_cpu_percent * 0.5 and
            current_latency_ms < 500  # 500ms threshold
        ) and self.current_replicas > self.min_replicas
    
    def get_target_replicas(self, metrics: dict) -> int:
        """Calculate target number of replicas."""
        cpu_percent = metrics.get('cpu_percent', 0)
        latency_ms = metrics.get('avg_latency_ms', 0)
        
        if self.should_scale_up(cpu_percent, latency_ms):
            return min(self.current_replicas + 1, self.max_replicas)
        elif self.should_scale_down(cpu_percent, latency_ms):
            return max(self.current_replicas - 1, self.min_replicas)
        else:
            return self.current_replicas
```

### Security Considerations

**1. API Authentication**
```python
import jwt
from functools import wraps

class APIAuth:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_token(self, user_id: str, expiration_hours: int = 24) -> str:
        """Generate JWT token."""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expiration_hours)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> dict:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

def require_auth(auth_instance: APIAuth):
    """Decorator for requiring authentication."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'error': 'No token provided'}), 401
            
            try:
                # Remove 'Bearer ' prefix
                token = token.replace('Bearer ', '')
                payload = auth_instance.verify_token(token)
                request.user = payload
                return f(*args, **kwargs)
            except ValueError as e:
                return jsonify({'error': str(e)}), 401
        
        return decorated_function
    return decorator
```

**2. Rate Limiting**
```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        current_time = time.time()
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if current_time - req_time < self.time_window
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(current_time)
            return True
        
        return False

# Usage as decorator
rate_limiter = RateLimiter(max_requests=100, time_window=60)

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_id = request.remote_addr  # Use IP as client ID
        
        if not rate_limiter.is_allowed(client_id):
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        return f(*args, **kwargs)
    
    return decorated_function
```

## 📚 Summary

Deploying and serving Large Language Models in production requires careful attention to:

### Key Components
- **Model Serialization**: Efficient storage and loading formats
- **Inference Optimization**: Hardware-specific optimizations and memory management
- **Batching Strategies**: Static, dynamic, and continuous batching for throughput
- **Streaming**: Real-time token generation for interactive applications
- **Monitoring**: System health, performance metrics, and alerting

### Performance Optimization
- **Latency**: Minimize response time through optimization techniques
- **Throughput**: Maximize requests per second through batching
- **Resource Utilization**: Efficient use of CPU, GPU, and memory
- **Scalability**: Horizontal and vertical scaling strategies

### Production Readiness
- **Reliability**: Error handling, failover, and recovery mechanisms
- **Security**: Authentication, authorization, and rate limiting
- **Monitoring**: Comprehensive observability and alerting
- **Maintenance**: Model updates, A/B testing, and rollback capabilities

Successful LLM deployment requires balancing performance, cost, and reliability while maintaining high availability and user experience.