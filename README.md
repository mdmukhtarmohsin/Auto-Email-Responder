# AI-Powered Email Responder

An intelligent email response system that automatically processes incoming Gmail messages and generates contextual responses using Google's Gemini 2.5 Flash, LangChain, LangGraph, and Gmail MCP integration.

## 🚀 Features

- **Automated Email Processing**: Fetches unread emails from Gmail and processes them automatically
- **Intent Classification**: Intelligently categorizes emails (billing, technical support, feature requests, general)
- **Policy-Based Responses**: Retrieves relevant company policies using vector search for accurate responses
- **Gemini 2.5 Integration**: Uses Google's latest AI model for generating professional email responses
- **Gmail MCP**: FastMCP integration for seamless Gmail API operations
- **Caching Layer**: Redis/memory caching for improved performance
- **LangGraph Workflow**: Orchestrates the complete email processing pipeline
- **Monitoring & Logging**: Comprehensive system status monitoring and logging

## 🏗️ Architecture

```
Gmail → Email Fetcher → Intent Classifier → Policy Retriever → Gemini LLM → Email Sender → Gmail
                           ↓                     ↓               ↓
                    Cache Manager ←→ Vector Database ←→ Response Cache
```

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- pnpm (for any Node.js dependencies)
- Gmail API credentials
- Google AI (Gemini) API key
- Redis (optional, for enhanced caching)

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd Auto-Email-Responder

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Required: Gemini API Configuration
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp

# Required: Gmail Configuration
GMAIL_EMAIL_ADDRESS=your-email@gmail.com
GMAIL_CREDENTIALS_PATH=credentials.json
GMAIL_TOKEN_PATH=token.json

# Optional: Redis Configuration (for enhanced caching)
REDIS_URL=redis://localhost:6379/0
USE_REDIS_CACHE=false

# Optional: Application Settings
LOG_LEVEL=INFO
RESPONSE_TONE=polite
PROCESSING_INTERVAL_MINUTES=5
MAX_EMAILS_PER_BATCH=50
MAX_RESPONSE_LENGTH=500
```

### 3. Gmail API Setup

1. **Enable Gmail API**:

   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Enable the Gmail API
   - Create credentials (OAuth 2.0 Client ID)
   - Download the credentials JSON file

2. **Configure Credentials**:

   ```bash
   # Place your Gmail credentials file
   cp path/to/your/credentials.json credentials.json
   ```

3. **First-time Authentication**:
   ```bash
   # Run this to authenticate and create token.json
   python3 main.py --test
   ```

### 4. Policy Knowledge Base

The system includes sample policy files. You can customize them:

```bash
# Edit policy files
nano config/policies/billing.md
nano config/policies/technical_support.md
nano config/policies/general.md
```

Add your own policy files in Markdown format to `config/policies/` directory.

## 📖 Usage

### Basic Commands

```bash
# Process emails once
python3 main.py

# Run continuously as daemon
python3 main.py --daemon

# Check system status
python3 main.py --status

# Test all components
python3 main.py --test

# Refresh policy knowledge base
python3 main.py --refresh-policies

# Enable verbose logging
python3 main.py --verbose
```

### Running as a Service

For production deployment, you can run the system as a systemd service:

```bash
# Create systemd service file
sudo nano /etc/systemd/system/email-responder.service
```

```ini
[Unit]
Description=AI-Powered Email Responder
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/Auto-Email-Responder
Environment=PATH=/path/to/Auto-Email-Responder/venv/bin
ExecStart=/path/to/Auto-Email-Responder/venv/bin/python3 main.py --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable email-responder
sudo systemctl start email-responder
sudo systemctl status email-responder
```

## 🔧 Configuration Options

### Environment Variables

| Variable                      | Description                    | Default                    |
| ----------------------------- | ------------------------------ | -------------------------- |
| `GOOGLE_API_KEY`              | Gemini API key (required)      | -                          |
| `GEMINI_MODEL`                | Gemini model name              | `gemini-2.0-flash-exp`     |
| `GMAIL_EMAIL_ADDRESS`         | Your Gmail address             | -                          |
| `GMAIL_CREDENTIALS_PATH`      | Path to Gmail credentials JSON | `credentials.json`         |
| `GMAIL_TOKEN_PATH`            | Path to Gmail token file       | `token.json`               |
| `REDIS_URL`                   | Redis connection URL           | `redis://localhost:6379/0` |
| `USE_REDIS_CACHE`             | Enable Redis caching           | `false`                    |
| `LOG_LEVEL`                   | Logging level                  | `INFO`                     |
| `RESPONSE_TONE`               | Email response tone            | `polite`                   |
| `PROCESSING_INTERVAL_MINUTES` | Processing interval            | `5`                        |
| `MAX_EMAILS_PER_BATCH`        | Max emails per batch           | `50`                       |
| `MAX_RESPONSE_LENGTH`         | Max response length            | `500`                      |
| `CHUNK_SIZE`                  | Document chunk size            | `1000`                     |
| `CHUNK_OVERLAP`               | Document chunk overlap         | `200`                      |
| `TOP_K_DOCS`                  | Top documents to retrieve      | `5`                        |

### Policy Files

Policy files should be placed in `config/policies/` as Markdown files:

- `billing.md` - Billing and payment related policies
- `technical_support.md` - Technical support procedures
- `general.md` - General company information
- Custom policies can be added as needed

## 🔍 System Monitoring

### Health Check

```bash
# Quick system status
python3 main.py --status
```

Output example:

```
=== EMAIL RESPONDER SYSTEM STATUS ===
Timestamp: 2024-01-15T10:30:00
Gemini Model: gemini-2.0-flash-exp
Gmail Email: support@company.com
Processing Interval: 5 minutes

Component Status:
  ✓ workflow_ready: Ready
  ✓ gmail_fetcher_ready: Ready
  ✓ policy_retriever_ready: Ready
  ✓ llm_response_ready: Ready
  ✓ email_sender_ready: Ready
  ✓ cache_ready: Ready

Policy Index:
  Vectorstore: Exists
  Documents: 3
```

### Logs

Logs are written to:

- `logs/email_responder.log` - Main application log
- Console output for immediate feedback

## 🧪 Testing

### Component Testing

```bash
# Test all components
python3 main.py --test
```

### Individual Component Tests

```python
# Test in Python shell
from src.email_responder.gmail_fetcher import gmail_fetcher
from src.email_responder.llm_response_chain import llm_response_chain

# Test Gmail connection
gmail_fetcher.test_connection()

# Test LLM connection
llm_response_chain.test_connection()
```

## 🔒 Security Considerations

1. **API Keys**: Store API keys securely in environment variables
2. **Gmail Credentials**: Protect OAuth credentials and tokens
3. **Cache Security**: Use Redis AUTH if deploying with Redis
4. **Network Security**: Consider VPC/firewall rules for production
5. **Logging**: Avoid logging sensitive email content

## 🚀 Performance Optimization

### Caching

- **Redis**: Enable Redis for better caching performance
- **Memory Cache**: Fallback to in-memory caching
- **Cache TTL**: Configure appropriate cache expiration times

### Batch Processing

- **Email Batching**: Process multiple emails in batches
- **Rate Limiting**: Respect Gmail API rate limits
- **Concurrent Processing**: Consider async processing for high volume

### Vector Search

- **Embedding Cache**: Cache document embeddings
- **Index Optimization**: Regularly refresh policy index
- **Document Chunking**: Optimize chunk size for your policies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Gmail Authentication Failed**

   - Check credentials.json file
   - Re-run authentication: `python3 main.py --test`
   - Verify Gmail API is enabled

2. **Gemini API Errors**

   - Verify API key is correct
   - Check API quotas and billing
   - Ensure model name is correct

3. **Vector Index Missing**

   - Run: `python3 main.py --refresh-policies`
   - Check policy files exist in `config/policies/`

4. **Redis Connection Failed**

   - Check Redis is running: `redis-cli ping`
   - Verify REDIS_URL in .env
   - Set USE_REDIS_CACHE=false to use memory cache

5. **Permission Errors**
   - Check file permissions for logs/ directory
   - Ensure virtual environment is activated
   - Verify Python path in systemd service

### Getting Help

- Check logs in `logs/email_responder.log`
- Run with `--verbose` flag for detailed output
- Use `--status` to check component health
- Review configuration in `.env` file

## 📊 Metrics and Analytics

The system provides comprehensive metrics:

- **Processing Statistics**: Success/failure rates
- **Response Times**: Average processing time per email
- **Cache Performance**: Hit/miss ratios
- **Component Health**: Real-time status monitoring
- **API Usage**: Gemini and Gmail API call tracking

For production deployments, consider integrating with monitoring tools like Prometheus, Grafana, or cloud monitoring services.
