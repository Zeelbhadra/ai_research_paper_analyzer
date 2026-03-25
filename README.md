🧠 AI Research Paper Analyzer

An AI-powered web application that retrieves research papers from multiple academic sources and generates concise summaries using Large Language Models (LLMs), with MongoDB-based caching for improved performance and reduced API usage.

🚀 Features
🔍 Multi-source Paper Retrieval
Fetches research papers from:
arXiv
Semantic Scholar
🧾 AI-Based Summarization
Uses OpenAI API to generate concise summaries from abstracts
Designed to provide quick insights into research papers
🗄️ MongoDB Summary Caching
Stores generated summaries
Avoids recomputation for previously processed papers
Improves response time and reduces API cost
⚡ Efficient API Workflow
Checks database before calling OpenAI
Returns cached summaries instantly when available
🌐 Flask-Based Web Interface
Simple UI for searching and analyzing research papers

🏗️ System Architecture
User Query
   ↓
Flask API (app.py)
   ↓
Paper Fetcher (arXiv + Semantic Scholar)
   ↓
Check MongoDB (summary exists?)
   ↓            ↓
Yes ✅         No ❌
 ↓              ↓
Return       Generate Summary (OpenAI)
                 ↓
           Store in MongoDB
                 ↓
            Return Response

🧑‍💻 Tech Stack
Backend: Flask
Database: MongoDB Atlas
AI: OpenAI API
APIs: arXiv, Semantic Scholar
