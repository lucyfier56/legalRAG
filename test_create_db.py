import os
import pickle
import logging
import json
import hashlib
import uuid
import gc
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict
import re
import tempfile
import time

# Core ML/NLP imports
import torch
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Vector Database
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition

# BM25 for hybrid retrieval
from rank_bm25 import BM25Okapi

# Document processing fallbacks
import fitz  # PyMuPDF
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image

# Graph RAG support
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalEntityExtractor:
    """Enhanced legal entity extraction for Graph RAG integration"""
    
    def __init__(self):
        self.judge_patterns = [
            r"HON'BLE\s+(?:MR\.|MS\.|MRS\.)?\s*JUSTICE\s+([A-Z][A-Z\.\s]+?)(?:\s+J\.|$|\s+JUDGMENT|\s+ORDER|\s+JUDGEMENT)",
            r"CORAM:\s*HON'BLE\s+(?:MR\.|MS\.|MRS\.)?\s*JUSTICE\s+([A-Z][A-Z\.\s]+?)(?:\s+J\.|$|\s+JUDGMENT|\s+ORDER|\s+JUDGEMENT)",
            r"JUSTICE\s+([A-Z][A-Z\.\s]+?)(?:\s+J\.|$|\s+JUDGMENT|\s+ORDER|\s+JUDGEMENT|\s+HON)",
            r"([A-Z][A-Z\.\s]+?),?\s*J\.(?:\s|$)",
            r"BEFORE.*?JUSTICE\s+([A-Z][A-Z\.\s]+?)(?:\s+J\.|$|\s+JUDGMENT|\s+ORDER|\s+JUDGEMENT)",
        ]
        
        self.exclude_words = {
            'judgment', 'order', 'judgement', 'court', 'high', 'supreme',
            'case', 'vs', 'versus', 'plaintiff', 'defendant', 'petitioner',
            'respondent', 'application', 'appeal', 'petition', 'suit',
            'civil', 'criminal', 'company', 'limited', 'ltd', 'inc',
            'private', 'public', 'government', 'state', 'union', 'india',
            'delhi', 'bombay', 'calcutta', 'madras', 'allahabad'
        }
    
    def extract_legal_metadata(self, text: str) -> Dict[str, List[str]]:
        """Extract legal metadata with enhanced precision for Graph RAG"""
        metadata = {
            "judges": [],
            "parties": [],
            "case_numbers": [],
            "courts": []
        }
        
        # Extract judges with validation
        for pattern in self.judge_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    for submatch in match:
                        judge_name = self._clean_and_validate_judge_name(submatch)
                        if judge_name and judge_name not in metadata["judges"]:
                            metadata["judges"].append(judge_name)
                else:
                    judge_name = self._clean_and_validate_judge_name(match)
                    if judge_name and judge_name not in metadata["judges"]:
                        metadata["judges"].append(judge_name)
        
        # Extract parties
        vs_patterns = [
            r"([A-Z][A-Z\s&\.]+)\s+(?:v\.?|versus)\s+([A-Z][A-Z\s&\.]+)",
            r"([A-Z][A-Z\s&\.]+)\s+\.{3,}\s*Plaintiff.*?([A-Z][A-Z\s&\.]+)\s+\.{3,}\s*Defendants?"
        ]
        
        for pattern in vs_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    plaintiff = re.sub(r'\s+', ' ', match[0].strip())
                    defendant = re.sub(r'\s+', ' ', match[1].strip())
                    if plaintiff not in metadata["parties"]:
                        metadata["parties"].append(plaintiff)
                    if defendant not in metadata["parties"]:
                        metadata["parties"].append(defendant)
        
        # Extract case numbers
        case_patterns = [
            r"CS\(COMM\)\s+(\d+/\d+)",
            r"Civil\s+Suit\s+No[.:]?\s*(\d+/\d+)",
            r"Case\s+No[.:]?\s*([A-Z0-9/\(\)]+)",
            r"(\d{4}/DHC/\d+)",
        ]
        
        for pattern in case_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match not in metadata["case_numbers"]:
                    metadata["case_numbers"].append(match)
        
        # Final validation
        metadata["judges"] = [name for name in metadata["judges"] if self._is_valid_judge_name(name)]
        
        return metadata
    
    def _clean_and_validate_judge_name(self, name: str) -> str:
        """Clean and validate judge names for Graph RAG"""
        if not name:
            return ""
        
        # Basic cleaning
        name = re.sub(r'\s+', ' ', name.strip())
        name = re.sub(r'\b(HON\'?BLE|MR|MS|MRS|JUSTICE|J\.)\b', '', name, flags=re.IGNORECASE).strip()
        name = re.sub(r'\b(JUDGMENT|ORDER|JUDGEMENT)\b', '', name, flags=re.IGNORECASE).strip()
        name = re.sub(r'[,.\-]+$', '', name).strip()
        
        # Remove duplicate names
        words = name.split()
        seen = set()
        cleaned_words = []
        for word in words:
            word_clean = word.strip().upper()
            if word_clean not in seen and len(word_clean) > 1:
                seen.add(word_clean)
                cleaned_words.append(word.strip())
        
        name = ' '.join(cleaned_words)
        
        # Validation checks
        words = name.split()
        if len(words) < 2 or len(name) < 4 or len(name) > 50:
            return ""
        
        name_lower = name.lower()
        if any(excluded in name_lower for excluded in self.exclude_words):
            return ""
        
        return name
    
    def _is_valid_judge_name(self, name: str) -> bool:
        """Validate extracted judge names"""
        if not name or len(name) < 4:
            return False
        
        words = name.split()
        if len(words) < 2:
            return False
        
        # Check for invalid phrases
        invalid_phrases = [
            'is allowed', 'and ought', 'would be', 'is done', 'must be',
            'case wherein', 'in any', 'through', 'present', 'vs', 'versus',
            'limited', 'private', 'company', 'ltd', 'inc'
        ]
        
        name_lower = name.lower()
        if any(phrase in name_lower for phrase in invalid_phrases):
            return False
        
        # Character composition check
        letter_count = sum(1 for char in name if char.isalpha())
        if letter_count < len(name) * 0.7:
            return False
        
        return True

class GraphRAGDataCreator:
    """Enhanced data creator with Graph RAG support - ONE PDF = ONE CHUNK"""
    
    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "test_documents",
                 data_dir: str = "legal_data",
                 batch_size: int = 10):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        
        # Initialize components
        self.device = self._setup_device()
        self.qdrant_client = None
        self.embedding_model = None
        self.bm25_index = None
        self.documents = []
        self.corpus_tokens = []
        self.processed_files = {}
        
        # Graph RAG components
        self.legal_extractor = LegalEntityExtractor()
        self.knowledge_graph = nx.Graph()
        self.judge_case_relationships = defaultdict(list)
        self.case_judge_relationships = defaultdict(list)
        
        # File paths
        self.bm25_index_path = self.data_dir / "bm25_index.pkl"
        self.documents_path = self.data_dir / "documents.pkl"
        self.processed_files_path = self.data_dir / "processed_files.json"
        self.knowledge_graph_path = self.data_dir / "knowledge_graph.pkl"
        self.processing_log_path = self.data_dir / "processing_log.txt"
        
        # Processing configuration
        self.min_content_threshold = 20
        self.min_page_content = 10
        self.max_retries = 3
        
        # Initialize system
        self._initialize_system()
    
    def _setup_device(self) -> str:
        """Setup optimal device for processing"""
        if torch.backends.mps.is_available():
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            logger.info("Using MPS device")
            return 'mps'
        elif torch.cuda.is_available():
            logger.info("Using CUDA device")
            return 'cuda'
        else:
            logger.info("Using CPU device")
            return 'cpu'
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            logger.info("✅ Qdrant client initialized")
            
            # Initialize embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Embedding model initialized")
            
            # Create collection
            self._setup_collection()
            
            # Load existing data
            self._load_existing_data()
            
            # Initialize processing log
            self._initialize_processing_log()
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {str(e)}")
            raise
    
    def _setup_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                logger.info(f"✅ Created collection: {self.collection_name}")
            else:
                logger.info(f"✅ Collection exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Collection setup failed: {str(e)}")
            raise
    
    def _load_existing_data(self):
        """Load existing processed data"""
        try:
            # Load BM25 index
            if self.bm25_index_path.exists():
                with open(self.bm25_index_path, 'rb') as f:
                    index_data = pickle.load(f)
                    self.bm25_index = index_data.get('bm25_index')
                    self.corpus_tokens = index_data.get('corpus_tokens', [])
                logger.info(f"✅ Loaded BM25 index with {len(self.corpus_tokens)} documents")
            
            # Load documents
            if self.documents_path.exists():
                with open(self.documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"✅ Loaded {len(self.documents)} documents")
            
            # Load processed files
            if self.processed_files_path.exists():
                with open(self.processed_files_path, 'r') as f:
                    self.processed_files = json.load(f)
                logger.info(f"✅ Loaded {len(self.processed_files)} processed file records")
            
            # Load knowledge graph
            if self.knowledge_graph_path.exists():
                with open(self.knowledge_graph_path, 'rb') as f:
                    graph_data = pickle.load(f)
                    self.knowledge_graph = graph_data.get('graph', nx.Graph())
                    self.judge_case_relationships = graph_data.get('judge_case_relationships', defaultdict(list))
                    self.case_judge_relationships = graph_data.get('case_judge_relationships', defaultdict(list))
                logger.info(f"✅ Loaded knowledge graph with {self.knowledge_graph.number_of_nodes()} nodes")
                
        except Exception as e:
            logger.warning(f"⚠️ Error loading existing data: {str(e)}")
    
    def _initialize_processing_log(self):
        """Initialize processing log file"""
        try:
            with open(self.processing_log_path, 'w') as f:
                f.write("LEGAL DOCUMENT PROCESSING - ONE PDF = ONE CHUNK\n")
                f.write("=" * 70 + "\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Collection: {self.collection_name}\n")
                f.write(f"Graph RAG: ENABLED\n")
                f.write(f"Mode: ONE PDF = ONE CHUNK = ONE ID\n\n")
        except Exception as e:
            logger.warning(f"Log initialization warning: {str(e)}")
    
    def process_and_store_document(self, pdf_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Process ONE PDF into ONE CHUNK with ONE ID"""
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                return {"success": False, "error": "File not found"}
            
            # Generate file hash
            file_hash = self._get_file_hash(str(pdf_path))
            
            # Check if already processed
            if not force_reprocess and file_hash in self.processed_files:
                self._log_processing(f"📋 Document already processed: {pdf_path.name}")
                return {"success": True, "message": "Already processed", "chunks": 1}
            
            # Extract text from all pages
            page_documents = self.extract_text_multiple_methods(str(pdf_path))
            if not page_documents:
                return {"success": False, "error": "No text extracted"}
            
            # ⚠️ KEY CHANGE: Combine ALL pages into ONE single text
            full_text = ""
            for i, page_doc in enumerate(page_documents):
                if i > 0:
                    full_text += "\n\n"  # Add page separator
                full_text += page_doc.page_content
            
            # Check minimum content threshold
            if len(full_text.strip()) < self.min_content_threshold:
                return {"success": False, "error": "Document content too short"}
            
            # Extract legal metadata from ENTIRE document
            legal_metadata = self.legal_extractor.extract_legal_metadata(full_text)
            
            # Build knowledge graph
            self._build_knowledge_graph(pdf_path.stem, legal_metadata)
            
            # ⚠️ KEY CHANGE: Generate ONE unique ID for the entire PDF
            document_id = str(uuid.uuid4())
            
            try:
                # Generate ONE embedding for the ENTIRE document
                embedding = self.embedding_model.embed_query(full_text)
                
                # Create comprehensive metadata for the ENTIRE document
                metadata = {
                    "text": full_text,
                    "content": full_text,
                    "source": str(pdf_path),
                    "source_name": pdf_path.stem,
                    "total_pages": len(page_documents),
                    "document_id": document_id,
                    "file_hash": file_hash,
                    "processed_at": datetime.now().isoformat(),
                    "collection": self.collection_name,
                    "chunking_mode": "single_pdf_one_chunk",
                    
                    # Legal metadata for Graph RAG
                    "document_judges": legal_metadata["judges"],
                    "document_parties": legal_metadata["parties"],
                    "document_case_numbers": legal_metadata["case_numbers"],
                    "document_courts": legal_metadata["courts"],
                    
                    # Document-level flags
                    "contains_judge_info": self._contains_judge_keywords(full_text),
                    "contains_parties": self._contains_party_keywords(full_text),
                    "contains_case_info": self._contains_case_keywords(full_text),
                    
                    # Document statistics
                    "document_length": len(full_text),
                    "word_count": len(full_text.split()),
                    "character_count": len(full_text),
                }
                
                # ⚠️ KEY CHANGE: Store ONE point with the document_id
                point = PointStruct(
                    id=document_id,  # Use the generated document_id
                    vector=embedding,
                    payload=metadata
                )
                
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )
                
                # Create ONE document for BM25
                single_document = Document(
                    page_content=full_text,
                    metadata=metadata
                )
                
                # Update BM25 index with ONE document
                self.documents.append(single_document)
                document_tokens = self._tokenize_text(full_text)
                self.corpus_tokens.append(document_tokens)
                self._rebuild_bm25_index()
                
                # Mark as processed
                self.processed_files[file_hash] = {
                    "file_path": str(pdf_path),
                    "file_name": pdf_path.name,
                    "document_id": document_id,
                    "chunks_count": 1,  # Always 1
                    "document_length": len(full_text),
                    "word_count": len(full_text.split()),
                    "total_pages": len(page_documents),
                    "processed_at": datetime.now().isoformat(),
                    "legal_metadata": legal_metadata,
                    "collection": self.collection_name,
                    "chunking_mode": "single_pdf_one_chunk",
                    "graph_entities_added": len(legal_metadata["judges"]) + len(legal_metadata["parties"])
                }
                
                # Save all data
                self._save_all_data()
                
                self._log_processing(f"✅ PDF processed as ONE CHUNK: {pdf_path.name}")
                self._log_processing(f"📊 Document ID: {document_id}")
                self._log_processing(f"📊 Document stats: {len(full_text)} chars, {len(page_documents)} pages")
                self._log_processing(f"📊 Legal entities: {len(legal_metadata['judges'])} judges, {len(legal_metadata['parties'])} parties")
                
                return {
                    "success": True,
                    "message": f"Successfully processed {pdf_path.name} as ONE chunk",
                    "document_id": document_id,
                    "chunks": 1,
                    "document_length": len(full_text),
                    "word_count": len(full_text.split()),
                    "total_pages": len(page_documents),
                    "file_hash": file_hash,
                    "legal_metadata": legal_metadata,
                    "graph_entities_added": len(legal_metadata["judges"]) + len(legal_metadata["parties"])
                }
                
            except Exception as e:
                logger.error(f"Error storing document: {str(e)}")
                return {"success": False, "error": f"Storage failed: {str(e)}"}
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _build_knowledge_graph(self, document_id: str, legal_metadata: Dict[str, List[str]]):
        """Build knowledge graph from legal metadata"""
        try:
            judges = legal_metadata.get("judges", [])
            parties = legal_metadata.get("parties", [])
            case_numbers = legal_metadata.get("case_numbers", [])
            
            # Add judge nodes and relationships
            for judge in judges:
                judge_id = f"judge_{judge.replace(' ', '_')}"
                
                # Add judge node
                self.knowledge_graph.add_node(judge_id, 
                                            type="judge", 
                                            name=judge, 
                                            document_id=document_id)
                
                # Build judge-case relationships
                self.judge_case_relationships[judge].append(document_id)
                self.case_judge_relationships[document_id].append(judge)
                
                # Add judge-document relationship
                doc_id = f"doc_{document_id}"
                self.knowledge_graph.add_node(doc_id, 
                                            type="document", 
                                            name=document_id)
                self.knowledge_graph.add_edge(judge_id, doc_id, 
                                            relationship="presided_over", 
                                            weight=1.0)
            
            # Add party nodes and relationships
            for party in parties:
                party_id = f"party_{party.replace(' ', '_')}"
                
                # Add party node
                self.knowledge_graph.add_node(party_id, 
                                            type="party", 
                                            name=party, 
                                            document_id=document_id)
                
                # Connect parties to judges
                for judge in judges:
                    judge_id = f"judge_{judge.replace(' ', '_')}"
                    self.knowledge_graph.add_edge(judge_id, party_id, 
                                                relationship="adjudicated_case_involving", 
                                                weight=1.0)
            
            # Add case number nodes
            for case_num in case_numbers:
                case_id = f"case_{case_num.replace('/', '_')}"
                
                # Add case node
                self.knowledge_graph.add_node(case_id, 
                                            type="case", 
                                            name=case_num, 
                                            document_id=document_id)
                
                # Connect cases to judges
                for judge in judges:
                    judge_id = f"judge_{judge.replace(' ', '_')}"
                    self.knowledge_graph.add_edge(judge_id, case_id, 
                                                relationship="decided", 
                                                weight=1.0)
            
            logger.info(f"Built knowledge graph for {document_id}: {len(judges)} judges, {len(parties)} parties")
            
        except Exception as e:
            logger.error(f"Knowledge graph building failed: {str(e)}")
    
    def _contains_judge_keywords(self, text: str) -> bool:
        """Check if text contains judge-related content"""
        judge_keywords = ["judge", "justice", "hon'ble", "coram", "bench"]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in judge_keywords)
    
    def _contains_party_keywords(self, text: str) -> bool:
        """Check if text contains party-related content"""
        party_keywords = ["plaintiff", "defendant", "petitioner", "respondent", "versus", "vs.", "appellant"]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in party_keywords)
    
    def _contains_case_keywords(self, text: str) -> bool:
        """Check if text contains case-related content"""
        case_keywords = ["case", "suit", "petition", "application", "appeal", "cs(comm)", "civil suit"]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in case_keywords)
    
    def _cleanup_memory(self):
        """Enhanced memory cleanup"""
        try:
            if self.device == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            elif self.device == 'cuda' and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.warning(f"Memory cleanup warning: {str(e)}")
    
    def _log_processing(self, message: str):
        """Enhanced logging"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            with open(self.processing_log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            logger.info(message)
        except Exception as e:
            logger.warning(f"Logging warning: {str(e)}")
    
    def extract_text_multiple_methods(self, pdf_path: str) -> List[Document]:
        """Extract text using multiple fallback methods"""
        extraction_methods = [
            ("PyPDFLoader", self._extract_with_pypdf_loader),
            ("PyMuPDF", self._extract_with_pymupdf),
            ("PyPDF2", self._extract_with_pypdf2),
            ("pdfplumber", self._extract_with_pdfplumber),
            ("OCR", self._extract_with_ocr)
        ]
        
        for method_name, method_func in extraction_methods:
            for attempt in range(self.max_retries):
                try:
                    self._log_processing(f"🔄 Trying {method_name} for {Path(pdf_path).name} (attempt {attempt + 1})")
                    documents = method_func(pdf_path)
                    
                    if documents and len(documents) > 0:
                        total_content = sum(len(doc.page_content.strip()) for doc in documents)
                        if total_content > self.min_content_threshold:
                            self._log_processing(f"✅ {method_name} succeeded: {len(documents)} pages, {total_content} chars")
                            return documents
                    
                    if documents:
                        break
                        
                except Exception as e:
                    self._log_processing(f"❌ {method_name} attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        continue
                
                time.sleep(0.5)
        
        self._log_processing(f"❌ All extraction methods failed for {pdf_path}")
        return []
    
    def _extract_with_pypdf_loader(self, pdf_path: str) -> List[Document]:
        """Extract with PyPDFLoader"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            return [doc for doc in documents if doc.page_content.strip()]
        except Exception as e:
            raise Exception(f"PyPDFLoader error: {str(e)}")
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[Document]:
        """Extract with PyMuPDF"""
        doc = None
        try:
            doc = fitz.open(pdf_path)
            documents = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num + 1, "method": "PyMuPDF"}
                    ))
            
            return documents
        except Exception as e:
            raise Exception(f"PyMuPDF error: {str(e)}")
        finally:
            if doc:
                doc.close()
    
    def _extract_with_pypdf2(self, pdf_path: str) -> List[Document]:
        """Extract with PyPDF2"""
        documents = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": pdf_path, "page": page_num + 1, "method": "PyPDF2"}
                        ))
            return documents
        except Exception as e:
            raise Exception(f"PyPDF2 error: {str(e)}")
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Document]:
        """Extract with pdfplumber"""
        documents = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": pdf_path, "page": page_num + 1, "method": "pdfplumber"}
                        ))
            return documents
        except Exception as e:
            raise Exception(f"pdfplumber error: {str(e)}")
    
    def _extract_with_ocr(self, pdf_path: str) -> List[Document]:
        """Extract with OCR as fallback"""
        documents = []
        doc = None
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Try normal text extraction first
                text = page.get_text()
                if len(text.strip()) > 50:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num + 1, "method": "text"}
                    ))
                    continue
                
                # OCR for problematic pages
                try:
                    zoom = 2.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        temp_file_path = temp_file.name
                    
                    pix.save(temp_file_path)
                    image = Image.open(temp_file_path)
                    ocr_text = pytesseract.image_to_string(image)
                    os.unlink(temp_file_path)
                    
                    if ocr_text.strip():
                        documents.append(Document(
                            page_content=ocr_text,
                            metadata={"source": pdf_path, "page": page_num + 1, "method": "OCR"}
                        ))
                        
                except Exception as ocr_error:
                    logger.warning(f"OCR failed for page {page_num + 1}: {str(ocr_error)}")
                    continue
            
            return documents
        except Exception as e:
            raise Exception(f"OCR extraction error: {str(e)}")
        finally:
            if doc:
                doc.close()
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        try:
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            tokens = text.split()
            return [token for token in tokens if len(token) > 2]
        except Exception:
            return []
    
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index"""
        try:
            if self.corpus_tokens:
                self.bm25_index = BM25Okapi(self.corpus_tokens)
                logger.info(f"✅ BM25 index rebuilt with {len(self.corpus_tokens)} documents")
        except Exception as e:
            logger.error(f"❌ BM25 index rebuild failed: {str(e)}")
    
    def _save_all_data(self):
        """Save all data including knowledge graph"""
        try:
            # Save BM25 index
            bm25_data = {
                'bm25_index': self.bm25_index,
                'corpus_tokens': self.corpus_tokens
            }
            
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump(bm25_data, f)
            
            # Save documents
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save processed files
            with open(self.processed_files_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, indent=2)
            
            # Save knowledge graph
            graph_data = {
                'graph': self.knowledge_graph,
                'judge_case_relationships': dict(self.judge_case_relationships),
                'case_judge_relationships': dict(self.case_judge_relationships)
            }
            
            with open(self.knowledge_graph_path, 'wb') as f:
                pickle.dump(graph_data, f)
            
            logger.info("✅ All data saved including knowledge graph")
            
        except Exception as e:
            logger.error(f"❌ Data saving failed: {str(e)}")
    
    def process_directory(self, directory_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Process directory - ONE PDF = ONE CHUNK"""
        try:
            directory_path = Path(directory_path)
            if not directory_path.exists():
                return {"success": False, "error": "Directory not found"}
            
            pdf_files = list(directory_path.glob("**/*.pdf"))
            self._log_processing(f"📁 Found {len(pdf_files)} PDF files in {directory_path}")
            
            results = {
                "success": True,
                "total_files": len(pdf_files),
                "processed_files": 0,
                "failed_files": 0,
                "total_chunks": 0,
                "total_graph_entities": 0,
                "total_document_length": 0,
                "total_word_count": 0,
                "files_results": {},
                "graph_rag_enabled": True,
                "chunking_mode": "one_pdf_one_chunk"
            }
            
            for i, pdf_file in enumerate(pdf_files, 1):
                try:
                    self._log_processing(f"📄 Processing ({i}/{len(pdf_files)}): {pdf_file.name}")
                    result = self.process_and_store_document(str(pdf_file), force_reprocess)
                    results["files_results"][pdf_file.name] = result
                    
                    if result["success"]:
                        chunks = result.get("chunks", 0)
                        graph_entities = result.get("graph_entities_added", 0)
                        doc_length = result.get("document_length", 0)
                        word_count = result.get("word_count", 0)
                        
                        if chunks > 0:
                            results["processed_files"] += 1
                            results["total_chunks"] += chunks
                            results["total_graph_entities"] += graph_entities
                            results["total_document_length"] += doc_length
                            results["total_word_count"] += word_count
                    else:
                        results["failed_files"] += 1
                    
                    # Memory cleanup every 5 files
                    if i % 5 == 0:
                        self._cleanup_memory()
                        
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                    results["failed_files"] += 1
                    results["files_results"][pdf_file.name] = {"success": False, "error": str(e)}
            
            # Final save
            self._save_all_data()
            
            self._log_processing(f"✅ Directory processing completed (ONE PDF = ONE CHUNK):")
            self._log_processing(f" 📊 Processed: {results['processed_files']} files")
            self._log_processing(f" 📚 Total chunks: {results['total_chunks']} (1 per PDF)")
            self._log_processing(f" 🕸️ Graph entities: {results['total_graph_entities']}")
            self._log_processing(f" 🔗 Graph nodes: {self.knowledge_graph.number_of_nodes()}")
            self._log_processing(f" 🔗 Graph edges: {self.knowledge_graph.number_of_edges()}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Directory processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics including Graph RAG metrics"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            return {
                "storage_stats": {
                    "vector_db_vectors": collection_info.points_count,
                    "bm25_documents": len(self.documents),
                    "processed_files": len(self.processed_files),
                },
                "graph_rag_stats": {
                    "knowledge_graph_nodes": self.knowledge_graph.number_of_nodes(),
                    "knowledge_graph_edges": self.knowledge_graph.number_of_edges(),
                    "judge_case_relationships": len(self.judge_case_relationships),
                    "case_judge_relationships": len(self.case_judge_relationships),
                },
                "system_info": {
                    "device": self.device,
                    "collection_name": self.collection_name,
                    "graph_rag_enabled": True,
                    "chunking_mode": "one_pdf_one_chunk",
                    "chunks_per_document": 1,
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Stats retrieval failed: {str(e)}")
            return {"error": str(e)}

def main():
    """Main function - ONE PDF = ONE CHUNK"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal Document Processing - ONE PDF = ONE CHUNK")
    parser.add_argument("--directory", "-d", required=True, help="Directory containing PDF files")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocess all files")
    parser.add_argument("--clear", action="store_true", help="Clear all existing data first")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Initialize Graph RAG data creator
    creator = GraphRAGDataCreator(batch_size=args.batch_size)
    
    print("🚀 Legal Document Processing - ONE PDF = ONE CHUNK")
    print("=" * 60)
    print("✨ Features:")
    print(" 📄 ONE PDF = ONE CHUNK = ONE ID")
    print(" 🔍 Enhanced legal entity extraction")
    print(" 🕸️ Knowledge Graph construction")
    print(" 📊 Graph RAG integration")
    print(" 💾 Persistent storage")
    print(" 🛡️ Error handling and recovery")
    
    if args.stats:
        stats = creator.get_system_stats()
        print("\n📊 System Statistics:")
        print(json.dumps(stats, indent=2))
        return
    
    # Process directory
    print(f"\n🔄 Processing directory: {args.directory}")
    print(f"📋 Mode: ONE PDF = ONE CHUNK")
    print(f"🔄 Force reprocess: {args.force}")
    
    results = creator.process_directory(args.directory, args.force)
    
    print(f"\n📊 PROCESSING RESULTS:")
    print(f" 📁 Total files: {results['total_files']}")
    print(f" ✅ Successfully processed: {results['processed_files']}")
    print(f" ❌ Failed: {results['failed_files']}")
    print(f" 📚 Total chunks: {results['total_chunks']} (1 per PDF)")
    print(f" 🕸️ Graph entities: {results.get('total_graph_entities', 0)}")
    
    # Show final stats
    stats = creator.get_system_stats()
    print(f"\n📊 FINAL STATS:")
    print(f" 🗄️ Vector DB points: {stats['storage_stats']['vector_db_vectors']}")
    print(f" 🕸️ Graph nodes: {stats['graph_rag_stats']['knowledge_graph_nodes']}")
    print(f" 🔗 Graph edges: {stats['graph_rag_stats']['knowledge_graph_edges']}")
    print(f" 📁 Mode: {stats['system_info']['chunking_mode']}")
    
    print(f"\n✅ Processing completed! ONE PDF = ONE CHUNK")

if __name__ == "__main__":
    main()
