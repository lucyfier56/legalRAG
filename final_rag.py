import os
import json
import pickle
import logging
import hashlib
import uuid
import gc
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import re
import time

# Core ML/NLP imports
import torch
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Vector Database
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition

# BM25 for hybrid retrieval
from rank_bm25 import BM25Okapi

# LLM for answer generation
from groq import Groq

# Enhanced fuzzy matching and phonetics
from fuzzywuzzy import fuzz, process
import jellyfish
from difflib import SequenceMatcher

# LangChain for AI Agents
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Document processing fallbacks
import fitz # PyMuPDF
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
import tempfile
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class GroqLLM(LLM, BaseModel):
    """Custom LangChain LLM wrapper for Groq"""
    client: Any = Field(default=None, description="Groq client instance")
    model_name: str = Field(default="deepseek-r1-distill-llama-70b", description="Model name to use")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, groq_client: Any, **kwargs):
        super().__init__(client=groq_client, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq LLM call failed: {str(e)}")
            return "Error in LLM response"

    @property
    def _llm_type(self) -> str:
        return "groq"

class EnhancedLegalRAG:
    """Universal Legal RAG System - Dynamic Case Discovery, No Hardcoding"""
    
    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "updated_documents",
                 groq_api_key: str = "",
                 data_dir: str = "legal_data"):
        
        # Core configuration
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.groq_api_key = groq_api_key
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Device setup
        self.device = self._setup_device()
        
        # Core components
        self.qdrant_client = None
        self.embedding_model = None
        self.groq_client = None
        self.groq_llm = None
        self.agent = None
        self.bm25_index = None
        self.documents = []
        self.corpus_tokens = []
        self.processed_files = {}
        
        # Processing configuration
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.max_results = 25
        self.min_content_threshold = 20
        self.min_page_content = 10
        self.max_retries = 3
        
        # Legal entity patterns and types
        self.judge_patterns = [
            r"HON'BLE\s+(?:MR\.|MS\.|MRS\.)?\s*JUSTICE\s+([A-Z][A-Z\.\s]+?)(?:\s+J\.|$)",
            r"JUSTICE\s+([A-Z][A-Z\.\s]+?)(?:\s+J\.|$)",
            r"Hon'ble\s+(?:Mr\.|Ms\.|Mrs\.)?\s*Justice\s+([A-Z][A-Z\.\s]+?)(?:\s+J\.|$)",
            r"HON'BLE\s+(?:MR\.|MS\.|MRS\.)?\s*J\.\s+([A-Z][A-Z\.\s]+?)(?:\s+J\.|$)",
            r"J\.\s+([A-Z][A-Z\.\s]+?)(?:\s+J\.|$)",
            r"JUDGE\s+([A-Z][A-Z\.\s]+?)(?:\s+J\.|$)",
            r"([A-Z][A-Z\.\s]+?)\s+J\."
        ]
        
        self.advocate_patterns = [
            r"(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-zA-Z\s]+),?\s+(?:Advocate|Counsel|Sr\. Advocate|Senior Counsel)",
            r"(?:Advocate|Counsel|Sr\. Advocate|Senior Counsel)\s+(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-zA-Z\s]+)",
            r"for\s+(?:the\s+)?(?:plaintiff|defendant|petitioner|respondent):\s*(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-zA-Z\s]+)",
            r"appearing\s+for.*?(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-zA-Z\s]+)",
            r"represented\s+by.*?(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-zA-Z\s]+)"
        ]
        
        self.phonetic_algorithms = [
            ('soundex', jellyfish.soundex),
            ('metaphone', jellyfish.metaphone),
            ('dmetaphone', lambda x: jellyfish.dmetaphone(x)[0]),
            ('nysiis', jellyfish.nysiis),
            ('match_rating', jellyfish.match_rating_codex)
        ]
        
        self.entity_types = {
            'judges': ['judge', 'justice', 'hon\'ble', 'coram', 'bench', 'honourable', 'judicial'],
            'advocates': ['advocate', 'counsel', 'lawyer', 'sr. advocate', 'senior counsel', 'appearing', 'representing'],
            'plaintiffs': ['plaintiff', 'petitioner', 'appellant', 'applicant', 'claimant'],
            'defendants': ['defendant', 'respondent', 'appellee', 'opposite party'],
            'parties': ['party', 'vs', 'versus', 'against', 'parties'],
            'courts': ['court', 'tribunal', 'commission', 'authority', 'bench'],
            'cases': ['case', 'suit', 'petition', 'application', 'appeal', 'writ'],
            'officials': ['additional solicitor general', 'solicitor general', 'attorney general', 'government pleader'],
            'others': ['amicus curiae', 'intervenor', 'witness', 'expert']
        }
        
        # Unified storage paths
        self._setup_storage_paths()
        self.debug_mode = False
        
        # Initialize system
        self._initialize_system()
        self._setup_ai_agent()

    def _setup_device(self) -> str:
        """Setup optimal device for processing"""
        if torch.backends.mps.is_available():
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def _setup_storage_paths(self):
        """Setup unified storage paths for any collection"""
        collection_data_dir = self.data_dir / self.collection_name
        collection_data_dir.mkdir(exist_ok=True)
        
        self.bm25_index_path = collection_data_dir / "bm25_index.pkl"
        self.documents_path = collection_data_dir / "documents.pkl"
        self.processed_files_path = collection_data_dir / "processed_files.json"

    def _initialize_system(self):
        """Initialize all system components"""
        try:
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            
            # Initialize embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize Groq client
            self.groq_client = Groq(api_key=self.groq_api_key)
            self.groq_llm = GroqLLM(self.groq_client)
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
            # Setup collection
            self._setup_collection()
            
            # Load existing data
            self._load_existing_data()
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
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
            logger.error(f"Collection setup failed: {str(e)}")
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
            
            # Load processed files tracking
            if self.processed_files_path.exists():
                with open(self.processed_files_path, 'r') as f:
                    self.processed_files = json.load(f)
                logger.info(f"✅ Loaded {len(self.processed_files)} processed file records")
                
        except Exception as e:
            logger.warning(f"⚠️ Error loading existing data: {str(e)}")
            self.documents = []
            self.corpus_tokens = []
            self.bm25_index = None
            self.processed_files = {}

    def _get_default_classification(self) -> Dict[str, Any]:
        """Get default classification when parsing fails"""
        return {
            "type": "general",
            "intent": "search",
            "entities": {},
            "requires_aggregation": False,
            "expected_response_type": "text",
            "confidence": 0.5,
            "case_specific": False
        }

    def _setup_ai_agent(self, query: str = None) -> Dict[str, Any]:
        """Setup Universal Legal Entity AI Agent"""
        print("Setting up AI Agent with query:", query)
        try:
            if query is None:
                # ...existing agent setup code...
                return None
            else:
                # Improved classification prompt with better examples and guidance
                classification_prompt = f"""Analyze this legal query and classify it accurately: "{query}"

    Rules for classification:
    1. "type" should be:
    - "case_explanation" for queries about specific cases (e.g., "explain Microsoft vs Apple case")
    - "judge_comprehensive" for queries about judges (e.g., "who is Judge Smith")
    - "entity_in_case" for queries about entities within specific cases
    - "list_search" for queries asking to list or enumerate (e.g., "list all cases", "show all judgments")
    - "general" for other queries

    2. "intent" should be:
    - "list" for listing or enumeration requests
    - "count" for counting requests
    - "search" for general searches
    - "information" for explanatory queries
    - "relationship" for queries about connections

    3. Set "case_specific" to true ONLY when a specific case is mentioned using "vs", "versus", or "case of"

    Format response as JSON:
    {{
        "type": "case_explanation|judge_comprehensive|entity_in_case|list_search|general",
        "intent": "search|count|list|relationship|information",
        "entities": {{
            "case": "case name if mentioned",
            "judge": "judge name if mentioned",
            "entity": "other entity if mentioned"
        }},
        "requires_aggregation": true/false,
        "expected_response_type": "text|list|number",
        "confidence": 0.0-1.0,
        "case_specific": true/false
    }}"""

                try:
                    print("Try block inside setup_ai_agent")
                    response = self.groq_llm._call(classification_prompt)
                    # Clean the response to ensure it contains only JSON
                    json_str = response.strip()
                    # Find JSON content between curly braces
                    start = json_str.find('{')
                    end = json_str.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = json_str[start:end]
                        classification = json.loads(json_str)
                        print("Classification JSON inside setup_ai_agents inside try block", classification)
                        return classification
                    
                    else:
                        logger.error("No valid JSON found in LLM response")
                        return self._get_default_classification()
                except json.JSONDecodeError as e:
                    logger.error(f"Query classification JSON parsing failed: {str(e)}")
                    return self._get_default_classification()
                except Exception as e:
                    logger.error(f"Query classification failed: {str(e)}")
                    return self._get_default_classification()

        except Exception as e:
            logger.error(f"AI Agent setup/classification failed: {str(e)}")
            return self._get_default_classification()


    # ====================================
    # DYNAMIC CASE DISCOVERY - NO HARDCODING
    # ====================================

    def _extract_case_name_enhanced(self, query: str) -> Tuple[str, str]:
        """Dynamic case name extraction - NO HARDCODING"""
        query = query.strip()
        
        # Dynamic case name patterns - works for ANY case format
        case_name_patterns = [
            # Generic "explain/about + case name" patterns
            r'(?:explain|about|regarding|case of|details of)\s+(?:the\s+)?(?:case\s+)?(?:of\s+)?([A-Z][a-zA-Z\s&\.]+?)\s+(?:case|vs?|versus|v\.)',
            
            # Standard "X vs Y" patterns
            r'([A-Z][a-zA-Z\s&\.]+?)\s+vs?\s+([A-Z][a-zA-Z\s&\.]+)',
            r'([A-Z][a-zA-Z\s&\.]+?)\s+versus\s+([A-Z][a-zA-Z\s&\.]+)',
            r'([A-Z][a-zA-Z\s&\.]+?)\s+v\.\s+([A-Z][a-zA-Z\s&\.]+)',
            
            # Generic case references
            r'case\s+(?:of\s+)?([A-Z][a-zA-Z\s&\.]+?)(?:\s+vs?|\s+versus|\s+v\.|\s+case|$)',
            
            # Company/entity name extraction
            r'(?:explain|about|details)\s+([A-Z][a-zA-Z\s&\.]+?)(?:\s+case|$)',
            
            # Single word company names followed by case indicators
            r'(?:explain|about|regarding)\s+([A-Z][a-zA-Z]+)(?:\s+case)?'
        ]
        
        for pattern in case_name_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:  # "X vs Y" format
                    case_name = f"{match.group(1).strip()} vs {match.group(2).strip()}"
                    original_term = case_name
                elif len(match.groups()) == 1:  # Single entity
                    case_name = match.group(1).strip()
                    original_term = case_name
                    # Try to find full case name from documents
                    full_case_name = self._discover_full_case_name(case_name)
                    if full_case_name:
                        case_name = full_case_name
                else:
                    continue
                
                # Clean up the case name
                case_name = re.sub(r'\b(explain|about|case|of|the|details)\b', '', case_name, flags=re.IGNORECASE).strip()
                
                if len(case_name) > 2:
                    return case_name, original_term
        
        return "", query

    def _discover_full_case_name(self, partial_name: str) -> Optional[str]:
        """Dynamically discover full case name from available documents"""
        try:
            all_source_names = self._get_all_source_names()
            
            # Find best matching source names that contain the partial name
            partial_lower = partial_name.lower()
            potential_matches = []
            
            for source_name in all_source_names:
                source_lower = source_name.lower()
                
                # Check if partial name is in source name
                if partial_lower in source_lower:
                    # Extract full case name from source
                    vs_patterns = [
                        r'([A-Z][a-zA-Z\s&\.]+?)\s+vs?\s+([A-Z][a-zA-Z\s&\.]+)',
                        r'([A-Z][a-zA-Z\s&\.]+?)\s+versus\s+([A-Z][a-zA-Z\s&\.]+)',
                        r'([A-Z][a-zA-Z\s&\.]+?)\s+v\.\s+([A-Z][a-zA-Z\s&\.]+)'
                    ]
                    
                    for pattern in vs_patterns:
                        match = re.search(pattern, source_name, re.IGNORECASE)
                        if match:
                            full_case = f"{match.group(1).strip()} vs {match.group(2).strip()}"
                            score = fuzz.partial_ratio(partial_lower, full_case.lower())
                            potential_matches.append((full_case, score))
                            break
                    else:
                        # If no "vs" pattern, use the source name itself
                        score = fuzz.partial_ratio(partial_lower, source_lower)
                        potential_matches.append((source_name, score))
            
            # Return best match
            if potential_matches:
                best_match = max(potential_matches, key=lambda x: x[1])
                if best_match[1] >= 70:  # Minimum similarity threshold
                    return best_match[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Case discovery failed: {str(e)}")
            return None

    def _get_all_source_names(self) -> List[str]:
        """Get all source names from the collection"""
        try:
            all_sources = set()
            offset = None
            
            while True:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=1000,
                        offset=offset,
                        with_payload=True
                    )
                except Exception as e:
                    logger.error(f"Error scrolling collection: {str(e)}")
                    break
                
                points, next_offset = scroll_result
                if not points:
                    break
                
                for point in points:
                    source_name = point.payload.get("source_name", "")
                    if source_name:
                        all_sources.add(source_name)
                
                offset = next_offset
                if not offset:
                    break
            
            return list(all_sources)
            
        except Exception as e:
            logger.error(f"Failed to get source names: {str(e)}")
            return []

    def _get_case_keywords(self, case_name: str) -> List[str]:
        """Dynamically extract keywords from case name - NO HARDCODING"""
        if not case_name:
            return []
        
        case_name_lower = case_name.lower()
        keywords = []
        
        # Extract company/entity names from "X vs Y" format
        vs_patterns = [
            r'([a-zA-Z\s&\.]+?)\s+vs?\s+([a-zA-Z\s&\.]+)',
            r'([a-zA-Z\s&\.]+?)\s+versus\s+([a-zA-Z\s&\.]+)',
            r'([a-zA-Z\s&\.]+?)\s+v\.\s+([a-zA-Z\s&\.]+)'
        ]
        
        for pattern in vs_patterns:
            match = re.search(pattern, case_name_lower)
            if match:
                # Add both parties as keywords
                party1 = match.group(1).strip()
                party2 = match.group(2).strip()
                
                # Extract meaningful words (remove common legal terms)
                stopwords = {'pvt', 'ltd', 'limited', 'private', 'company', 'corp', 'corporation', 
                            'inc', 'incorporated', 'co', 'enterprises', 'technologies', 'india'}
                
                for party in [party1, party2]:
                    words = party.split()
                    meaningful_words = [w for w in words if w.lower() not in stopwords and len(w) > 2]
                    keywords.extend(meaningful_words)
                
                # Add the full party names as well
                keywords.extend([party1, party2])
                break
        else:
            # If no "vs" pattern, extract words from the case name
            words = case_name_lower.split()
            stopwords = {'case', 'vs', 'versus', 'v', 'pvt', 'ltd', 'limited', 'private', 
                        'company', 'corp', 'corporation', 'inc', 'incorporated', 'co'}
            meaningful_words = [w for w in words if w not in stopwords and len(w) > 2]
            keywords.extend(meaningful_words)
        
        # Remove duplicates and return
        return list(set(keywords))

    def _filter_documents_by_case(self, results: List[Dict[str, Any]], case_name: str, strict_mode: bool = True) -> List[Dict[str, Any]]:
        """Filter documents to focus only on the specific case mentioned"""
        if not case_name:
            return results
        
        case_keywords = self._get_case_keywords(case_name)
        filtered_results = []
        
        for result in results:
            source_name = result.get("source_name", "").lower()
            text_content = result.get("text", result.get("content", "")).lower()
            
            # Check if document is from the specific case
            case_match_score = 0
            
            # Primary matching - source name contains case keywords
            for keyword in case_keywords:
                if keyword.lower() in source_name:
                    case_match_score += 3
            
            # Secondary matching - text contains case keywords
            for keyword in case_keywords:
                if keyword.lower() in text_content:
                    case_match_score += 1
            
            # In strict mode, only include documents with high case match scores
            if strict_mode:
                if case_match_score >= 3:  # Must match source name
                    filtered_results.append(result)
            else:
                if case_match_score >= 1:  # Can match text content
                    filtered_results.append(result)
        
        return filtered_results

    def _case_specific_search(self, query: str, case_name: str) -> List[Dict[str, Any]]:
        """Perform case-specific search with high precision"""
        try:
            case_keywords = self._get_case_keywords(case_name)
            all_results = []
            processed_docs = set()
            
            print("Performing case-specific search for:", case_name)
            print("--------------------------------------------------------------")
            print("case_keywords:", case_keywords)
            print("--------------------------------------------------------------")

            # Search with case-specific query construction
            search_query = f"{case_name} {' '.join(case_keywords)}"
            
            # Vector search with case focus
            vector_results = self._vector_search(search_query)
            case_filtered_vector = self._filter_documents_by_case(vector_results, case_name, strict_mode=True)
            all_results.extend(case_filtered_vector)
            
            # BM25 search with case focus  
            bm25_results = self._bm25_search(search_query)
            case_filtered_bm25 = self._filter_documents_by_case(bm25_results, case_name, strict_mode=True)
            all_results.extend(case_filtered_bm25)
            
            # Direct document search for case-specific documents
            offset = None
            while True:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=100000,
                        offset=offset,
                        with_payload=True
                    )
                except Exception as e:
                    logger.error(f"Error scrolling collection: {str(e)}")
                    break
                
                points, next_offset = scroll_result
                if not points:
                    break
                
                for point in points:
                    source_name = point.payload.get("source_name", "").lower()
                    text = point.payload.get("text", point.payload.get("content", ""))
                    
                    # Check if this document is from the specific case
                    if any(keyword.lower() in source_name for keyword in case_keywords):
                        doc_key = source_name
                        if doc_key not in processed_docs:
                            all_results.append({
                                "id": point.id,
                                "text": text,
                                "content": text,
                                "source": point.payload.get("source", ""),
                                "source_name": point.payload.get("source_name", ""),
                                "page": point.payload.get("page", 1),
                                "chunk_index": point.payload.get("chunk_index", 0),
                                "method": "case_specific_direct",
                                "score": 0.95,  # High score for direct case match
                                "collection_name": self.collection_name
                            })
                            processed_docs.add(doc_key)
                
                offset = next_offset
                if not offset:
                    break
            
            # Remove duplicates and sort by relevance
            unique_results = self._deduplicate_results(all_results)
            
            # Sort by case relevance
            def case_relevance_score(result):
                source_name = result.get("source_name", "").lower()
                base_score = result.get("score", 0)
                
                # Boost score for exact case matches
                for keyword in case_keywords:
                    if keyword.lower() in source_name:
                        base_score += 0.5
                        
                return base_score
            
            unique_results.sort(key=case_relevance_score, reverse=True)
            print(all_results)
            return all_results
            
        except Exception as e:
            logger.error(f"Case-specific search failed: {str(e)}")
            return []

    def _extract_case_from_context(self, query: str) -> Optional[str]:
        """Extract case name from query context dynamically"""
        # Look for patterns like "judge for X case" or "advocate in Y vs Z"
        context_patterns = [
            r'(?:judge|justice|advocate|counsel)\s+(?:for|in|of)\s+(?:the\s+)?([A-Z][a-zA-Z\s&\.]+?)\s+(?:case|vs|versus)',
            r'(?:judge|justice|advocate|counsel)\s+(?:for|in|of)\s+([A-Z][a-zA-Z\s&\.]+?)\s+vs?\s+([A-Z][a-zA-Z\s&\.]+)',
            r'([A-Z][a-zA-Z\s&\.]+?)\s+(?:case|matter).*?(?:judge|justice|advocate|counsel)'
        ]
        
        for pattern in context_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:  # "X vs Y" format
                    return f"{match.group(1).strip()} vs {match.group(2).strip()}"
                elif len(match.groups()) == 1:  # Single entity
                    return match.group(1).strip()
        
        return None

    # ====================================
    # ENHANCED QUERY CLASSIFICATION
    # ====================================

    def enhanced_query_classification(self, query: str, previous_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced query classification with dynamic case detection"""
        try:
            # Dynamic case detection
            case_name, original_term = self._extract_case_name_enhanced(query)
            
            if case_name:
                return {
                    "type": "case_explanation",
                    "intent": "information", 
                    "entities": {"case": case_name},
                    "original_entities": {"case": original_term},
                    "requires_aggregation": False,
                    "expected_response_type": "text",
                    "keywords": self._get_case_keywords(case_name),
                    "confidence": 0.95,
                    "original_query": query,
                    "is_followup": False,
                    "case_specific": True
                }
            
            # Dynamic judge/advocate detection with case context
            if any(word in query.lower() for word in ["judge", "justice", "advocate", "counsel"]):
                # Try to extract case context dynamically
                potential_case = self._extract_case_from_context(query)
                if potential_case:
                    return {
                        "type": "entity_in_case",
                        "intent": "information",
                        "entities": {
                            "case": potential_case,
                            "entity_type": "judge" if "judge" in query.lower() or "justice" in query.lower() else "advocate"
                        },
                        "original_entities": {"case": potential_case},
                        "requires_aggregation": False,
                        "expected_response_type": "text",
                        "keywords": self._get_case_keywords(potential_case) + ["judge" if "judge" in query.lower() else "advocate"],
                        "confidence": 0.9,
                        "original_query": query,
                        "is_followup": False,
                        "case_specific": True
                    }
            
            # Fallback to general classification
            return self._fallback_classification(query, previous_context)
            
        except Exception as e:
            logger.error(f"Enhanced classification failed: {str(e)}")
            return self._fallback_classification(query, previous_context)

    # ====================================
    # ENTITY SEARCH METHODS
    # ====================================

    def _search_entities(self, query: str) -> str:
        """Unified entity search for all types"""
        try:
            # Extract search term
            search_term = self._extract_search_term(query)
            
            # Multi-stage search strategy
            results = []
            
            # Stage 1: Vector search
            vector_results = self._vector_search(search_term)
            results.extend(vector_results)
            
            # Stage 2: BM25 search
            bm25_results = self._bm25_search(search_term)
            results.extend(bm25_results)
            
            # Stage 3: Pattern-based search
            pattern_results = self._pattern_search(search_term)
            results.extend(pattern_results)
            
            # Stage 4: Fuzzy search
            fuzzy_results = self._fuzzy_search(search_term)
            results.extend(fuzzy_results)
            
            # Stage 5: Phonetic search
            phonetic_results = self._phonetic_search(search_term)
            results.extend(phonetic_results)
            
            # Deduplicate and sort
            unique_results = self._deduplicate_results(results)
            sorted_results = sorted(unique_results, 
                                  key=lambda x: x.get("score", 0), reverse=True)
            
            return json.dumps({
                "results": sorted_results[:self.max_results],
                "total_found": len(sorted_results),
                "search_term": search_term,
                "method": "unified_multi_stage"
            })
            
        except Exception as e:
            logger.error(f"Entity search failed: {str(e)}")
            return json.dumps({"results": [], "error": str(e)})

    def _classify_entities(self, query: str) -> str:
        """Unified entity classification"""
        classification_prompt = f"""
Analyze this legal query and identify ALL legal entities mentioned.

Query: "{query}"

Return JSON with this structure:
{{
    "primary_entity_type": "judges|advocates|plaintiffs|defendants|parties|courts|cases|officials|others|multi_entity",
    "entities_found": {{
        "judges": ["list of judge names"],
        "advocates": ["list of advocate names"],
        "plaintiffs": ["list of plaintiff names"],
        "defendants": ["list of defendant names"],
        "parties": ["list of party names"],
        "courts": ["list of court names"],
        "cases": ["list of case numbers/names"],
        "officials": ["list of official titles"],
        "others": ["list of other entities"]
    }},
    "query_intent": "search|count|list|relationship|compare|information",
    "confidence": 0.0-1.0
}}

Return ONLY the JSON object.
"""
        
        try:
           
            response = self.groq_llm._call(classification_prompt)
            print(response)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json_match.group(0)
            else:
                return json.dumps({
                    "primary_entity_type": "general",
                    "entities_found": {},
                    "query_intent": "search",
                    "confidence": 0.5
                })
        except Exception as e:
            logger.error(f"Entity classification failed: {str(e)}")
            return json.dumps({
                "primary_entity_type": "general",
                "entities_found": {},
                "query_intent": "search",
                "confidence": 0.3
            })

    def _find_entity_matches(self, entity_name: str) -> str:
        """Unified entity name matching"""
        try:
            all_entities = self._get_all_entities()
            matches = []
            
            # Exact matching
            exact_matches = [name for name in all_entities if name.lower() == entity_name.lower()]
            matches.extend([(name, 1.0, "exact") for name in exact_matches])
            
            # Fuzzy matching
            fuzzy_matches = process.extract(entity_name, all_entities, limit=15, scorer=fuzz.token_sort_ratio)
            for name, score in fuzzy_matches:
                if score >= 70:
                    matches.append((name, score/100.0, "fuzzy"))
            
            # Partial matching
            partial_matches = process.extract(entity_name, all_entities, limit=12, scorer=fuzz.partial_ratio)
            for name, score in partial_matches:
                if score >= 80:
                    matches.append((name, score/100.0, "partial"))
            
            # Remove duplicates and sort
            unique_matches = {}
            for name, score, method in matches:
                if name not in unique_matches or score > unique_matches[name][0]:
                    unique_matches[name] = (score, method)
            
            sorted_matches = sorted(unique_matches.items(), key=lambda x: x[1][0], reverse=True)
            
            best_matches = [
                {"name": name, "score": score, "method": method}
                for name, (score, method) in sorted_matches[:15]
            ]
            
            return json.dumps(best_matches)
            
        except Exception as e:
            logger.error(f"Entity matching failed: {str(e)}")
            return json.dumps([])

    def _find_relationships(self, query: str) -> str:
        """Unified relationship finding"""
        print("unified relationship finding")
        try:
            # Classify to find entities
            classification_json = self._classify_entities(query)
            classification = json.loads(classification_json)
            print(classification_json)
            
            entities_found = classification.get("entities_found", {})
            
            # Find documents containing multiple entities
            entity_results = {}
            for entity_type, entity_names in entities_found.items():
                if entity_names:
                    entity_results[entity_type] = []
                    for entity_name in entity_names:
                        results = self._search_by_entity_type(entity_name, entity_type)
                        entity_results[entity_type].extend(results)
            
            # Cross-reference entities in same documents
            common_documents = defaultdict(set)
            for entity_type, results in entity_results.items():
                for result in results:
                    doc_name = result.get("source_name", "")
                    if doc_name:
                        common_documents[doc_name].add(entity_type)
            
            # Find documents with multiple entity types
            multi_entity_docs = {doc: entities for doc, entities in common_documents.items() if len(entities) > 1}
            
            relationship_analysis = {
                "entities_found": entities_found,
                "common_documents": dict(multi_entity_docs),
                "total_relationships": len(multi_entity_docs)
            }
            
            return json.dumps(relationship_analysis)
            
        except Exception as e:
            logger.error(f"Relationship analysis failed: {str(e)}")
            return json.dumps({"error": str(e)})

    # ====================================
    # CORE SEARCH METHODS - DYNAMIC
    # ====================================

    def _extract_search_term(self, query: str) -> str:
        """Extract search term from any query type with dynamic detection"""
        query = query.strip()
        
        # Dynamic case name detection patterns
        case_patterns = [
            r'(?:explain|about|regarding|information about|details of|case of)\s+([A-Z][a-zA-Z\s&\.]+?)\s+(?:case|vs?|versus|v\.)',
            r'([A-Z][a-zA-Z\s&\.]+?)\s+vs?\s+([A-Z][a-zA-Z\s&\.]+)',
            r'([A-Z][a-zA-Z\s&\.]+?)\s+versus\s+([A-Z][a-zA-Z\s&\.]+)',
            r'([A-Z][a-zA-Z\s&\.]+?)\s+v\.\s+([A-Z][a-zA-Z\s&\.]+)',
            r'case\s+(?:of\s+)?([A-Z][a-zA-Z\s&\.]+?)(?:\s+vs?|\s+versus|\s+v\.|\s+case|$)',
            r'(?:explain|about|details)\s+([A-Z][a-zA-Z\s&\.]+?)(?:\s+case|$)'
        ]
        
        # First try case-specific patterns
        for pattern in case_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if match.groups():
                    # For vs. patterns, combine both parties
                    if len(match.groups()) == 2:
                        return f"{match.group(1).strip()} vs {match.group(2).strip()}"
                    else:
                        return match.group(1).strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "who is judge ", "who is justice ", "who was judge ", "who was justice ",
            "find judge ", "find justice ", "search judge ", "search justice ",
            "about judge ", "about justice ", "tell me about judge ", "tell me about justice ",
            "information about judge ", "information about justice ",
            "who is ", "who was ", "find ", "search ", "about ", "tell me about ",
            "information about ", "show me ", "list all ", "count ",
            "explain ", "details of ", "regarding ", "case of "
        ]
        
        query_lower = query.lower()
        for prefix in prefixes_to_remove:
            if query_lower.startswith(prefix):
                query = query[len(prefix):].strip()
                break
        
        # Extract using patterns for all entity types
        patterns = [
            r'judge\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z\.]*)*?)',
            r'justice\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z\.]*)*?)',
            r'hon\'ble\s+justice\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z\.]*)*?)',
            r'advocate\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z\.]*)*?)',
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z\.]*)*?)\s+(?:j\.|judge|justice|advocate)',
            r'case\s+(?:no\.?\s*)?([A-Z0-9\/\-]+)',
            r'([A-Z][a-zA-Z\s\&\.]+?)\s+vs?\s+([A-Z][a-zA-Z\s\&\.]+)',
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z\.]*)*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 2:
                    return extracted
        
        return query

    def _vector_search(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced vector search with dynamic query construction"""
        try:
            # Dynamic query construction based on content
            search_query = f"legal {query} judge justice advocate case court"
            query_embedding = self.embedding_model.embed_query(search_query)
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=self.max_results,
                with_payload=True
            )
            
            results = []
            for result in search_results:
                if result.score >= 0.3:  # Lowered threshold for better recall
                    results.append({
                        "id": result.id,
                        "text": result.payload.get("text", result.payload.get("content", "")),
                        "content": result.payload.get("text", result.payload.get("content", "")),
                        "source": result.payload.get("source", ""),
                        "source_name": result.payload.get("source_name", ""),
                        "page": result.payload.get("page", 1),
                        "chunk_index": result.payload.get("chunk_index", 0),
                        "score": result.score,
                        "method": "vector_search",
                        "collection_name": self.collection_name
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def _bm25_search(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced BM25 search"""
        try:
            if not self.bm25_index or not self.corpus_tokens:
                return []
            
            query_tokens = self._tokenize_text(query)
            if not query_tokens:
                return []
            
            scores = self.bm25_index.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:self.max_results]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents) and scores[idx] > 0:
                    doc = self.documents[idx]
                    results.append({
                        "text": doc.page_content,
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "source_name": Path(doc.metadata.get("source", "")).stem,
                        "page": doc.metadata.get("page", 1),
                        "score": float(scores[idx]),
                        "method": "bm25",
                        "collection_name": self.collection_name
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            return []

    def _pattern_search(self, search_term: str) -> List[Dict[str, Any]]:
        """Enhanced pattern-based search with dynamic patterns"""
        try:
            # Dynamic pattern generation based on search term
            all_patterns = {
                'judges': [
                    r"HON'BLE\s+(?:MR\.|MS\.|MRS\.)?\s*JUSTICE\s+" + re.escape(search_term),
                    r"CORAM:\s*.*?" + re.escape(search_term),
                    r"BEFORE.*?" + re.escape(search_term),
                    r"JUDGE\s+" + re.escape(search_term),
                    r"J\.\s*" + re.escape(search_term)
                ],
                'advocates': [
                    r"(?:Mr\.|Ms\.|Mrs\.)\s+" + re.escape(search_term) + r",?\s+(?:Advocate|Counsel|Sr\. Advocate|Senior Counsel)",
                    r"(?:Advocate|Counsel|Sr\. Advocate|Senior Counsel)\s+(?:Mr\.|Ms\.|Mrs\.)\s+" + re.escape(search_term),
                    r"for\s+(?:the\s+)?(?:plaintiff|defendant|petitioner|respondent):\s*" + re.escape(search_term),
                    r"appearing\s+for.*?" + re.escape(search_term),
                    r"represented\s+by.*?" + re.escape(search_term)
                ],
                'parties': [
                    r"PLAINTIFF[:\s]*" + re.escape(search_term),
                    r"PETITIONER[:\s]*" + re.escape(search_term),
                    r"APPELLANT[:\s]*" + re.escape(search_term),
                    r"DEFENDANT[:\s]*" + re.escape(search_term),
                    r"RESPONDENT[:\s]*" + re.escape(search_term)
                ],
                'cases': [
                    r"CASE\s+NO\.?\s*" + re.escape(search_term),
                    r"SUIT\s+NO\.?\s*" + re.escape(search_term),
                    r"PETITION\s+NO\.?\s*" + re.escape(search_term)
                ]
            }
            
            # Combine all patterns
            combined_patterns = []
            for pattern_list in all_patterns.values():
                combined_patterns.extend([p for p in pattern_list if p])
            
            return self._search_with_patterns(combined_patterns, search_term)
            
        except Exception as e:
            logger.error(f"Pattern search failed: {str(e)}")
            return []

    def _fuzzy_search(self, search_term: str) -> List[Dict[str, Any]]:
        """Enhanced fuzzy search with dynamic matching"""
        try:
            results = []
            processed_sources = set()
            offset = None
            
            while True:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=1000,
                        offset=offset,
                        with_payload=True
                    )
                except Exception as e:
                    logger.error(f"Error scrolling collection: {str(e)}")
                    break
                
                points, next_offset = scroll_result
                if not points:
                    break
                
                for point in points:
                    source_name = point.payload.get("source_name", "")
                    if source_name in processed_sources:
                        continue
                    
                    text = point.payload.get("text", point.payload.get("content", ""))
                    
                    # Dynamic fuzzy matching
                    source_ratio = fuzz.partial_ratio(search_term.lower(), source_name.lower()) if source_name else 0
                    text_ratio = fuzz.token_sort_ratio(search_term.lower(), text.lower())
                    
                    if source_ratio >= 70:  # Source name match
                        results.append({
                            "id": point.id,
                            "text": text,
                            "content": text,
                            "source": point.payload.get("source", ""),
                            "source_name": source_name,
                            "page": point.payload.get("page", 1),
                            "chunk_index": point.payload.get("chunk_index", 0),
                            "method": "fuzzy_source",
                            "fuzzy_score": source_ratio,
                            "score": source_ratio / 100.0,
                            "collection_name": self.collection_name
                        })
                        processed_sources.add(source_name)
                    elif text_ratio >= 60:  # Text content match
                        results.append({
                            "id": point.id,
                            "text": text,
                            "content": text,
                            "source": point.payload.get("source", ""),
                            "source_name": source_name,
                            "page": point.payload.get("page", 1),
                            "chunk_index": point.payload.get("chunk_index", 0),
                            "method": "fuzzy_content",
                            "fuzzy_score": text_ratio,
                            "score": text_ratio / 100.0,
                            "collection_name": self.collection_name
                        })
                        processed_sources.add(source_name)
                
                offset = next_offset
                if not offset:
                    break
            
            results.sort(key=lambda x: x.get("fuzzy_score", 0), reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Fuzzy search failed: {str(e)}")
            return []

    def _phonetic_search(self, search_term: str) -> List[Dict[str, Any]]:
        """Enhanced phonetic search"""
        try:
            # Get phonetic representations
            phonetic_codes = {}
            for name, algorithm in self.phonetic_algorithms:
                try:
                    code = algorithm(search_term)
                    if code:
                        phonetic_codes[name] = code
                except:
                    continue
            
            if not phonetic_codes:
                return []
            
            results = []
            offset = None
            
            while True:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=1000,
                        offset=offset,
                        with_payload=True
                    )
                except Exception as e:
                    logger.error(f"Error scrolling collection: {str(e)}")
                    break
                
                points, next_offset = scroll_result
                if not points:
                    break
                
                for point in points:
                    text = point.payload.get("text", point.payload.get("content", ""))
                    
                    # Extract potential names from text using all patterns
                    all_patterns = self.judge_patterns + self.advocate_patterns + [
                        r"(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-zA-Z\s]+)",
                        r"([A-Z][a-zA-Z]+)\s+(?:Advocate|Counsel)",
                        r"([A-Z][a-zA-Z\s]+)\s+vs?\s+",
                        r"([A-Z][a-zA-Z\s]+)(?:\s+J\.|\s+Judge|\s+Justice)"
                    ]
                    
                    for pattern in all_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        for match in matches:
                            if isinstance(match, tuple):
                                match = match[0] if match else ""
                            
                            match_cleaned = re.sub(r'[^\w\s]', '', match).strip()
                            if len(match_cleaned) > 2:
                                # Check phonetic similarity
                                for name, algorithm in self.phonetic_algorithms:
                                    try:
                                        match_code = algorithm(match_cleaned)
                                        if match_code and name in phonetic_codes:
                                            if match_code == phonetic_codes[name]:
                                                results.append({
                                                    "id": point.id,
                                                    "text": text,
                                                    "content": text,
                                                    "source": point.payload.get("source", ""),
                                                    "source_name": point.payload.get("source_name", ""),
                                                    "page": point.payload.get("page", 1),
                                                    "chunk_index": point.payload.get("chunk_index", 0),
                                                    "method": f"phonetic_{name}",
                                                    "matched_name": match_cleaned,
                                                    "phonetic_match": True,
                                                    "score": 0.8,
                                                    "collection_name": self.collection_name
                                                })
                                                break
                                    except:
                                        continue
                
                offset = next_offset
                if not offset:
                    break
            
            # Remove duplicates based on source_name
            unique_results = {}
            for result in results:
                key = result.get("source_name", "")
                if key not in unique_results or result.get("score", 0) > unique_results[key].get("score", 0):
                    unique_results[key] = result
            
            return list(unique_results.values())
            
        except Exception as e:
            logger.error(f"Phonetic search failed: {str(e)}")
            return []

    def _search_with_patterns(self, patterns: List[str], search_term: str) -> List[Dict[str, Any]]:
        """Enhanced pattern search with better validation"""
        try:
            all_results = []
            processed_docs = set()
            offset = None
            
            while True:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=1000,
                        offset=offset,
                        with_payload=True
                    )
                except Exception as e:
                    logger.error(f"Error scrolling collection: {str(e)}")
                    break
                
                points, next_offset = scroll_result
                if not points:
                    break
                
                for point in points:
                    text = point.payload.get("text", point.payload.get("content", ""))
                    source_name = point.payload.get("source_name", "")
                    
                    if not source_name:
                        source = point.payload.get("source", "")
                        source_name = Path(source).stem if source else ""
                    
                    doc_key = source_name
                    if doc_key in processed_docs:
                        continue
                    
                    # Check if any pattern matches
                    pattern_matched = False
                    for pattern in patterns:
                        if pattern and re.search(pattern, text, re.IGNORECASE):
                            pattern_matched = True
                            break
                    
                    # Also check for general presence validation
                    if not pattern_matched:
                        if self._validate_presence(text, source_name, search_term):
                            pattern_matched = True
                    
                    if pattern_matched:
                        confidence = self._calculate_confidence(text, search_term, source_name)
                        if confidence >= 1.5:  # Lowered threshold
                            all_results.append({
                                "id": point.id,
                                "text": text,
                                "content": text,
                                "source": point.payload.get("source", ""),
                                "source_name": source_name,
                                "page": point.payload.get("page", 1),
                                "chunk_index": point.payload.get("chunk_index", 0),
                                "method": "pattern_enhanced",
                                "match_confidence": confidence,
                                "score": confidence,
                                "matched_term": search_term,
                                "collection_name": self.collection_name
                            })
                            processed_docs.add(doc_key)
                
                offset = next_offset
                if not offset:
                    break
            
            all_results.sort(key=lambda x: x.get("match_confidence", 0), reverse=True)
            return all_results
            
        except Exception as e:
            logger.error(f"Pattern search failed: {str(e)}")
            return []

    def _search_by_entity_type(self, entity_name: str, entity_type: str) -> List[Dict[str, Any]]:
        """Search for specific entity type with dynamic patterns"""
        try:
            # Get patterns for entity type
            patterns = {
                'judges': [
                    r"HON'BLE\s+(?:MR\.|MS\.|MRS\.)?\s*JUSTICE\s+" + re.escape(entity_name),
                    r"CORAM:\s*.*?" + re.escape(entity_name),
                    r"BEFORE.*?" + re.escape(entity_name),
                    r"JUDGE\s+" + re.escape(entity_name),
                    r"J\.\s*" + re.escape(entity_name)
                ],
                'advocates': [
                    r"(?:Mr\.|Ms\.|Mrs\.)\s+" + re.escape(entity_name) + r",?\s+(?:Advocate|Counsel|Sr\. Advocate|Senior Counsel)",
                    r"(?:Advocate|Counsel|Sr\. Advocate|Senior Counsel)\s+(?:Mr\.|Ms\.|Mrs\.)\s+" + re.escape(entity_name),
                    r"for\s+(?:the\s+)?(?:plaintiff|defendant|petitioner|respondent):\s*" + re.escape(entity_name),
                    r"appearing\s+for.*?" + re.escape(entity_name),
                    r"represented\s+by.*?" + re.escape(entity_name)
                ],
                'plaintiffs': [
                    r"PLAINTIFF[:\s]*" + re.escape(entity_name),
                    r"PETITIONER[:\s]*" + re.escape(entity_name),
                    r"APPELLANT[:\s]*" + re.escape(entity_name),
                    r"CLAIMANT[:\s]*" + re.escape(entity_name)
                ],
                'defendants': [
                    r"DEFENDANT[:\s]*" + re.escape(entity_name),
                    r"RESPONDENT[:\s]*" + re.escape(entity_name),
                    r"APPELLEE[:\s]*" + re.escape(entity_name),
                    r"OPPOSITE\s+PARTY[:\s]*" + re.escape(entity_name)
                ],
                'courts': [
                    r"(?:HIGH\s+)?COURT\s+OF\s+" + re.escape(entity_name),
                    r"SUPREME\s+COURT.*?" + re.escape(entity_name),
                    r"TRIBUNAL.*?" + re.escape(entity_name),
                    r"COMMISSION.*?" + re.escape(entity_name)
                ],
                'cases': [
                    r"CASE\s+NO\.?\s*" + re.escape(entity_name),
                    r"SUIT\s+NO\.?\s*" + re.escape(entity_name),
                    r"PETITION\s+NO\.?\s*" + re.escape(entity_name),
                    r"APPEAL\s+NO\.?\s*" + re.escape(entity_name)
                ],
                'officials': [
                    r"ADDITIONAL\s+SOLICITOR\s+GENERAL.*?" + re.escape(entity_name),
                    r"ATTORNEY\s+GENERAL.*?" + re.escape(entity_name),
                    r"GOVERNMENT\s+PLEADER.*?" + re.escape(entity_name),
                    r"SOLICITOR\s+GENERAL.*?" + re.escape(entity_name)
                ]
            }
            
            entity_patterns = patterns.get(entity_type, [])
            entity_patterns = [p for p in entity_patterns if p]  # Filter out empty patterns
            
            if not entity_patterns:
                # Fallback to general search
                return self._comprehensive_search(entity_name)
            
            return self._search_with_patterns(entity_patterns, entity_name)
            
        except Exception as e:
            logger.error(f"Entity type search failed: {str(e)}")
            return []

    # ====================================
    # DYNAMIC VALIDATION METHODS
    # ====================================

    def _validate_presence(self, text: str, source_name: str, entity_name: str) -> bool:
        """Enhanced presence validation with dynamic logic"""
        return (self._validate_exact_presence(text, source_name, entity_name) or
                self._validate_fuzzy_presence(text, source_name, entity_name) or
                self._validate_pattern_presence(text, source_name, entity_name) or
                self._validate_phonetic_presence(text, source_name, entity_name) or
                self._validate_partial_presence(text, source_name, entity_name))

    def _validate_exact_presence(self, text: str, source_name: str, entity_name: str) -> bool:
        """Validate exact presence"""
        text_lower = text.lower()
        source_lower = source_name.lower()
        entity_lower = entity_name.lower()
        return (entity_lower in text_lower or entity_lower in source_lower)

    def _validate_fuzzy_presence(self, text: str, source_name: str, entity_name: str) -> bool:
        """Validate fuzzy presence"""
        text_lower = text.lower()
        source_lower = source_name.lower()
        entity_lower = entity_name.lower()
        return (fuzz.partial_ratio(entity_lower, text_lower) >= 70 or
                fuzz.partial_ratio(entity_lower, source_lower) >= 70)

    def _validate_pattern_presence(self, text: str, source_name: str, entity_name: str) -> bool:
        """Validate pattern presence"""
        all_patterns = self.judge_patterns + self.advocate_patterns
        for pattern in all_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                if fuzz.partial_ratio(entity_name.lower(), match.lower()) >= 70:
                    return True
        return False

    def _validate_phonetic_presence(self, text: str, source_name: str, entity_name: str) -> bool:
        """Validate phonetic presence"""
        try:
            entity_soundex = jellyfish.soundex(entity_name)
            
            # Check source name
            if source_name:
                source_parts = source_name.split()
                for part in source_parts:
                    if len(part) > 2:
                        try:
                            if jellyfish.soundex(part) == entity_soundex:
                                return True
                        except:
                            continue
            
            # Check text content
            words = re.findall(r'\b[A-Za-z]{3,}\b', text)
            for word in words:
                try:
                    if jellyfish.soundex(word) == entity_soundex:
                        return True
                except:
                    continue
            
            return False
            
        except Exception:
            return False

    def _validate_partial_presence(self, text: str, source_name: str, entity_name: str) -> bool:
        """Validate partial presence"""
        text_lower = text.lower()
        source_lower = source_name.lower()
        entity_parts = entity_name.lower().split()
        
        if len(entity_parts) > 1:
            parts_found = 0
            for part in entity_parts:
                if len(part) > 2 and (part in text_lower or part in source_lower):
                    parts_found += 1
            return parts_found >= len(entity_parts) / 2
        
        return False

    def _calculate_confidence(self, text: str, search_term: str, source_name: str = "") -> float:
        """Enhanced confidence calculation with dynamic logic"""
        confidence = 0.0
        text_lower = text.lower()
        search_term_lower = search_term.lower()
        source_lower = source_name.lower()
        
        # Exact matches
        if search_term_lower in source_lower:
            confidence += 10.0
        elif search_term_lower in text_lower:
            confidence += 8.0
        
        # Fuzzy matches
        fuzzy_score = max(
            fuzz.partial_ratio(search_term_lower, text_lower),
            fuzz.partial_ratio(search_term_lower, source_lower)
        ) / 100.0
        
        confidence += fuzzy_score * 5.0
        
        # Pattern matches
        all_patterns = self.judge_patterns + self.advocate_patterns
        for pattern in all_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                pattern_score = fuzz.partial_ratio(search_term_lower, match.lower()) / 100.0
                confidence += pattern_score * 3.0
        
        # Phonetic matches
        try:
            entity_soundex = jellyfish.soundex(search_term)
            words = re.findall(r'\b[A-Za-z]{3,}\b', text + " " + source_name)
            for word in words:
                try:
                    if jellyfish.soundex(word) == entity_soundex:
                        confidence += 2.0
                        break
                except:
                    continue
        except:
            pass
        
        return min(confidence, 15.0)

    # ====================================
    # UTILITY METHODS
    # ====================================

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on source and page"""
        unique_results = {}
        for result in results:
            key = f"{result.get('source_name', '')}_{result.get('page', 1)}"
            if key not in unique_results or result.get('score', 0) > unique_results[key].get('score', 0):
                unique_results[key] = result
        return list(unique_results.values())

    def _get_all_entities(self) -> List[str]:
        """Extract all entity names from the database"""
        try:
            all_names = set()
            offset = None
            
            while True:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=1000,
                        offset=offset,
                        with_payload=True
                    )
                except Exception as e:
                    logger.error(f"Error scrolling collection: {str(e)}")
                    break
                
                points, next_offset = scroll_result
                if not points:
                    break
                
                for point in points:
                    source_name = point.payload.get("source_name", "")
                    if not source_name:
                        source = point.payload.get("source", "")
                        source_name = Path(source).stem if source else ""
                    
                    if source_name:
                        all_names.add(source_name)
                
                offset = next_offset
                if not offset:
                    break
            
            return list(all_names)
            
        except Exception as e:
            logger.error(f"Failed to get entity names: {str(e)}")
            return []

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if len(token) > 2]

    def _rebuild_bm25_index(self):
        """Rebuild BM25 index"""
        try:
            if self.corpus_tokens:
                self.bm25_index = BM25Okapi(self.corpus_tokens)
                logger.info(f"✅ Rebuilt BM25 index with {len(self.corpus_tokens)} documents")
        except Exception as e:
            logger.error(f"BM25 index rebuild failed: {str(e)}")

    def _save_all_data(self):
        """Save all persistent data"""
        try:
            # Save BM25 index
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump({
                    'bm25_index': self.bm25_index,
                    'corpus_tokens': self.corpus_tokens
                }, f)
            
            # Save documents
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save processed files
            with open(self.processed_files_path, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
                
        except Exception as e:
            logger.error(f"Data saving failed: {str(e)}")

    def _get_file_hash(self, file_path: str) -> str:
        """Get file hash for tracking"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"File hash calculation failed: {str(e)}")
            return str(uuid.uuid4())

    # ====================================
    # COMPREHENSIVE SEARCH
    # ====================================

    def _comprehensive_search(self, query: str, search_term: str = None) -> List[Dict[str, Any]]:
        """Enhanced comprehensive search system with complete chunk retrieval"""
        try:
            if not search_term:
                search_term = self._extract_search_term(query)
            
            all_chunks = []
            all_results = []
            processed_ids = set()
            
            # First pass: Collect all chunks
            offset = None
            while True:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=10000000,
                        offset=offset,
                        with_payload=True
                    )
                except Exception as e:
                    logger.error(f"Error scrolling collection: {str(e)}")
                    break
                
                points, next_offset = scroll_result
                if not points:
                    break
                
                for point in points:
                    chunk = {
                        "id": point.id,
                        "text": point.payload.get("text", point.payload.get("content", "")),
                        "content": point.payload.get("text", point.payload.get("content", "")),
                        "source": point.payload.get("source", ""),
                        "source_name": point.payload.get("source_name", ""),
                        "page": point.payload.get("page", 1),
                        "chunk_index": point.payload.get("chunk_index", 0)
                    }
                    all_chunks.append(chunk)
                
                offset = next_offset
                if not offset:
                    break
            
            # Sort chunks by source and chunk index
            all_chunks.sort(key=lambda x: (x["source_name"], x["page"], x["chunk_index"]))
            
            # Second pass: Find matching chunks
            for chunk in all_chunks:
                if chunk["id"] in processed_ids:
                    continue
                
                text = chunk["text"]
                source_name = chunk["source_name"]
                
                if self._validate_presence(text, source_name, search_term):
                    confidence = self._calculate_confidence(text, search_term, source_name)
                    if confidence >= 0.5:
                        # Create result with complete chunk data
                        result = {
                            **chunk,  # Original complete chunk data
                            "method": "comprehensive_unified",
                            "match_confidence": confidence,
                            "score": confidence,
                            "matched_term": search_term,
                            "collection_name": self.collection_name,
                            "full_context": {
                                "source_name": source_name,
                                "page": chunk["page"],
                                "chunk_index": chunk["chunk_index"]
                            }
                        }
                        
                        all_results.append(result)
                        processed_ids.add(chunk["id"])
            
            # Sort by confidence
            all_results.sort(key=lambda x: x.get("match_confidence", 0), reverse=True)
            print("Comprehensive search: ",all_results)
            return all_results
            
        except Exception as e:
            logger.error(f"Comprehensive search failed: {str(e)}")
            return []

    def hybrid_retrieval(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Enhanced hybrid retrieval combining all search methods"""
        try:
            # Get vector search results with increased limit
            vector_results = self._vector_search(query)
            
            # Get BM25 results with increased limit
            bm25_results = self._bm25_search(query)
            
            # Combine and deduplicate while preserving chunks
            combined_results = {}
            
            # Add vector results
            for result in vector_results:
                # Use more specific key including chunk index
                key = f"{result.get('source_name', '')}_{result.get('page', 1)}_{result.get('chunk_index', 0)}"
                result['vector_score'] = result.get('score', 0)
                combined_results[key] = result
            
            # Add BM25 results
            for result in bm25_results:
                key = f"{result.get('source_name', '')}_{result.get('page', 1)}_{result.get('chunk_index', 0)}"
                if key in combined_results:
                    # Combine scores if entry exists
                    combined_results[key]['bm25_score'] = result.get('score', 0)
                    combined_results[key]['hybrid_score'] = (
                        combined_results[key].get('vector_score', 0) * 0.6 +
                        result.get('score', 0) * 0.4
                    )
                    combined_results[key]['method'] = 'hybrid'
                else:
                    # Add new entry with BM25 score
                    result['bm25_score'] = result.get('score', 0)
                    result['hybrid_score'] = result.get('score', 0) * 0.4
                    result['vector_score'] = 0.0
                    combined_results[key] = result
            
            # Convert to list and sort by hybrid score
            results = list(combined_results.values())
            
            # Enhanced sorting that considers both hybrid score and source relevance
            def sort_key(x):
                hybrid_score = x.get('hybrid_score', x.get('score', 0))
                source_name = x.get('source_name', '').lower()
                # Boost score if source name contains query terms
                query_terms = query.lower().split()
                source_relevance = sum(term in source_name for term in query_terms) * 0.1
                return hybrid_score + source_relevance
            
            results.sort(key=sort_key, reverse=True)
            # Return top_k results, but ensure we get all chunks from relevant documents
            top_sources = {r.get('source_name') for r in results[:top_k]}
            final_results = [
                r for r in results 
                if r.get('source_name') in top_sources
            ]
        
            return final_results
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            return []
        # Return top_k results, but ensure we get all chunks from relevant documents

    
    # ====================================
    # ENHANCED FALLBACK CLASSIFICATION
    # ====================================

    def _fallback_classification(self, query: str, previous_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced fallback classification method with dynamic case detection"""
        query_lower = query.lower()
        
        # Dynamic case explanation detection
        if any(word in query_lower for word in ["explain", "about", "case", "regarding"]):
            # Try to extract any case name dynamically
            case_name, original_term = self._extract_case_name_enhanced(query)
            if case_name:
                return {
                    "type": "case_explanation",
                    "intent": "information",
                    "entities": {"case": case_name},
                    "original_entities": {"case": original_term},
                    "requires_aggregation": False,
                    "expected_response_type": "text",
                    "keywords": self._get_case_keywords(case_name),
                    "confidence": 0.8,
                    "original_query": query,
                    "is_followup": previous_context is not None and self._is_followup_question(query),
                    "case_specific": True
                }
        
        if any(word in query_lower for word in ["how many", "count", "number of"]):
            intent = "count"
            requires_aggregation = True
            response_type = "number"
        elif any(word in query_lower for word in ["list", "show all", "find all"]):
            intent = "list"
            requires_aggregation = False
            response_type = "list"
        else:
            intent = "search"
            requires_aggregation = False
            response_type = "text"
        
        entities = {}
        if "judge" in query_lower:
            judge_match = re.search(r'judge\s+([A-Za-z\s]+)', query, re.IGNORECASE)
            if judge_match:
                entities["judge"] = judge_match.group(1).strip()
        
        # Check if this is a follow-up question
        is_followup = previous_context is not None and self._is_followup_question(query)
        
        return {
            "type": "judge_comprehensive" if entities.get("judge") else "general",
            "intent": intent,
            "entities": entities,
            "original_entities": entities,
            "requires_aggregation": requires_aggregation,
            "expected_response_type": response_type,
            "keywords": query_lower.split(),
            "confidence": 0.6,
            "original_query": query,
            "is_followup": is_followup,
            "case_specific": False
        }

    def _is_followup_question(self, query: str) -> bool:
        """Detect if this is a follow-up question"""
        followup_indicators = [
            "this case", "that case", "the case", "this matter", "the outcome", "the result",
            "what happened", "what was decided", "the decision", "the judgment", "the ruling",
            "it", "its", "their", "them", "he", "she", "they",
            "outcome of this", "result of this", "decision in this", "what was the",
            "main cause of this", "reason for this", "why did this", "how did this",
            "what is the outcome", "what was the outcome", "what is the main cause",
            "what was the main cause", "why", "how", "when", "where"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in followup_indicators)

    # ====================================
    # ANSWER GENERATION METHODS
    # ====================================

    def generate_answer(self, query: str, results: List[Dict[str, Any]], classification: Dict[str, Any] = None, previous_context: Dict[str, Any] = None) -> str:
        """Enhanced answer generation system with dynamic handling"""
        try:
            if not results:
                if classification and classification.get("is_followup") and previous_context:
                    entity_name = (previous_context.get("entities", {}).get("case") or 
                                previous_context.get("entities", {}).get("judge") or 
                                previous_context.get("entities", {}).get("entity", "the previous case"))
                    return f"I couldn't find additional information about {entity_name} in the collection '{self.collection_name}' to answer your follow-up question: '{query}'"
                return f"No relevant information found for your query: '{query}' in collection '{self.collection_name}'"
            
            # Default classification if not provided
            if not classification:
                classification = {"type": "general", "intent": "search", "entities": {}}
            
            query_type = classification.get("type", "general")
            intent = classification.get("intent", "search")
            entities = classification.get("entities", {})
            requires_aggregation = classification.get("requires_aggregation", False)
            is_followup = classification.get("is_followup", False)
            
            # Enhanced validation for results
            validated_results = self._validate_results(results, entities, intent, query)
            
            if not validated_results:
                entity_name = entities.get("case") or entities.get("judge") or entities.get("entity", "")
                if entity_name:
                    return f"No documents actually contain '{entity_name}' after validation in collection '{self.collection_name}'."
                else:
                    return f"No validated results found for your query in collection '{self.collection_name}'."
            
            # Sort results by relevance
            sorted_results = sorted(validated_results,
                                key=lambda x: x.get("match_confidence",
                                                    x.get("relevance_score",
                                                        x.get("hybrid_score",
                                                            x.get("score", 0)))),
                                reverse=True)
            
            # Generate appropriate answer based on intent and type
            if query_type == "case_explanation":
                return self._generate_case_explanation_answer(query, sorted_results, classification)
            elif intent == "count" and requires_aggregation:
                return self._generate_count_answer(query, sorted_results, classification)
            elif intent == "list":
                return self._generate_list_answer(query, sorted_results, classification)
            elif intent in ["outcome", "cause"] or is_followup:
                return self._generate_followup_answer(query, sorted_results, classification, previous_context)
            else:
                return self._generate_detailed_answer(query, sorted_results, classification)
                
        except Exception as e:
            logger.error(f"Answer generation failed for collection {self.collection_name}: {str(e)}")
            return f"Error generating answer for collection {self.collection_name}: {str(e)}"

    def _validate_results(self, results: List[Dict[str, Any]], entities: Dict[str, Any], intent: str, query: str = "") -> List[Dict[str, Any]]:
        """Enhanced results validation with query-specific logic"""
        validated_results = []
        entity_name = entities.get("case") or entities.get("judge") or entities.get("entity", "")
        
        # For case queries, be more permissive with any matching documents
        if any(word in query.lower() for word in ["explain", "about", "case", "regarding"]):
            # Extract case keywords dynamically
            case_name, _ = self._extract_case_name_enhanced(query)
            if case_name:
                case_keywords = self._get_case_keywords(case_name)
                for result in results:
                    text = result.get("text", result.get("content", "")).lower()
                    source_name = result.get("source_name", "").lower()
                    
                    # Check if any case keywords match
                    for keyword in case_keywords:
                        if keyword.lower() in text or keyword.lower() in source_name:
                            validated_results.append(result)
                            break
                return validated_results
        
        # General validation for other queries
        if not entity_name or intent in ["search", "information", "explain"]:
            validated_results = results
        else:
            # Only validate for specific entity searches
            for result in results:
                text = result.get("text", result.get("content", ""))
                source_name = result.get("source_name", "")
                if not source_name:
                    source = result.get("source", "")
                    source_name = Path(source).stem if source else ""
                
                if entity_name and self._validate_presence(text, source_name, entity_name):
                    validated_results.append(result)
                elif not entity_name:
                    validated_results.append(result)
        
        return validated_results

    def _generate_case_explanation_answer(self, query: str, results: List[Dict[str, Any]], classification: Dict[str, Any]) -> str:
        """Generate comprehensive case explanation answers"""
        try:
            entities = classification.get("entities", {})
            case_name = entities.get("case", "")
            
            if not results:
                return f"No information found about the {case_name} case in collection '{self.collection_name}'"
            
            # Focus on the most relevant results
            max_results = 8
            max_text_length = 12000000000
            total_context_length = 1000000
            
            context_parts = []
            sources = set()
            
            for result in results[:max_results]:
                source_name = result.get("source_name", "Unknown")
                if not source_name:
                    source = result.get("source", "")
                    source_name = Path(source).stem if source else "Unknown"
                
                page = result.get("page", 1)
                text = result.get("text", result.get("content", ""))
                
                if len(text) > max_text_length:
                    truncated = text[:max_text_length]
                    last_period = truncated.rfind('.')
                    if last_period > max_text_length * 0.7:
                        text = truncated[:last_period + 1]
                    else:
                        text = truncated + "..."
                
                context_parts.append(f"**Source: {source_name} (Page {page})**\n{text}\n")
                sources.add(source_name)
            
            context = "\n---\n\n".join(context_parts)
            
            # Enhanced system prompt for case explanations
            system_prompt = f"""You are a legal research assistant specializing in case analysis and explanation.

Your task is to provide a comprehensive explanation of the {case_name} case based on the legal documents provided.

Guidelines:
1. Provide a clear, structured explanation of the case
2. Include key facts, legal issues, parties involved, and outcomes if available
3. Mention judges, advocates, and other important legal entities
4. Reference specific documents and page numbers
5. Be comprehensive but organized and readable
6. If specific details are not available, clearly state this"""
            
            user_prompt = f"""Based on the legal documents provided, please provide a comprehensive explanation of the {case_name} case.

Include:
- Background and key facts
- Parties involved (plaintiffs, defendants, advocates, judges)
- Legal issues and arguments
- Court decisions and outcomes (if available)
- Any other relevant details

Legal Documents from Collection '{self.collection_name}':

{context[:total_context_length]}

Please provide a well-structured, comprehensive explanation of this case based on the available documents.

Answer:"""
            
            try:
                response = self.groq_client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                answer = response.choices[0].message.content
                
                # Add source references
                if sources:
                    source_list = "\n".join(f"• **{source}**" for source in sorted(sources))
                    answer += f"\n\n**📚 Sources Referenced:**\n{source_list}"
                
                return answer
                
            except Exception as e:
                logger.error(f"LLM call failed for case explanation: {str(e)}")
                # Fallback with actual content
                fallback_answer = f"Based on the documents about the {case_name} case, here's what I found:\n\n"
                
                for i, part in enumerate(context_parts[:4], 1):
                    fallback_answer += f"**Source {i}:**\n{part}\n"
                
                return fallback_answer
            
        except Exception as e:
            logger.error(f"Case explanation generation failed: {str(e)}")
            return f"Error generating explanation for the {case_name} case."

    def _generate_count_answer(self, query: str, results: List[Dict[str, Any]], classification: Dict[str, Any]) -> str:
        """Generate count answers"""
        try:
            entities = classification.get("entities", {})
            entity_name = entities.get("case") or entities.get("judge") or entities.get("entity", "")
            
            # Group results by document
            sources_by_name = defaultdict(list)
            for result in results:
                source_name = result.get("source_name", "Unknown")
                sources_by_name[source_name].append(result)
            
            total_documents = len(sources_by_name)
            total_mentions = len(results)
            
            answer = f"**📊 SEARCH RESULTS COUNT:**\n\n"
            
            if entity_name:
                answer += f"**Entity Searched:** {entity_name}\n\n"
            
            answer += f"**Total Documents:** {total_documents}\n"
            answer += f"**Total Mentions:** {total_mentions}\n\n"
            
            if total_documents > 0:
                answer += f"**📋 Document List:**\n"
                for i, source_name in enumerate(sorted(sources_by_name.keys()), 1):
                    chunks_count = len(sources_by_name[source_name])
                    answer += f"{i}. {source_name} ({chunks_count} sections)\n"
            
            return answer
            
        except Exception as e:
            logger.error(f"Count answer generation failed: {str(e)}")
            return f"Error generating count answer: {str(e)}"

    def _generate_list_answer(self, query: str, results: List[Dict[str, Any]], classification: Dict[str, Any]) -> str:
        """Generate list answers"""
        try:
            entities = classification.get("entities", {})
            entity_name = entities.get("case") or entities.get("judge") or entities.get("entity", "")
            
            # Group by source
            sources_by_name = defaultdict(list)
            for result in results:
                source_name = result.get("source_name", "Unknown")
                sources_by_name[source_name].append(result)
            
            answer = f"**📋 SEARCH RESULTS LIST:**\n\n"
            
            if entity_name:
                answer += f"**Searched For:** {entity_name}\n\n"
            
            answer += f"**Found in {len(sources_by_name)} documents:**\n"
            for i, source_name in enumerate(sorted(sources_by_name.keys()), 1):
                answer += f"{i}. {source_name}\n"
            
            return answer
            
        except Exception as e:
            logger.error(f"List answer generation failed: {str(e)}")
            return f"Error generating list answer: {str(e)}"

    def _generate_followup_answer(self, query: str, results: List[Dict[str, Any]], 
                                 classification: Dict[str, Any], previous_context: Dict[str, Any] = None) -> str:
        """Generate answers for follow-up questions about outcomes and causes"""
        try:
            if not results:
                return f"No additional information found to answer your follow-up question in collection '{self.collection_name}'"
            
            entities = classification.get("entities", {})
            intent = classification.get("intent", "search")
            
            # Determine the main entity being discussed
            entity_name = entities.get("case") or entities.get("judge") or entities.get("entity", "")
            if previous_context and not entity_name:
                prev_entities = previous_context.get("entities", {})
                entity_name = prev_entities.get("case") or prev_entities.get("judge") or prev_entities.get("entity", "")
            
            # Focus on the most relevant results
            max_results = 10
            max_text_length = 1000
            total_context_length = 8000
            
            context_parts = []
            sources = set()
            
            # Prioritize results that contain outcome/cause keywords
            outcome_keywords = ["outcome", "decision", "judgment", "ruling", "held", "decided", "concluded", 
                            "dismissed", "allowed", "granted", "rejected", "cause", "reason", "because", 
                            "due to", "arising from", "resulted from", "led to"]
            
            prioritized_results = []
            regular_results = []
            
            for result in results[:max_results * 2]:
                text = result.get("text", result.get("content", "")).lower()
                
                if any(keyword in text for keyword in outcome_keywords):
                    prioritized_results.append(result)
                else:
                    regular_results.append(result)
            
            # Use prioritized results first, then regular results
            combined_results = prioritized_results + regular_results
            
            for result in combined_results[:max_results]:
                source_name = result.get("source_name", "Unknown")
                if not source_name:
                    source = result.get("source", "")
                    source_name = Path(source).stem if source else "Unknown"
                
                page = result.get("page", 1)
                text = result.get("text", result.get("content", ""))
                
                if len(text) > max_text_length:
                    truncated = text[:max_text_length]
                    last_period = truncated.rfind('.')
                    if last_period > max_text_length * 0.7:
                        text = truncated[:last_period + 1]
                    else:
                        text = truncated + "..."
                
                context_parts.append(f"**Source: {source_name} (Page {page})**\n{text}\n")
                sources.add(source_name)
            
            context = "\n---\n\n".join(context_parts)
            
            # Better system prompt for follow-up questions
            system_prompt = f"""You are a legal research assistant specializing in case analysis. You are answering a follow-up question about a specific case.
            
The user previously asked about "{entity_name}" and now wants to know about the outcome or cause of this case.

Guidelines:
1. Focus SPECIFICALLY on answering the follow-up question
2. Look for information about case outcomes, decisions, rulings, judgments
3. Look for information about causes, reasons, origins of the dispute
4. Be direct and specific in your answer
5. Reference specific documents and page numbers
6. If the outcome/cause is not clear from the documents, state this clearly"""
            
            # Specific user prompt for follow-up questions
            if intent == "outcome" or "outcome" in query.lower():
                specific_instruction = """Focus on:
- Final court decision or judgment
- Whether the case was dismissed, allowed, granted, or rejected
- Any relief granted or denied
- Final orders or rulings
- Settlement details if any"""
            elif intent == "cause" or any(word in query.lower() for word in ["cause", "reason", "why"]):
                specific_instruction = """Focus on:
- Original dispute or legal issue
- Reasons for the legal action
- What led to the case being filed
- Background circumstances
- Root cause of the conflict"""
            else:
                specific_instruction = """Focus on both the cause and outcome of the case."""
            
            user_prompt = f"""Based on the legal documents provided, please answer this follow-up question about {entity_name}: "{query}"

{specific_instruction}

Context Documents from Collection '{self.collection_name}':

{context[:total_context_length]}

Please provide a direct, focused answer that specifically addresses the question asked. Reference the source documents and page numbers.

Answer:"""
            
            try:
                response = self.groq_client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                answer = response.choices[0].message.content
                
                # Add source references
                if sources:
                    source_list = "\n".join(f"• **{source}**" for source in sorted(sources))
                    answer += f"\n\n**📚 Sources Referenced:**\n{source_list}"
                
                return answer
                
            except Exception as e:
                logger.error(f"LLM call failed for follow-up answer: {str(e)}")
                # Fallback with actual content
                fallback_answer = f"Based on the documents about {entity_name}, here's what I found:\n\n"
                
                for i, part in enumerate(context_parts[:3], 1):
                    fallback_answer += f"**Source {i}:**\n{part}\n"
                
                return fallback_answer
            
        except Exception as e:
            logger.error(f"Follow-up answer generation failed: {str(e)}")
            return f"Error generating follow-up answer about {entity_name}."

    def _generate_detailed_answer(self, query: str, results: List[Dict[str, Any]], classification: Dict[str, Any]) -> str:
        """Generate detailed answers without truncating chunks"""
        try:
            if not results:
                return f"No relevant information found for your query"
            
            # Get context from results
            max_results = 10000000
            context_parts = []
            sources = set()
            
            for result in results[:max_results]:
                source_name = result.get("source_name", "Unknown")
                page = result.get("page", 1)
                text = result.get("text", result.get("content", ""))
                
                # Remove truncation - use full chunk
                context_parts.append(f"**Source: {source_name} (Page {page})**\n{text}\n")
                sources.add(source_name)
            
            context = "\n---\n\n".join(context_parts)
            
            # Enhanced system prompt for universal entities
            system_prompt = f"""You are a highly knowledgeable legal research assistant specializing in comprehensive legal entity analysis.

    Your task is to provide detailed, accurate answers about ANY legal entity mentioned in the query, including:
    - Judges, Justices, and Judicial Officers
    - Advocates, Lawyers, and Legal Counsel
    - Plaintiffs, Petitioners, and Appellants
    - Defendants, Respondents, and Appellees
    - Courts, Tribunals, and Legal Authorities
    - Case Numbers, Citations, and Legal References
    - Government Officials and Legal Officers
    - Any other legal entities or persons
    - If the question is regarding justice, judge or coram of a particular case, then understand it all means the same, in the case it might be CORAM: 
    HON'BLE MR. JUSTICE AMIT BANSAL,here the judge is amit bansal

    Provide comprehensive, well-structured answers based on the legal documents provided."""

            user_prompt = f"""Based on the provided legal documents, please answer this comprehensive query: "{query}"

    Legal Documents Context:
    {context}

    Provide a detailed, accurate response that addresses all aspects of the query."""

            try:
                response = self.groq_client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                answer = response.choices[0].message.content
                
                # Add source references
                if sources:
                    source_list = "\n".join(f"• **{source}**" for source in sorted(sources))
                    answer += f"\n\n**📚 Sources Referenced:**\n{source_list}"
                
                return answer
                
            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}")
                # Fallback with actual content (no truncation)
                fallback_answer = f"Based on the documents, I found relevant information from {len(sources)} sources:\n\n"
                for i, part in enumerate(context_parts[:3], 1):
                    fallback_answer += f"**Source {i}:**\n{part}\n"
                return fallback_answer
                
        except Exception as e:
            logger.error(f"Detailed answer generation failed: {str(e)}")
            return f"Error generating detailed response: {str(e)}"

    def _format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information"""
        sources_info = defaultdict(lambda: {"pages": set(), "chunks": 0, "max_score": 0})
        
        for result in results:
            source_name = result.get("source_name", "Unknown")
            page = result.get("page", 1)
            score = result.get("match_confidence", result.get("score", 0))
            
            sources_info[source_name]["pages"].add(page)
            sources_info[source_name]["chunks"] += 1
            sources_info[source_name]["max_score"] = max(sources_info[source_name]["max_score"], score)
        
        return [
            {
                "document": source_name,
                "pages": sorted(list(info["pages"])),
                "chunks": info["chunks"],
                "max_score": info["max_score"],
                "collection": self.collection_name
            }
            for source_name, info in sources_info.items()
        ]

    # ====================================
    # MAIN QUERY INTERFACE
    # ====================================

    def query(self, query: str, debug: bool = False, previous_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main query processing interface - Dynamic and Universal"""
        try:
            self.debug_mode = debug
            logger.info(f"Processing query in collection {self.collection_name}: {query}")
            
            # Enhanced query classification
            classification = self._setup_ai_agent(query)
            print("classification:", classification)
            query_type = classification["type"]
            intent = classification["intent"]
            entities = classification["entities"]
            is_case_specific = classification.get("case_specific", False)
            print("Entities extracted:", entities)
            
            # Initialize results as empty list to avoid reference errors
            results = []
            
            # Handle case-specific queries with high precision
            if is_case_specific and entities.get("case"):
                case_name = entities["case"]
                print("Case:", case_name)
                if query_type == "case_explanation":
                    # Use case-specific search for case explanations
                    results = self._case_specific_search(query, case_name)
                    method_used = "case_specific_explanation"
                    
                elif query_type == "entity_in_case":
                    print("Inside query method - entity_in_case")
                    # Search for specific entity within the case
                    results = self._case_specific_search(query, case_name)
                    print("-------------------------------------------")
                    print("Results from entity_in_case:", len(results))  
                    print("-------------------------------------------")
                    method_used = "entity_in_case_search"
                    
                else:
                    # Fallback to comprehensive search but filter by case
                    comprehensive_results = self._comprehensive_search(query, case_name)
                    results = self._filter_documents_by_case(comprehensive_results, case_name, strict_mode=True)
                    method_used = "case_filtered_comprehensive"
            
            elif query_type == "list_search":
                # Handle list queries - FIX: Handle all entity types, not just "entity"
                print("Inside query method - list_search")
                
                # Extract the entity value regardless of its type (judge, advocate, entity, etc.)
                entity_value = None
                entity_type_name = "entity"
                
                # Check judge first as it's most common for list queries
                if entities.get("judge"):
                    entity_value = entities["judge"]
                    entity_type_name = "judge"
                # Then check for other entity types
                elif entities.get("entity"):
                    entity_value = entities["entity"]
                    entity_type_name = "entity"
                elif entities.get("advocate"):
                    entity_value = entities["advocate"]
                    entity_type_name = "advocate"
                elif entities.get("court"):
                    entity_value = entities["court"]
                    entity_type_name = "court"
                
                if entity_value:
                    print(f"Searching for {entity_type_name}: {entity_value}")
                    # Search for entity mentions
                    results = self._comprehensive_search(query, entity_value)
                    
                    # Get unique sources while preserving result structure
                    unique_sources = {result.get('source_name', 'Unknown') for result in results}
                    
                    # Create new results list with one entry per unique source
                    unique_results = []
                    seen_sources = set()
                    
                    for result in results:
                        source_name = result.get('source_name', 'Unknown')
                        if source_name not in seen_sources:
                            unique_results.append(result)
                            seen_sources.add(source_name)
        
                    results = unique_results
                    print(f"Unique sources found: {len(unique_results)}")
                    print("Results from entity_list_search:", len(results))
                    method_used = "entity_list_search"
                else:
                    # No entity found for list search
                    results = self.hybrid_retrieval(query, top_k=30)
                    method_used = "list_search_fallback"
            
            elif query_type == "judge_comprehensive":
                judge_name = entities.get("judge", "")
                if not judge_name:
                    judge_name = self._extract_search_term(query)
                
                if judge_name:
                    results = self._comprehensive_search(query, judge_name)
                    method_used = "enhanced_comprehensive"
                else:
                    results = []
                    method_used = "no_valid_judge_extracted"
            else:
                results = self.hybrid_retrieval(query, top_k=30)
                method_used = "hybrid_enhanced"
            
            # Generate answer with dynamic context
            answer = self.generate_answer(query, results, classification, previous_context)
            
            # Build response
            response = {
                "query": query,
                "answer": answer,
                "method": method_used,
                "query_type": query_type,
                "intent": intent,
                "entities": entities,
                "original_entities": classification.get("original_entities", {}),
                "requires_aggregation": classification.get("requires_aggregation", False),
                "expected_response_type": classification.get("expected_response_type", "text"),
                "results_count": len(results),
                "sources": self._format_sources(results),
                "classification": classification,
                "ai_agent_used": self.agent is not None,
                "enhanced_search_applied": True,
                "collection_name": self.collection_name,
                "search_precision": "case_specific" if is_case_specific else "enhanced",
                "case_focused": is_case_specific
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "method": "error",
                "query_type": "error",
                "intent": "error",
                "entities": {},
                "results_count": 0,
                "sources": [],
                "collection_name": self.collection_name
            }

    # ====================================
    # DOCUMENT PROCESSING METHODS
    # ====================================

    def process_and_index_document(self, pdf_path: str, force_reindex: bool = False) -> Dict[str, Any]:
        """Process and index a single document"""
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                return {"success": False, "error": "File not found"}
            
            # Check if already processed
            file_hash = self._get_file_hash(str(pdf_path))
            if not force_reindex and file_hash in self.processed_files:
                return {
                    "success": True,
                    "message": "Document already processed",
                    "chunks": self.processed_files[file_hash].get("chunks", 0)
                }
            
            # Extract text
            documents = self.extract_text_multiple_methods(str(pdf_path))
            if not documents:
                return {"success": False, "error": "No text extracted"}
            
            # Create chunks
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_documents([doc])
                chunks.extend(doc_chunks)
            
            valid_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > self.min_content_threshold]
            if not valid_chunks:
                return {"success": False, "error": "No valid chunks created"}
            
            # Create embeddings and store
            chunk_count = 0
            for i, chunk in enumerate(valid_chunks):
                try:
                    if self.device == 'mps' and hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    
                    embedding = self.embedding_model.embed_query(chunk.page_content)
                    
                    point_id = str(uuid.uuid4())
                    metadata = {
                        "text": chunk.page_content,
                        "content": chunk.page_content,
                        "source": str(pdf_path),
                        "source_name": pdf_path.stem,
                        "page": chunk.metadata.get("page", 1),
                        "chunk_index": i,
                        "file_hash": file_hash,
                        "processed_at": datetime.now().isoformat(),
                        "collection": self.collection_name
                    }
                    
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=metadata
                    )
                    
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=[point]
                    )
                    
                    chunk_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing chunk {i}: {str(e)}")
                    continue
            
            # Update BM25 index
            self.documents.extend(valid_chunks)
            new_tokens = [self._tokenize_text(chunk.page_content) for chunk in valid_chunks]
            self.corpus_tokens.extend(new_tokens)
            self._rebuild_bm25_index()
            
            # Mark as processed
            self.processed_files[file_hash] = {
                "file_path": str(pdf_path),
                "processed_at": datetime.now().isoformat(),
                "chunks": chunk_count,
                "collection": self.collection_name
            }
            
            self._save_all_data()
            
            return {
                "success": True,
                "message": f"Processed {chunk_count} chunks from {pdf_path.name}",
                "chunks": chunk_count,
                "file_hash": file_hash,
                "collection": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def extract_text_multiple_methods(self, pdf_path: str) -> List[Document]:
        """Extract text using multiple fallback methods"""
        documents = []
        
        # Method 1: PyPDFLoader (Primary)
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            if docs:
                documents.extend(docs)
                logger.info(f"✅ PyPDFLoader extracted {len(docs)} pages")
                return documents
        except Exception as e:
            logger.warning(f"PyPDFLoader failed: {str(e)}")
        
        # Method 2: PyMuPDF (Fallback 1)
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num + 1}
                    ))
            doc.close()
            if documents:
                logger.info(f"✅ PyMuPDF extracted {len(documents)} pages")
                return documents
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {str(e)}")
        
        # Method 3: pdfplumber (Fallback 2)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": pdf_path, "page": page_num + 1}
                        ))
            if documents:
                logger.info(f"✅ pdfplumber extracted {len(documents)} pages")
                return documents
        except Exception as e:
            logger.warning(f"pdfplumber failed: {str(e)}")
        
        # Method 4: PyPDF2 (Fallback 3)
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": pdf_path, "page": page_num + 1}
                        ))
            if documents:
                logger.info(f"✅ PyPDF2 extracted {len(documents)} pages")
                return documents
        except Exception as e:
            logger.warning(f"PyPDF2 failed: {str(e)}")
        
        logger.error(f"❌ All PDF extraction methods failed for {pdf_path}")
        return documents

    def process_directory(self, directory_path: str, force_reindex: bool = False) -> Dict[str, Any]:
        """Process all PDFs in a directory"""
        try:
            directory_path = Path(directory_path)
            if not directory_path.exists():
                return {"success": False, "error": "Directory not found"}
            
            pdf_files = list(directory_path.glob("*.pdf"))
            if not pdf_files:
                return {"success": False, "error": "No PDF files found"}
            
            results = {
                "total_files": len(pdf_files),
                "processed": 0,
                "skipped": 0,
                "failed": 0,
                "details": []
            }
            
            for pdf_file in pdf_files:
                result = self.process_and_index_document(str(pdf_file), force_reindex)
                
                if result["success"]:
                    if "already processed" in result.get("message", ""):
                        results["skipped"] += 1
                    else:
                        results["processed"] += 1
                else:
                    results["failed"] += 1
                
                results["details"].append({
                    "file": pdf_file.name,
                    "result": result
                })
            
            results["success"] = True
            results["message"] = f"Processed {results['processed']} files, skipped {results['skipped']}, failed {results['failed']}"
            
            return results
            
        except Exception as e:
            logger.error(f"Directory processing failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "total_vectors": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "processed_files": len(self.processed_files),
                "bm25_documents": len(self.corpus_tokens),
                "langchain_documents": len(self.documents)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics - alias for get_collection_stats for backward compatibility"""
        return self.get_collection_stats()

    def delete_collection(self, confirm: bool = False) -> Dict[str, Any]:
        """Delete the entire collection"""
        if not confirm:
            return {
                "success": False,
                "error": "Please set confirm=True to delete the collection"
            }
        
        try:
            # Delete from Qdrant
            self.qdrant_client.delete_collection(self.collection_name)
            
            # Clear local data
            self.documents = []
            self.corpus_tokens = []
            self.bm25_index = None
            self.processed_files = {}
            
            # Remove data files
            for file_path in [self.bm25_index_path, self.documents_path, self.processed_files_path]:
                if file_path.exists():
                    file_path.unlink()
            
            logger.info(f"🗑️ Deleted collection: {self.collection_name}")
            
            return {
                "success": True,
                "message": f"Collection '{self.collection_name}' deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Collection deletion failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _find_best_name_matches_enhanced(self, entity_name: str) -> str:
        """Enhanced name matching with better validation"""
        try:
            all_entities = self._get_all_entities()
            matches = []
            
            # Stage 1: Exact matching
            exact_matches = [name for name in all_entities if name.lower() == entity_name.lower()]
            if exact_matches:
                matches.extend([(name, 1.0, "exact") for name in exact_matches])
            
            # Stage 2: Fuzzy matching
            fuzzy_matches = process.extract(entity_name, all_entities, limit=15, scorer=fuzz.token_sort_ratio)
            for name, score in fuzzy_matches:
                if score >= 70:
                    matches.append((name, score/100.0, "fuzzy"))
            
            # Stage 3: Partial matching
            partial_matches = process.extract(entity_name, all_entities, limit=12, scorer=fuzz.partial_ratio)
            for name, score in partial_matches:
                if score >= 80:
                    matches.append((name, score/100.0, "partial"))
            
            # Remove duplicates and sort
            unique_matches = {}
            for name, score, method in matches:
                if name not in unique_matches or score > unique_matches[name][0]:
                    unique_matches[name] = (score, method)
            
            sorted_matches = sorted(unique_matches.items(), key=lambda x: x[1][0], reverse=True)
            
            best_matches = [
                {
                    "name": name,
                    "score": score,
                    "method": method
                }
                for name, (score, method) in sorted_matches[:15]
            ]
            
            return json.dumps(best_matches)
            
        except Exception as e:
            logger.error(f"Enhanced name matching failed: {str(e)}")
            return json.dumps([])


if __name__ == "__main__":
    # Example usage
    rag = EnhancedLegalRAG(
        qdrant_url="http://localhost:6333",
        collection_name="legal_collection",
        data_dir="legal_data"
    )
    
    # Process documents
    # result = rag.process_directory("path/to/pdf/directory")
    # print(result)
    
    # Query the system - Now works with ANY legal documents
    # response = rag.query("explain Microsoft vs Apple case")
    # response = rag.query("who is the judge for XYZ Corp vs ABC Ltd")
    # response = rag.query("about Samsung vs Sony dispute")
    # print(response["answer"])