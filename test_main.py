import os
import logging
import tempfile
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
from io import BytesIO
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import your existing systems
from final_rag import EnhancedLegalRAG
from timeline import (
    TimelineExtractor,
    process_timeline,
    create_word_document,
    format_timeline_for_api,
    get_pdf_preview,
    validate_pdf_path,
    get_pdf_files_from_folder,
    get_file_info
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - Keep for RAG system only
RAG_DOCUMENTS_FOLDER = Path("/home/rudra-panda/Desktop/LegalRAG/testdoc")

# ====== EXISTING PYDANTIC MODELS (unchanged) ======

class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query")
    debug: bool = Field(False, description="Enable debug mode")

class SimpleQueryResponse(BaseModel):
    """Simplified response with only the answer"""
    answer: str
    context_used: Optional[bool] = None
    resolved_query: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    resolved_query: Optional[str] = None
    context_used: Optional[bool] = False
    answer: str
    method: str
    query_type: str
    intent: str
    entities: Dict[str, Any]
    original_entities: Optional[Dict[str, Any]] = None
    results_count: int
    sources: List[Dict[str, Any]]
    conversation_context: Optional[Dict[str, Any]] = None
    total_documents: Optional[int] = None
    total_mentions: Optional[int] = None
    confidence_metrics: Optional[Dict[str, float]] = None
    maximum_recall_applied: Optional[bool] = None
    ai_agent_used: Optional[bool] = None

class DocumentProcessRequest(BaseModel):
    force_reindex: bool = Field(False, description="Force reindexing of documents")

class DocumentProcessResponse(BaseModel):
    success: bool
    message: str
    chunks: Optional[int] = None
    error: Optional[str] = None

class TimelineResponse(BaseModel):
    success: bool
    timeline_data: Optional[List[Dict[str, Any]]] = None
    total_events: Optional[int] = None
    document_name: Optional[str] = None
    word_document_path: Optional[str] = None
    error: Optional[str] = None

class SystemStatsResponse(BaseModel):
    vector_db: Dict[str, Any]
    bm25_index: Dict[str, Any]
    system: Dict[str, Any]
    validation_thresholds: Dict[str, Any]
    supported_query_types: List[str]
    supported_intents: List[str]
    conversation_features: Optional[Dict[str, Any]] = None

class FolderContentsResponse(BaseModel):
    success: bool
    files: List[Dict[str, Any]]
    total_files: int
    folder_path: str
    error: Optional[str] = None

class PDFSelectionRequest(BaseModel):
    selected_files: List[str] = Field(..., description="List of selected PDF file paths")

class FolderUploadResponse(BaseModel):
    success: bool
    uploaded_files: List[str]
    failed_files: List[str]
    total_files: int
    folder_session_id: str
    error: Optional[str] = None

class ConversationHistoryResponse(BaseModel):
    success: bool
    history: List[Dict[str, Any]]
    total_interactions: int
    active_context: bool
    context_entities: Dict[str, Any]
    error: Optional[str] = None

class ConversationContextResponse(BaseModel):
    success: bool
    message: str
    active_context: bool
    context_entities: Dict[str, Any]
    error: Optional[str] = None

# ====== NEW COLLECTION PYDANTIC MODELS ======

class CollectionUploadRequest(BaseModel):
    collection_name: str = Field(..., description="Name for the document collection")
    force_reindex: bool = Field(False, description="Force reindexing of documents")



class CollectionUploadResponse(BaseModel):
    success: bool
    collection_id: str
    collection_name: str
    uploaded_files: List[str]
    failed_files: List[str]
    total_files: int  # Make sure this field exists
    file_count: int  # Keep this for backward compatibility
    processed_chunks: int
    error: Optional[str] = None

class CollectionQueryRequest(BaseModel):
    query: str = Field(..., description="The search query")
    collection_id: str = Field(..., description="ID of the collection to query")
    debug: bool = Field(False, description="Enable debug mode")

class CollectionListResponse(BaseModel):
    success: bool
    collections: List[Dict[str, Any]]
    total_collections: int
    error: Optional[str] = None

class EnhancedIntegratedLegalSystem:
    """Enhanced system with separate collection management alongside existing RAG"""
    
    def __init__(self):
        # Existing system paths
        self.upload_folder = Path("uploads")
        self.output_folder = Path("outputs")
        self.temp_folder = Path("temp")
        self.rag_documents_folder = RAG_DOCUMENTS_FOLDER  # Keep original
        
        # NEW: Collection management paths
        self.collections_folder = Path("collections")
        self.collections_folder.mkdir(exist_ok=True)
        
        # Create necessary directories
        self.upload_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        self.temp_folder.mkdir(exist_ok=True)
        
        # Existing RAG system (unchanged)
        self.rag_system = None
        self.timeline_extractor = None
        self.session_folders = {}  # Track uploaded folder sessions
        
        # NEW: Collection management
        self.collection_systems = {}  # collection_id -> RAG system instance
        self.collection_metadata = {}  # collection_id -> metadata
        
        # Initialize systems
        self._initialize_systems()

    def _initialize_systems(self):
        """Initialize both existing and new systems"""
        try:
            # Original RAG system initialization (unchanged)
            logger.info("Initializing Enhanced Legal RAG system with conversation context...")
            self.rag_system = EnhancedLegalRAG()
            logger.info("RAG system with conversation context initialized successfully")
            
            logger.info("Initializing Timeline Extractor...")
            self.timeline_extractor = TimelineExtractor()
            logger.info("Timeline extractor initialized successfully")
            
            # Process original documents folder ONLY if it exists and has PDFs
            if self.rag_documents_folder.exists():
                pdf_files = list(self.rag_documents_folder.glob("**/*.pdf"))
                if pdf_files:
                    logger.info(f"Found {len(pdf_files)} PDF files in RAG documents folder. Processing for RAG...")
                    try:
                        self.process_directory_for_rag(str(self.rag_documents_folder), force_reindex=False)
                    except Exception as e:
                        logger.warning(f"Some PDF files in RAG folder failed to process: {e}")
            
            # Load existing collections if any
            self._load_existing_collections()
            
        except Exception as e:
            logger.error(f"Error initializing systems: {str(e)}")
            raise

    def _load_existing_collections(self):
        """Load any existing collection metadata"""
        try:
            collections_metadata_file = self.collections_folder / "collections_metadata.json"
            if collections_metadata_file.exists():
                import json
                with open(collections_metadata_file, 'r') as f:
                    self.collection_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.collection_metadata)} existing collections")
        except Exception as e:
            logger.warning(f"Could not load existing collections metadata: {str(e)}")

    def _save_collections_metadata(self):
        """Save collection metadata to disk"""
        try:
            collections_metadata_file = self.collections_folder / "collections_metadata.json"
            import json
            with open(collections_metadata_file, 'w') as f:
                json.dump(self.collection_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save collections metadata: {str(e)}")

    # ====== EXISTING METHODS (unchanged) ======
    
    # Update the get_pdf_file_path method around line 250

    def get_pdf_file_path(self, filename: str, session_id: str = None, collection_id: str = None) -> Optional[str]:
        """Get full path of PDF file from various sources including collections"""
        # NEW: Check collection folders first if collection_id provided
        if collection_id and collection_id in self.collection_metadata:
            collection_metadata = self.collection_metadata[collection_id]
            
            # Check session folder for this collection
            if "session_folder" in collection_metadata:
                session_folder = Path(collection_metadata["session_folder"])
                collection_pdf_path = session_folder / filename
                if collection_pdf_path.exists():
                    return str(collection_pdf_path)
            
            # Check collection documents folder
            if "documents_folder" in collection_metadata:
                docs_folder = Path(collection_metadata["documents_folder"])
                collection_pdf_path = docs_folder / filename
                if collection_pdf_path.exists():
                    return str(collection_pdf_path)
        
        # Check session folder if session_id provided
        if session_id and session_id in self.session_folders:
            session_folder = self.session_folders[session_id]
            session_path = Path(session_folder) / filename
            if session_path.exists():
                return str(session_path)
        
        # Check in uploads folder
        upload_path = self.upload_folder / filename
        if upload_path.exists():
            return str(upload_path)
        
        # Check in original RAG documents folder
        for pdf_file in self.rag_documents_folder.rglob("*.pdf"):
            if pdf_file.name == filename:
                return str(pdf_file)
        
        # NEW: Check all collection folders as last resort
        for coll_id, metadata in self.collection_metadata.items():
            if "session_folder" in metadata:
                session_folder = Path(metadata["session_folder"])
                pdf_path = session_folder / filename
                if pdf_path.exists():
                    return str(pdf_path)
        
        return None
    
    
    # Update the process_query method in test_main.py to ensure all required fields are present

    def process_query(self, query: str, debug: bool = False, simple_response: bool = False) -> Dict[str, Any]:
        """Process query using ORIGINAL RAG system with fixed field validation"""
        try:
            if not self.rag_system:
                raise HTTPException(status_code=500, detail="RAG system not initialized")
            
            # Get the raw response from the RAG system
            raw_response = self.rag_system.query(query, debug=debug)
            
            # Return simple response if requested
            if simple_response:
                return {
                    "answer": raw_response.get("answer", "No answer available."),
                    "context_used": raw_response.get("context_used", False),
                    "resolved_query": raw_response.get("resolved_query", query)
                }
            
            # Enhanced sources with full file paths
            enhanced_sources = []
            for source in raw_response.get('sources', []):
                document_name = source.get('document', '')
                full_path = self.get_pdf_file_path(document_name + '.pdf')
                enhanced_source = {
                    'document': document_name,
                    'pages': source.get('pages', []),
                    'chunks': source.get('chunks', 0),
                    'max_score': source.get('max_score', 0),
                    'file_path': full_path,
                    'downloadable': full_path is not None
                }
                enhanced_sources.append(enhanced_source)
            
            # FIXED: Ensure all required fields are present for QueryResponse model
            validated_response = {
                # Required fields that were missing
                "query": query,
                "answer": raw_response.get("answer", "No answer available"),
                "method": raw_response.get("method", "default_search"),
                "query_type": raw_response.get("query_type", "general"),
                "intent": raw_response.get("intent", "search"),
                "entities": raw_response.get("entities", {}),
                "results_count": raw_response.get("results_count", len(enhanced_sources)),
                "sources": enhanced_sources,
                
                # Optional fields
                "resolved_query": raw_response.get("resolved_query", query),
                "context_used": raw_response.get("context_used", False),
                "original_entities": raw_response.get("original_entities", {}),
                "total_documents": raw_response.get("total_documents", len(set(s.get("document", "") for s in enhanced_sources))),
                "total_mentions": raw_response.get("total_mentions", raw_response.get("results_count", 0)),
                "maximum_recall_applied": raw_response.get("maximum_recall_applied", False),
                "ai_agent_used": raw_response.get("ai_agent_used", False),
            }
            
            # Add conversation context information if available
            if raw_response.get('context_used', False):
                validated_response['conversation_context'] = raw_response.get('conversation_context', {})
            
            # Add confidence metrics if available
            if raw_response.get('results_count', 0) > 0:
                validated_response['confidence_metrics'] = {
                    'average_confidence': raw_response.get('average_confidence', 0),
                    'very_high_confidence_matches': raw_response.get('very_high_confidence_matches', 0),
                    'high_confidence_matches': raw_response.get('high_confidence_matches', 0),
                    'medium_confidence_matches': raw_response.get('medium_confidence_matches', 0),
                    'low_confidence_matches': raw_response.get('low_confidence_matches', 0),
                    'confidence_threshold': raw_response.get('confidence_threshold', 0.6)
                }
            
            return validated_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # Return a valid object even in case of errors
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "method": "error",
                "query_type": "error",
                "intent": "error",
                "entities": {},
                "results_count": 0,
                "sources": []
            }

    def get_conversation_history(self) -> Dict[str, Any]:
        """Get conversation history from RAG system (unchanged)"""
        try:
            if not self.rag_system or not hasattr(self.rag_system, 'conversation_context'):
                return {
                    'success': False,
                    'error': 'Conversation context not available',
                    'history': [],
                    'total_interactions': 0,
                    'active_context': False,
                    'context_entities': {}
                }
            
            history = self.rag_system.conversation_context.get_conversation_history()
            context_entities = self.rag_system.conversation_context.context_entities
            
            return {
                'success': True,
                'history': history,
                'total_interactions': len(history),
                'active_context': bool(context_entities),
                'context_entities': context_entities
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'history': [],
                'total_interactions': 0,
                'active_context': False,
                'context_entities': {}
            }

    def clear_conversation_context(self) -> Dict[str, Any]:
        """Clear conversation context (unchanged)"""
        try:
            if not self.rag_system or not hasattr(self.rag_system, 'conversation_context'):
                return {
                    'success': False,
                    'error': 'Conversation context not available',
                    'message': 'No conversation context to clear',
                    'active_context': False,
                    'context_entities': {}
                }
            
            self.rag_system.conversation_context.clear_context()
            return {
                'success': True,
                'message': 'Conversation context cleared successfully',
                'active_context': False,
                'context_entities': {}
            }
            
        except Exception as e:
            logger.error(f"Error clearing conversation context: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to clear conversation context',
                'active_context': False,
                'context_entities': {}
            }

    def process_document_for_rag(self, file_path: str, force_reindex: bool = False) -> Dict[str, Any]:
        """Process a document for RAG indexing (unchanged)"""
        try:
            if not self.rag_system:
                raise HTTPException(status_code=500, detail="RAG system not initialized")
            
            result = self.rag_system.process_and_index_document(file_path, force_reindex)
            return result
            
        except Exception as e:
            logger.error(f"Error processing document for RAG: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

    def process_directory_for_rag(self, directory_path: str, force_reindex: bool = False) -> Dict[str, Any]:
        """Process a directory of documents for RAG indexing (unchanged)"""
        try:
            if not self.rag_system:
                raise HTTPException(status_code=500, detail="RAG system not initialized")
            
            result = self.rag_system.process_directory(directory_path, force_reindex)
            return result
            
        except Exception as e:
            logger.error(f"Error processing directory for RAG: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Directory processing failed: {str(e)}")

    def upload_folder_files(self, files: List[UploadFile]) -> Dict[str, Any]:
        """Upload multiple files and create a session folder (unchanged)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"folder_session_{timestamp}"
            session_folder = self.upload_folder / session_id
            session_folder.mkdir(exist_ok=True)
            
            uploaded_files = []
            failed_files = []
            
            for file in files:
                try:
                    if not file.filename.lower().endswith('.pdf'):
                        failed_files.append(file.filename)
                        continue
                    
                    file_path = session_folder / file.filename
                    with open(file_path, "wb") as buffer:
                        content = file.file.read()
                        buffer.write(content)
                    
                    uploaded_files.append(file.filename)
                    
                except Exception as e:
                    logger.error(f"Error uploading file {file.filename}: {str(e)}")
                    failed_files.append(file.filename)
            
            # Store session folder path
            self.session_folders[session_id] = str(session_folder)
            
            return {
                'success': True,
                'uploaded_files': uploaded_files,
                'failed_files': failed_files,
                'total_files': len(uploaded_files),
                'folder_session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"Error uploading folder: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'uploaded_files': [],
                'failed_files': [],
                'total_files': 0,
                'folder_session_id': None
            }

    def get_folder_contents(self, session_id: str = None) -> Dict[str, Any]:
        """Get contents of uploaded folder session ONLY (unchanged)"""
        try:
            # ONLY work with uploaded sessions - no default folder
            if not session_id or session_id not in self.session_folders:
                return {
                    'success': False,
                    'error': 'No folder session provided. Please upload a folder first.',
                    'files': [],
                    'total_files': 0,
                    'folder_path': '',
                    'requires_upload': True
                }
            
            folder_path = Path(self.session_folders[session_id])
            folder_name = f"Uploaded Session: {session_id}"
            
            if not folder_path.exists():
                return {
                    'success': False,
                    'error': f'Session folder not found: {folder_path}',
                    'files': [],
                    'total_files': 0,
                    'folder_path': str(folder_path)
                }
            
            # Get PDF files from the uploaded session folder only
            pdf_files = list(folder_path.glob("*.pdf"))
            files_info = []
            
            for pdf_file in pdf_files:
                try:
                    file_info = get_file_info(str(pdf_file))
                    # Add relative path for frontend
                    relative_path = pdf_file.name  # Just the filename since it's flat
                    file_info['relative_path'] = relative_path
                    file_info['display_name'] = pdf_file.name
                    file_info['session_id'] = session_id
                    files_info.append(file_info)
                except Exception as e:
                    logger.warning(f"Error getting file info for {pdf_file}: {e}")
                    continue
            
            return {
                'success': True,
                'files': files_info,
                'total_files': len(files_info),
                'folder_path': str(folder_path),
                'folder_name': folder_name,
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"Error getting folder contents: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'files': [],
                'total_files': 0,
                'folder_path': '',
                'requires_upload': True
            }

    def extract_timeline_enhanced(self, file_path: str, document_name: str = None) -> Dict[str, Any]:
        """Extract timeline from a PDF document with enhanced page number tracking (unchanged)"""
        try:
            if not self.timeline_extractor:
                raise HTTPException(status_code=500, detail="Timeline extractor not initialized")
            
            # Validate PDF file
            if not validate_pdf_path(file_path):
                raise HTTPException(status_code=400, detail="Invalid PDF file")
            
            # Process the document
            if not document_name:
                document_name = Path(file_path).stem
            
            df, doc = process_timeline(file_path, document_name)
            
            if df is None or df.empty:
                return {
                    'success': False,
                    'error': 'No timeline events found in the document',
                    'timeline_data': [],
                    'total_events': 0,
                    'source_pdf_path': file_path
                }
            
            # Enhanced timeline data formatting with proper page number handling
            timeline_data = []
            for _, row in df.iterrows():
                # Extract page numbers more carefully
                page_numbers = []
                if 'Page Numbers' in row and row['Page Numbers']:
                    if isinstance(row['Page Numbers'], list):
                        page_numbers = row['Page Numbers']
                    elif isinstance(row['Page Numbers'], str):
                        # Parse page numbers from string like "Pages 1, 2, 3"
                        page_str = row['Page Numbers'].replace("Pages ", "").replace("Page ", "")
                        try:
                            page_numbers = [int(p.strip()) for p in page_str.split(",") if p.strip().isdigit()]
                        except:
                            page_numbers = [1]  # Default to page 1 if parsing fails
                    else:
                        page_numbers = [int(row['Page Numbers'])] if str(row['Page Numbers']).isdigit() else [1]
                
                # Ensure we have at least one page number
                if not page_numbers:
                    page_numbers = [1]
                
                entry = {
                    "date": str(row['Date']),
                    "event_details": str(row['Event Details']),
                    "source_document": document_name,
                    "page_numbers": page_numbers,
                    "source_pages": str(row.get('Source Pages', f"Page {page_numbers[0]}")),
                    "pdf_filename": os.path.basename(file_path)
                }
                
                timeline_data.append(entry)
            
            # Save Word document
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            word_filename = f"timeline_{document_name}_{timestamp}.docx"
            word_path = self.output_folder / word_filename
            doc.save(str(word_path))
            
            return {
                'success': True,
                'timeline_data': timeline_data,
                'total_events': len(timeline_data),
                'document_name': document_name,
                'word_document_path': str(word_path),
                'source_pdf_path': file_path
            }
            
        except Exception as e:
            logger.error(f"Error extracting timeline: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timeline_data': [],
                'total_events': 0
            }

    def extract_timeline(self, file_path: str, document_name: str = None) -> Dict[str, Any]:
        """Extract timeline from a PDF document (unchanged)"""
        return self.extract_timeline_enhanced(file_path, document_name)

    def process_selected_pdfs_timeline(self, selected_files: List[str], session_id: str = None) -> Dict[str, Any]:
        """Process selected PDF files for timeline extraction (unchanged)"""
        try:
            if not session_id or session_id not in self.session_folders:
                return {
                    'success': False,
                    'error': 'No valid session provided. Please upload a folder first.',
                    'processed_files': 0,
                    'failed_files': len(selected_files)
                }
            
            session_folder = Path(self.session_folders[session_id])
            all_results = []
            successful_files = []
            failed_files = []
            
            for file_path in selected_files:
                try:
                    # All files should be in the session folder
                    if not os.path.isabs(file_path):
                        abs_path = session_folder / file_path
                    else:
                        abs_path = Path(file_path)
                    
                    if not abs_path.exists():
                        failed_files.append(str(file_path))
                        continue
                    
                    result = self.extract_timeline_enhanced(str(abs_path))
                    if result['success']:
                        all_results.append(result)
                        successful_files.append(str(abs_path))
                    else:
                        failed_files.append(str(abs_path))
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    failed_files.append(str(file_path))
            
            return {
                'success': len(successful_files) > 0,
                'processed_files': len(successful_files),
                'failed_files': len(failed_files),
                'results': all_results,
                'successful_files': successful_files,
                'failed_files': failed_files
            }
            
        except Exception as e:
            logger.error(f"Error processing selected PDFs: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processed_files': 0,
                'failed_files': len(selected_files)
            }

    def save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file and return the path (unchanged)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            file_path = self.upload_folder / filename
            
            with open(file_path, "wb") as buffer:
                content = file.file.read()
                buffer.write(content)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    def cleanup_temp_files(self):
        """Clean up temporary files (unchanged)"""
        try:
            for file_path in self.temp_folder.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {str(e)}")

    # ====== NEW COLLECTION METHODS ======

    # Update the create_collection method around line 700

    def create_collection(self, collection_name: str, documents_folder: str, 
                        force_reindex: bool = False) -> Dict[str, Any]:
        """Create a new collection from uploaded documents"""
        try:
            # Create unique collection ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            collection_id = f"collection_{collection_name}_{timestamp}".replace(" ", "_").lower()
            
            # Create collection-specific folder
            collection_folder = self.collections_folder / collection_id
            collection_folder.mkdir(exist_ok=True)
            
            # Create new RAG system instance for this collection with unique collection name
            collection_rag = EnhancedLegalRAG(
                collection_name=collection_id,  # Use unique collection_id as Qdrant collection name
                qdrant_url="http://localhost:6333",
                data_dir=str(collection_folder)  # Use collection-specific data directory
            )
            
            # Process documents for this collection
            logger.info(f"Processing documents for collection: {collection_id}")
            result = collection_rag.process_directory(documents_folder, force_reindex)
            
            if result.get("success", False):
                # Store the collection system and metadata
                self.collection_systems[collection_id] = collection_rag
                self.collection_metadata[collection_id] = {
                    "collection_name": collection_name,
                    "collection_id": collection_id,
                    "documents_folder": documents_folder,
                    "created_at": datetime.now().isoformat(),
                    "total_files": result.get("total_files", 0),
                    "processed_files": result.get("processed_files", 0),
                    "failed_files": result.get("failed_files", 0),
                    "total_chunks": result.get("total_chunks", 0),
                    "qdrant_collection_name": collection_id  # Track Qdrant collection name
                }
                
                # Save metadata
                self._save_collections_metadata()
                
                logger.info(f"Successfully created isolated collection: {collection_id}")
                return {
                    "success": True,
                    "collection_id": collection_id,
                    "collection_name": collection_name,
                    "processed_files": result.get("processed_files", 0),
                    "total_chunks": result.get("total_chunks", 0),
                    "result": result
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to process documents: {result.get('error', 'Unknown error')}"
                }
                
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _calculate_collection_size(self, folder_path: str) -> float:
        """Calculate total size of files in MB"""
        try:
            total_size = 0
            for file_path in Path(folder_path).glob("*.pdf"):
                total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.error(f"Error calculating collection size: {str(e)}")
            return 0.0

    # Update the upload_and_create_collection method around line 780

    def upload_and_create_collection(self, files: List[UploadFile], collection_name: str, 
                                force_reindex: bool = False) -> Dict[str, Any]:
        """Upload files and create a new collection"""
        try:
            # Create upload session folder for this collection
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"collection_upload_{timestamp}"
            session_folder = self.upload_folder / session_id
            session_folder.mkdir(exist_ok=True)
            
            uploaded_files = []
            failed_files = []
            
            # Save uploaded files
            for file in files:
                try:
                    if not file.filename.lower().endswith('.pdf'):
                        failed_files.append(file.filename)
                        continue
                    
                    file_path = session_folder / file.filename
                    with open(file_path, "wb") as buffer:
                        content = file.file.read()
                        buffer.write(content)
                    
                    uploaded_files.append(file.filename)
                    logger.info(f"Uploaded file for collection: {file.filename}")
                    
                except Exception as e:
                    logger.error(f"Error uploading file {file.filename}: {str(e)}")
                    failed_files.append(file.filename)
            
            if not uploaded_files:
                return {
                    "success": False,
                    "error": "No valid PDF files were uploaded",
                    "collection_id": "",
                    "collection_name": collection_name,
                    "uploaded_files": [],
                    "failed_files": failed_files,
                    "file_count": 0,
                    "total_files": 0,  # Add this field
                    "processed_chunks": 0
                }
            
            # Create collection from uploaded files
            collection_result = self.create_collection(
                collection_name, 
                str(session_folder), 
                force_reindex
            )
            
            if collection_result["success"]:
                # Update metadata to include session folder
                collection_id = collection_result["collection_id"]
                self.collection_metadata[collection_id]["session_folder"] = str(session_folder)
                self._save_collections_metadata()
                
                return {
                    "success": True,
                    "collection_id": collection_id,
                    "collection_name": collection_name,
                    "uploaded_files": uploaded_files,
                    "failed_files": failed_files,
                    "file_count": len(uploaded_files),
                    "total_files": len(uploaded_files),  # Add this field
                    "processed_chunks": collection_result.get("total_chunks", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to create collection: {collection_result.get('error')}",
                    "collection_id": "",
                    "collection_name": collection_name,
                    "uploaded_files": uploaded_files,
                    "failed_files": failed_files,
                    "file_count": len(uploaded_files),
                    "total_files": len(uploaded_files),  # Add this field
                    "processed_chunks": 0
                }
                
        except Exception as e:
            logger.error(f"Error uploading and creating collection: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "collection_id": "",
                "collection_name": collection_name,
                "uploaded_files": [],
                "failed_files": [],
                "file_count": 0,
                "total_files": 0,  # Add this field
                "processed_chunks": 0
            }

    # Update the query_collection method around line 850

    def query_collection(self, query: str, collection_id: str, debug: bool = False, previous_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query a specific collection with context awareness"""
        try:
            if collection_id not in self.collection_systems:
                # Try to load collection if it exists but not loaded
                if collection_id in self.collection_metadata:
                    try:
                        collection_rag = EnhancedLegalRAG(
                            collection_name=collection_id,
                            qdrant_url="http://localhost:6333"
                        )
                        self.collection_systems[collection_id] = collection_rag
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Could not load collection '{collection_id}': {str(e)}"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Collection '{collection_id}' not found"
                    }
            
            # Get the RAG system for this collection
            collection_rag = self.collection_systems[collection_id]
            
            # IMPROVED: Pass previous context to the query
            response = collection_rag.query(query, debug=debug, previous_context=previous_context)
            
            # IMPROVED: Enhanced source information for collections
            enhanced_sources = []
            for source in response.get('sources', []):
                document_name = source.get('document', '')
                
                # Try multiple filename formats
                possible_filenames = [
                    f"{document_name}.pdf",
                    f"{document_name}",
                    document_name
                ]
                
                full_path = None
                for filename in possible_filenames:
                    full_path = self.get_pdf_file_path(filename, collection_id=collection_id)
                    if full_path:
                        break
                
                enhanced_source = {
                    'document': document_name,
                    'pages': source.get('pages', []),
                    'chunks': source.get('chunks', 0),
                    'max_score': source.get('max_score', 0),
                    'file_path': full_path,
                    'downloadable': full_path is not None,
                    'collection_id': collection_id,
                    'collection_name': self.collection_metadata[collection_id]["collection_name"]
                }
                enhanced_sources.append(enhanced_source)
            
            response['sources'] = enhanced_sources
            
            # Add collection information to response
            response["collection_id"] = collection_id
            response["collection_name"] = self.collection_metadata[collection_id]["collection_name"]
            response["queried_collection"] = True
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_collections_list(self) -> Dict[str, Any]:
        """Get list of all available collections"""
        try:
            collections = []
            for collection_id, metadata in self.collection_metadata.items():
                # Calculate current size
                if "session_folder" in metadata:
                    size_mb = self._calculate_collection_size(metadata["session_folder"])
                else:
                    size_mb = 0.0
                
                collection_info = {
                    "id": collection_id,
                    "name": metadata["collection_name"],
                    "file_count": metadata.get("total_files", 0),
                    "created_date": metadata.get("created_at", datetime.now().isoformat()),
                    "total_size_mb": size_mb,
                    "status": "loaded" if collection_id in self.collection_systems else "stored",
                    "processed_files": metadata.get("processed_files", 0),
                    "total_chunks": metadata.get("total_chunks", 0)
                }
                
                collections.append(collection_info)
            
            return {
                "success": True,
                "collections": collections,
                "total_collections": len(collections)
            }
            
        except Exception as e:
            logger.error(f"Error getting collections list: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "collections": [],
                "total_collections": 0
            }

    def delete_collection(self, collection_id: str) -> Dict[str, Any]:
        """Delete a collection"""
        try:
            if collection_id not in self.collection_metadata:
                return {
                    "success": False,
                    "error": f"Collection '{collection_id}' not found"
                }
            
            # Remove from Qdrant if loaded
            if collection_id in self.collection_systems:
                rag_system = self.collection_systems[collection_id]
                try:
                    rag_system.qdrant_client.delete_collection(collection_id)
                except Exception as e:
                    logger.warning(f"Could not delete Qdrant collection: {str(e)}")
                
                # Remove from memory
                del self.collection_systems[collection_id]
            
            # Remove metadata
            del self.collection_metadata[collection_id]
            
            # Clean up collection folder
            collection_folder = self.collections_folder / collection_id
            if collection_folder.exists():
                shutil.rmtree(collection_folder)
            
            # Save updated metadata
            self._save_collections_metadata()
            
            return {
                "success": True,
                "message": f"Collection '{collection_id}' deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get enhanced system statistics with collections information"""
        try:
            stats = {}
            
            # Original RAG system stats
            if self.rag_system:
                try:
                    original_stats = self.rag_system.get_system_stats()
                    stats['original_rag_system'] = original_stats
                except AttributeError:
                    # Handle case where get_system_stats doesn't exist
                    stats['original_rag_system'] = {
                        'vector_db': {'total_vectors': 0},
                        'bm25_index': {'total_documents': 0},
                        'system': {},
                        'validation_thresholds': {},
                        'supported_query_types': [],
                        'supported_intents': []
                    }
            
            # Collections stats
            stats['collections'] = {
                'total_collections': len(self.collection_metadata),
                'loaded_collections': len(self.collection_systems),
                'collections_list': list(self.collection_metadata.keys())
            }
            
            # Add original folder information
            stats['rag_documents_folder'] = {
                'path': str(self.rag_documents_folder),
                'exists': self.rag_documents_folder.exists(),
                'total_pdfs': len(list(self.rag_documents_folder.glob("**/*.pdf"))) if self.rag_documents_folder.exists() else 0
            }
            
            # Add timeline session information
            stats['timeline_sessions'] = {
                'active_sessions': len(self.session_folders),
                'session_ids': list(self.session_folders.keys())
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get system stats: {str(e)}")

# Initialize the enhanced system
integrated_system = EnhancedIntegratedLegalSystem()

# Create FastAPI app
app = FastAPI(
    title="Enhanced Legal System: Original RAG + Dynamic Collections",
    description="Legal document processing with original RAG system + separate collection management",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== ORIGINAL RAG ENDPOINTS (unchanged) ======

@app.post("/rag/query")
async def query_rag_system(request: QueryRequest):
    """Query the RAG system with conversation context - returns only the answer"""
    try:
        response = integrated_system.process_query(request.query, request.debug, simple_response=True)
        return response
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query-detailed", response_model=QueryResponse)
async def query_rag_system_detailed(request: QueryRequest):
    """Query the RAG system with conversation context - returns detailed response"""
    try:
        response = integrated_system.process_query(request.query, request.debug, simple_response=False)
        
        # Verify all required fields are present
        required_fields = ["query", "answer", "method", "query_type", "intent", "entities", "results_count", "sources"]
        missing_fields = [field for field in required_fields if field not in response]
        
        if missing_fields:
            # Add any missing fields with default values
            for field in missing_fields:
                if field == "query":
                    response["query"] = request.query
                elif field == "answer":
                    response["answer"] = "No answer available."
                elif field == "method":
                    response["method"] = "default_search"
                elif field == "query_type":
                    response["query_type"] = "general"
                elif field == "intent":
                    response["intent"] = "search"
                elif field == "entities":
                    response["entities"] = {}
                elif field == "results_count":
                    response["results_count"] = len(response.get("sources", []))
                elif field == "sources":
                    response["sources"] = []
        
        return QueryResponse(**response)
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        # Return a valid error response with all required fields
        error_response = {
            "query": request.query,
            "answer": f"Error processing query: {str(e)}",
            "method": "error",
            "query_type": "error",
            "intent": "error", 
            "entities": {},
            "results_count": 0,
            "sources": []
        }
        return QueryResponse(**error_response)
# New Conversation Context Management Endpoints
@app.get("/rag/conversation-history", response_model=ConversationHistoryResponse)
async def get_conversation_history():
    """Get current conversation history and context"""
    try:
        result = integrated_system.get_conversation_history()
        return ConversationHistoryResponse(**result)
    except Exception as e:
        logger.error(f"Getting conversation history failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/clear-context", response_model=ConversationContextResponse)
async def clear_conversation_context():
    """Clear current conversation context"""
    try:
        result = integrated_system.clear_conversation_context()
        return ConversationContextResponse(**result)
    except Exception as e:
        logger.error(f"Clearing conversation context failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/context-status")
async def get_context_status():
    """Get current conversation context status"""
    try:
        if (integrated_system.rag_system and 
            hasattr(integrated_system.rag_system, 'conversation_context')):
            context = integrated_system.rag_system.conversation_context
            return {
                "success": True,
                "active_context": bool(context.context_entities),
                "context_entities": context.context_entities,
                "conversation_count": len(context.conversation_history),
                "last_interaction": context.last_query_time.isoformat() if context.last_query_time else None,
                "context_timeout": context.context_timeout
            }
        else:
            return {
                "success": False,
                "error": "Conversation context not available"
            }
    except Exception as e:
        logger.error(f"Getting context status failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/process-document", response_model=DocumentProcessResponse)
async def process_document_rag(
    file: UploadFile = File(...),
    force_reindex: bool = Form(False)
):
    """Process a PDF document for RAG indexing"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        file_path = integrated_system.save_uploaded_file(file)
        
        # Process the document
        result = integrated_system.process_document_for_rag(file_path, force_reindex)
        return DocumentProcessResponse(**result)
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/process-directory", response_model=Dict[str, Any])
async def process_directory_rag(
    directory_path: str = Form(...),
    force_reindex: bool = Form(False)
):
    """Process a directory of PDF documents for RAG indexing"""
    try:
        # Validate directory path
        if not os.path.exists(directory_path):
            raise HTTPException(status_code=400, detail="Directory not found")
        
        result = integrated_system.process_directory_for_rag(directory_path, force_reindex)
        return result
    except Exception as e:
        logger.error(f"Directory processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/reindex-documents")
async def reindex_documents():
    """Reindex all documents in the RAG documents folder"""
    try:
        result = integrated_system.process_directory_for_rag(str(RAG_DOCUMENTS_FOLDER), force_reindex=True)
        return result
    except Exception as e:
        logger.error(f"Reindexing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== NEW COLLECTION ENDPOINTS ======

@app.post("/collections/upload", response_model=CollectionUploadResponse)
async def upload_and_create_collection(
    files: List[UploadFile] = File(...),
    collection_name: str = Form(...),
    force_reindex: bool = Form(False)
):
    """Upload files and create a new collection"""
    try:
        result = integrated_system.upload_and_create_collection(files, collection_name, force_reindex)
        return CollectionUploadResponse(**result)
    except Exception as e:
        logger.error(f"Collection upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/query")
async def query_collection(request: CollectionQueryRequest):
    """Query a specific collection"""
    try:
        response = integrated_system.query_collection(
            request.query, 
            request.collection_id, 
            request.debug
        )
        return response
    except Exception as e:
        logger.error(f"Collection query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/list", response_model=CollectionListResponse)
async def get_collections_list():
    """Get list of all available collections"""
    try:
        result = integrated_system.get_collections_list()
        return CollectionListResponse(**result)
    except Exception as e:
        logger.error(f"Getting collections list failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{collection_id}")
async def delete_collection(collection_id: str):
    """Delete a collection"""
    try:
        result = integrated_system.delete_collection(collection_id)
        return result
    except Exception as e:
        logger.error(f"Collection deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== PDF SERVING ENDPOINTS (unchanged) ======

# Update the serve_pdf_file endpoint around line 1350

@app.get("/pdf/serve/{filename}")
async def serve_pdf_file(
    filename: str, 
    session_id: str = Query(None),
    collection_id: str = Query(None)
):
    """Serve a PDF file for display in frontend - ENHANCED for collections"""
    try:
        file_path = integrated_system.get_pdf_file_path(filename, session_id, collection_id)
        if not file_path or not os.path.exists(file_path):
            # Try alternative filename formats
            alt_filename = filename.replace('.pdf', '') + '.pdf'
            file_path = integrated_system.get_pdf_file_path(alt_filename, session_id, collection_id)
            
            if not file_path or not os.path.exists(file_path):
                logger.error(f"PDF file not found: {filename}, session_id: {session_id}, collection_id: {collection_id}")
                raise HTTPException(status_code=404, detail=f"PDF file '{filename}' not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"PDF serving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdf/download/{filename}")
async def download_pdf_file(
    filename: str, 
    session_id: str = Query(None),
    collection_id: str = Query(None)
):
    """Download a PDF file - ENHANCED for collections"""
    try:
        file_path = integrated_system.get_pdf_file_path(filename, session_id, collection_id)
        if not file_path or not os.path.exists(file_path):
            logger.error(f"PDF file not found for download: {filename}")
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"PDF download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdf/download/{filename}")
async def download_pdf_file(filename: str, session_id: str = Query(None)):
    """Download a PDF file"""
    try:
        file_path = integrated_system.get_pdf_file_path(filename, session_id)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"PDF download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== TIMELINE ENDPOINTS (unchanged) ======

@app.post("/timeline/extract", response_model=TimelineResponse)
async def extract_timeline(
    file: UploadFile = File(...),
    document_name: Optional[str] = Form(None)
):
    """Extract timeline from a PDF document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        file_path = integrated_system.save_uploaded_file(file)
        
        # Use filename as document name if not provided
        if not document_name:
            document_name = Path(file.filename).stem
        
        # Extract timeline
        result = integrated_system.extract_timeline(file_path, document_name)
        
        return TimelineResponse(**result)
    except Exception as e:
        logger.error(f"Timeline extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/timeline/upload-folder", response_model=FolderUploadResponse)
async def upload_folder_for_timeline(files: List[UploadFile] = File(...)):
    """Upload multiple PDF files (representing a folder) for timeline processing"""
    try:
        result = integrated_system.upload_folder_files(files)
        return FolderUploadResponse(**result)
    except Exception as e:
        logger.error(f"Folder upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/timeline/folder-contents")
async def get_timeline_folder_contents(session_id: str = Query(None)):
    """Get list of PDF files in the uploaded folder session for timeline processing"""
    try:
        result = integrated_system.get_folder_contents(session_id)
        return result
    except Exception as e:
        logger.error(f"Getting folder contents failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/timeline/process-selected")
async def process_selected_pdfs_timeline(
    request: PDFSelectionRequest,
    session_id: str = Query(None)
):
    """Process selected PDF files for timeline extraction"""
    try:
        result = integrated_system.process_selected_pdfs_timeline(request.selected_files, session_id)
        return result
    except Exception as e:
        logger.error(f"Processing selected PDFs failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/timeline/pdf-preview/{filename}")
async def get_pdf_preview_endpoint(
    filename: str,
    page: int = Query(1, ge=1),
    session_id: str = Query(None)
):
    """Get PDF page preview as base64 image"""
    try:
        file_path = integrated_system.get_pdf_file_path(filename, session_id)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        preview_base64 = get_pdf_preview(file_path, page)
        if not preview_base64:
            raise HTTPException(status_code=500, detail="Failed to generate preview")
        
        return {
            "success": True,
            "filename": filename,
            "page": page,
            "preview": preview_base64
        }
    except Exception as e:
        logger.error(f"PDF preview failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/timeline/download/{filename}")
async def download_timeline_document(filename: str):
    """Download a generated timeline Word document"""
    try:
        file_path = integrated_system.output_folder / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception as e:
        logger.error(f"File download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== COMBINED PROCESSING ENDPOINTS (unchanged) ======

@app.post("/process-document-complete")
async def process_document_complete(
    file: UploadFile = File(...),
    force_reindex: bool = Form(False),
    extract_timeline: bool = Form(True),
    document_name: Optional[str] = Form(None)
):
    """Process a document for both RAG indexing and timeline extraction"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        file_path = integrated_system.save_uploaded_file(file)
        
        # Use filename as document name if not provided
        if not document_name:
            document_name = Path(file.filename).stem
        
        results = {}
        
        # Process for RAG
        try:
            rag_result = integrated_system.process_document_for_rag(file_path, force_reindex)
            results['rag_processing'] = rag_result
        except Exception as e:
            logger.error(f"RAG processing failed: {str(e)}")
            results['rag_processing'] = {'success': False, 'error': str(e)}
        
        # Extract timeline if requested
        if extract_timeline:
            try:
                timeline_result = integrated_system.extract_timeline(file_path, document_name)
                results['timeline_extraction'] = timeline_result
            except Exception as e:
                logger.error(f"Timeline extraction failed: {str(e)}")
                results['timeline_extraction'] = {'success': False, 'error': str(e)}
        
        return {
            'success': True,
            'document_name': document_name,
            'file_path': file_path,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Complete document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== SYSTEM ENDPOINTS (updated) ======

@app.get("/system/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """Get enhanced system statistics with conversation context"""
    try:
        stats = integrated_system.get_system_stats()
        return SystemStatsResponse(**stats)
    except Exception as e:
        logger.error(f"System stats failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/health")
async def health_check():
    """Enhanced health check endpoint with conversation context status"""
    try:
        # Check if both systems are initialized
        rag_status = integrated_system.rag_system is not None
        timeline_status = integrated_system.timeline_extractor is not None
        rag_documents_folder_exists = RAG_DOCUMENTS_FOLDER.exists()
        
        # Check conversation context status
        conversation_context_status = False
        active_context = False
        if (integrated_system.rag_system and 
            hasattr(integrated_system.rag_system, 'conversation_context')):
            conversation_context_status = True
            active_context = bool(integrated_system.rag_system.conversation_context.context_entities)
        
        return {
            'status': 'healthy' if (rag_status and timeline_status) else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'rag_system': 'active' if rag_status else 'inactive',
                'timeline_extractor': 'active' if timeline_status else 'inactive',
                'conversation_context': 'active' if conversation_context_status else 'inactive',
                'collections': f'{len(integrated_system.collection_systems)} loaded'
            },
            'configuration': {
                'rag_documents_folder': str(RAG_DOCUMENTS_FOLDER),
                'rag_documents_folder_exists': rag_documents_folder_exists,
                'rag_total_pdfs': len(list(RAG_DOCUMENTS_FOLDER.glob("**/*.pdf"))) if rag_documents_folder_exists else 0,
                'timeline_mode': 'upload_only',
                'active_timeline_sessions': len(integrated_system.session_folders),
                'conversation_context_enabled': conversation_context_status,
                'active_conversation_context': active_context,
                'total_collections': len(integrated_system.collection_metadata),
                'loaded_collections': len(integrated_system.collection_systems)
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with updated API information"""
    return {
        'message': 'Enhanced Legal System: Original RAG + Dynamic Collections',
        'version': '2.1.0',
        'rag_documents_folder': str(RAG_DOCUMENTS_FOLDER),
        'timeline_mode': 'upload_only',
        'conversation_context': 'enabled',
        'features': {
            'original_rag_system': True,
            'dynamic_collections': True,
            'conversation_context': True,
            'maximum_recall_search': True,
            'intent_recognition': True,
            'context_resolution': True,
            'reference_handling': True,
            'multi_strategy_search': True,
            'timeline_extraction': True
        },
        'endpoints': {
            'original_rag': {
                'query': '/rag/query',
                'query_detailed': '/rag/query-detailed',
                'process_document': '/rag/process-document',
                'process_directory': '/rag/process-directory',
                'reindex_documents': '/rag/reindex-documents',
                'conversation_history': '/rag/conversation-history',
                'clear_context': '/rag/clear-context',
                'context_status': '/rag/context-status'
            },
            'collections': {
                'upload': '/collections/upload',
                'query': '/collections/query',
                'list': '/collections/list',
                'delete': '/collections/{collection_id}'
            },
            'pdf_serving': {
                'serve_pdf': '/pdf/serve/{filename}',
                'download_pdf': '/pdf/download/{filename}'
            },
            'timeline': {
                'extract': '/timeline/extract',
                'upload_folder': '/timeline/upload-folder',
                'folder_contents': '/timeline/folder-contents',
                'process_selected': '/timeline/process-selected',
                'pdf_preview': '/timeline/pdf-preview/{filename}',
                'download': '/timeline/download/{filename}'
            },
            'combined': {
                'process_complete': '/process-document-complete'
            },
            'system': {
                'stats': '/system/stats',
                'health': '/system/health'
            }
        },
        'documentation': {
            'swagger': '/docs',
            'redoc': '/redoc'
        }
    }

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down application...")
    integrated_system.cleanup_temp_files()
    logger.info("Application shutdown complete")

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Legal System: Original RAG + Dynamic Collections")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Enhanced Legal System on {args.host}:{args.port}")
    logger.info(f"🚀 DUAL-MODE FEATURES:")
    logger.info(f"  📂 Original RAG Documents: {RAG_DOCUMENTS_FOLDER}")
    logger.info(f"  📁 Dynamic Collections: Upload & Query Independently")
    logger.info(f"  📋 Timeline Processing: Upload Only")
    logger.info(f"  💬 Conversation Context: Enabled")
    logger.info(f"  🔍 Collection Management: Real-time Processing")
    logger.info(f"API Documentation available at: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )
