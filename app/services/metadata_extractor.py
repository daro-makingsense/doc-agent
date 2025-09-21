"""
Metadata extraction service for financial documents.
Extracts key financial facts, investment data, and metrics from processed documents.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import json

from openai import AsyncOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.config import get_settings
from app.database.models import Document, DocumentChunk
from app.database.connection import get_db_session

settings = get_settings()
logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Service for extracting financial metadata from documents."""
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Financial patterns for regex extraction
        self.financial_patterns = {
            'revenue': [
                r'revenue[s]?\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|billion|b)?',
                r'net\s+revenue[s]?\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|billion|b)?',
                r'total\s+revenue[s]?\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|billion|b)?',
            ],
            'profit': [
                r'net\s+(?:profit|income)[s]?\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|billion|b)?',
                r'profit[s]?\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|billion|b)?',
                r'earnings\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|billion|b)?',
            ],
            'ebitda': [
                r'ebitda\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|billion|b)?',
                r'adjusted\s+ebitda\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|billion|b)?',
            ],
            'valuation': [
                r'valuation\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|billion|b)?',
                r'enterprise\s+value\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|billion|b)?',
                r'market\s+cap(?:italization)?\s*(?:of|:)?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|billion|b)?',
            ]
        }
    
    async def extract_metadata(self, document_id: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from a document.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            logger.info(f"Starting metadata extraction for document {document_id}")
            
            # Get document and chunks
            with get_db_session() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                if not document:
                    raise ValueError(f"Document {document_id} not found")
                
                chunks = db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document_id
                ).order_by(DocumentChunk.chunk_index).all()
            
            # Combine all chunks for full text analysis
            full_text = "\n\n".join([chunk.content for chunk in chunks])
            
            # Extract metadata using multiple methods
            results = await asyncio.gather(
                self._extract_financial_facts_ai(full_text),
                self._extract_investment_data_ai(full_text),
                self._extract_key_metrics_regex(full_text),
                self._extract_document_structure(full_text),
                return_exceptions=True
            )
            
            financial_facts = results[0] if not isinstance(results[0], Exception) else {}
            investment_data = results[1] if not isinstance(results[1], Exception) else {}
            key_metrics = results[2] if not isinstance(results[2], Exception) else {}
            structure_info = results[3] if not isinstance(results[3], Exception) else {}
            
            # Combine all metadata
            metadata = {
                'financial_facts': financial_facts,
                'investment_data': investment_data,
                'key_metrics': key_metrics,
                'document_structure': structure_info,
                'extraction_timestamp': str(asyncio.get_event_loop().time())
            }
            
            # Update document in database
            with get_db_session() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.financial_facts = financial_facts
                    document.investment_data = investment_data
                    document.key_metrics = {**key_metrics, **structure_info}
            
            logger.info(f"Metadata extraction completed for document {document_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata for document {document_id}: {str(e)}")
            raise e
    
    async def _extract_financial_facts_ai(self, text: str) -> Dict[str, Any]:
        """Extract financial facts using OpenAI GPT."""
        system_prompt = """
        You are a financial analyst expert. Extract key financial facts from the provided document text.
        Focus on numerical data like revenue, profit, losses, expenses, cash flow, debt, etc.
        
        Return the results as a valid JSON object with the following structure:
        {
            "revenue": {
                "current_year": number or null,
                "previous_year": number or null,
                "currency": "USD" or other,
                "period": "annual" or "quarterly" or "monthly"
            },
            "profit_loss": {
                "net_income": number or null,
                "gross_profit": number or null,
                "operating_profit": number or null,
                "currency": "USD" or other
            },
            "cash_flow": {
                "operating_cash_flow": number or null,
                "free_cash_flow": number or null,
                "currency": "USD" or other
            },
            "debt_equity": {
                "total_debt": number or null,
                "equity": number or null,
                "debt_to_equity_ratio": number or null
            },
            "other_metrics": {
                "ebitda": number or null,
                "margin_percentage": number or null,
                "growth_rate": number or null
            }
        }
        
        If a value is not found or unclear, use null. All monetary values should be in the base unit (e.g., if document says "$5.2M", return 5200000).
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract financial facts from this document:\n\n{text[:4000]}..."}  # Limit text length
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    logger.warning("Could not parse financial facts JSON response")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error in AI financial facts extraction: {str(e)}")
            return {}
    
    async def _extract_investment_data_ai(self, text: str) -> Dict[str, Any]:
        """Extract investment-related data using OpenAI GPT."""
        system_prompt = """
        You are an investment analyst. Extract investment-related information from the provided document.
        Focus on investment highlights, risks, opportunities, market data, and strategic information.
        
        Return results as a valid JSON object:
        {
            "investment_highlights": [
                "Key point 1",
                "Key point 2"
            ],
            "risk_factors": [
                "Risk 1",
                "Risk 2"
            ],
            "market_opportunity": {
                "market_size": number or null,
                "growth_rate": number or null,
                "competitive_position": "string or null"
            },
            "business_model": {
                "type": "string or null",
                "revenue_streams": ["stream1", "stream2"],
                "key_customers": ["customer1", "customer2"]
            },
            "strategic_initiatives": [
                "Initiative 1",
                "Initiative 2"
            ],
            "exit_strategy": {
                "timeline": "string or null",
                "target_multiple": number or null,
                "potential_buyers": ["buyer1", "buyer2"]
            }
        }
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract investment data from this document:\n\n{text[:4000]}..."}
                ],
                temperature=0.2,
                max_tokens=1200
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Error in AI investment data extraction: {str(e)}")
            return {}
    
    def _extract_key_metrics_regex(self, text: str) -> Dict[str, Any]:
        """Extract key financial metrics using regex patterns."""
        metrics = {}
        
        # Convert text to lowercase for pattern matching
        text_lower = text.lower()
        
        # Extract financial figures using patterns
        for category, patterns in self.financial_patterns.items():
            values = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    try:
                        # Clean and convert the number
                        value_str = match.group(1).replace(',', '')
                        value = float(value_str)
                        
                        # Check for scale indicators in the surrounding text
                        context = text_lower[max(0, match.start()-20):match.end()+20]
                        if 'million' in context or ' m ' in context:
                            value *= 1_000_000
                        elif 'billion' in context or ' b ' in context:
                            value *= 1_000_000_000
                        
                        values.append(value)
                        
                    except (ValueError, IndexError):
                        continue
            
            if values:
                metrics[category] = {
                    'values': values,
                    'primary_value': max(values),  # Use the highest value as primary
                    'count': len(values)
                }
        
        # Extract percentages
        percentage_patterns = [
            r'(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)\s*percent'
        ]
        
        percentages = []
        for pattern in percentage_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                try:
                    percentages.append(float(match.group(1)))
                except ValueError:
                    continue
        
        if percentages:
            metrics['percentages'] = {
                'values': percentages,
                'count': len(percentages)
            }
        
        return metrics
    
    def _extract_document_structure(self, text: str) -> Dict[str, Any]:
        """Extract document structure information."""
        structure = {}
        
        # Count sections (indicated by headers or numbering)
        section_patterns = [
            r'^[0-9]+\.\s+[A-Z]',  # 1. SECTION
            r'^[IVX]+\.\s+[A-Z]',   # I. SECTION  
            r'^[A-Z][A-Z\s]+$',     # ALL CAPS HEADERS
        ]
        
        sections = 0
        for pattern in section_patterns:
            sections += len(re.findall(pattern, text, re.MULTILINE))
        
        structure['estimated_sections'] = sections
        
        # Count tables (rough estimation)
        table_indicators = ['table', 'figure', '|', '\t']
        table_score = sum(text.lower().count(indicator) for indicator in table_indicators)
        structure['estimated_tables'] = min(table_score // 10, 50)  # Cap at 50
        
        # Count bullet points
        bullet_patterns = [r'^\s*[•\-\*]\s+', r'^\s*\d+\.\s+']
        bullet_points = 0
        for pattern in bullet_patterns:
            bullet_points += len(re.findall(pattern, text, re.MULTILINE))
        
        structure['bullet_points'] = bullet_points
        
        # Estimate reading time (average 200 words per minute)
        word_count = len(text.split())
        structure['estimated_reading_time_minutes'] = max(1, word_count // 200)
        
        # Document complexity score (0-10)
        complexity = min(10, (
            sections * 0.5 + 
            structure['estimated_tables'] * 0.3 + 
            bullet_points * 0.1 + 
            word_count / 1000
        ))
        structure['complexity_score'] = round(complexity, 1)
        
        return structure
    
    async def summarize_document(self, document_id: str, max_length: int = 500) -> str:
        """Generate a concise summary of the document."""
        try:
            # Get document content
            with get_db_session() as db:
                chunks = db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document_id
                ).order_by(DocumentChunk.chunk_index).limit(5).all()  # First 5 chunks
            
            if not chunks:
                return "No content available for summarization."
            
            text = "\n\n".join([chunk.content for chunk in chunks])
            
            system_prompt = f"""
            Create a concise summary of this financial document in {max_length} words or less.
            Focus on the key business information, financial highlights, and main value propositions.
            Make it suitable for executive review.
            """
            
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize this document:\n\n{text[:3000]}..."}
                ],
                temperature=0.3,
                max_tokens=max_length // 2  # Rough token estimation
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary for document {document_id}: {str(e)}")
            return f"Error generating summary: {str(e)}"


# Global instance
metadata_extractor = MetadataExtractor()