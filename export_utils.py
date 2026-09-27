#!/usr/bin/env python3

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import pandas as pd
from database import DatabaseManager, get_db_manager
import logging

logger = logging.getLogger(__name__)

class ExportManager:
    """Manages various export formats for RAG responses and documents"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for reports"""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_LEFT
        )
        
        self.caption_style = ParagraphStyle(
            'CustomCaption',
            parent=self.styles['Normal'],
            fontSize=9,
            spaceAfter=3,
            textColor=colors.grey
        )
    
    def export_query_to_pdf(self, query_data: Dict, output_path: str = None) -> str:
        """Export a single query and response to PDF"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"query_report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title
        title = Paragraph("RAG Query Report", self.title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Query section
        story.append(Paragraph("Question", self.heading_style))
        story.append(Paragraph(query_data['query'], self.body_style))
        story.append(Spacer(1, 12))
        
        # Response section
        story.append(Paragraph("Answer", self.heading_style))
        story.append(Paragraph(query_data['response'], self.body_style))
        story.append(Spacer(1, 12))
        
        # Analysis section
        confidence = query_data.get('hallucination_confidence', 0)
        grounding = max(0.0, min(1.0, 1.0 - confidence))
        
        analysis_data = [
            ['Metric', 'Value'],
            ['Grounding Confidence', f"{grounding:.2f}"],
            ['Hallucination Detected', 'Yes' if query_data.get('is_hallucination') else 'No'],
            ['Processing Time', f"{query_data.get('processing_time', 0):.2f}s"],
            ['Timestamp', query_data.get('time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))]
        ]
        
        analysis_table = Table(analysis_data, colWidths=[2*inch, 3*inch])
        analysis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("Analysis", self.heading_style))
        story.append(analysis_table)
        story.append(Spacer(1, 12))
        
        # Sources section
        sources = query_data.get('source_docs', [])
        if sources:
            story.append(Paragraph("Source Documents", self.heading_style))
            
            for i, doc in enumerate(sources[:5], 1):
                source_text = f"Source {i}: {doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page_number', 'N/A')})"
                story.append(Paragraph(source_text, self.caption_style))
                
                # Add a preview of the source content
                preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                story.append(Paragraph(preview, self.body_style))
                story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated: {output_path}")
        return output_path
    
    def export_user_report_to_pdf(self, user_id: int, output_path: str = None) -> str:
        """Export comprehensive user activity report to PDF"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"user_report_{user_id}_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Get user data
        with self.db_manager.get_session() as session:
            user = session.query(self.db_manager.User).filter(self.db_manager.User.id == user_id).first()
            if not user:
                raise ValueError("User not found")
            
            documents = self.db_manager.get_user_documents(user_id)
            queries = self.db_manager.get_recent_queries(user_id, 100)
        
        # Title
        title = Paragraph(f"User Activity Report - {user.full_name or user.username}", self.title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # User info
        story.append(Paragraph("User Information", self.heading_style))
        user_info = [
            ['Field', 'Value'],
            ['Username', user.username],
            ['Email', user.email],
            ['Role', user.role],
            ['Member Since', user.created_at.strftime('%Y-%m-%d')],
            ['Last Login', user.last_login.strftime('%Y-%m-%d %H:%M') if user.last_login else 'Never']
        ]
        
        user_table = Table(user_info, colWidths=[1.5*inch, 3.5*inch])
        user_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(user_table)
        story.append(Spacer(1, 20))
        
        # Statistics
        total_queries = len(queries)
        hallucination_count = sum(1 for q in queries if q.is_hallucination)
        avg_processing_time = sum(q.processing_time for q in queries) / total_queries if total_queries > 0 else 0
        
        story.append(Paragraph("Activity Statistics", self.heading_style))
        stats_data = [
            ['Metric', 'Value'],
            ['Total Documents', len(documents)],
            ['Total Queries', total_queries],
            ['Hallucination Rate', f"{(hallucination_count/total_queries*100):.1f}%" if total_queries > 0 else "0%"],
            ['Average Processing Time', f"{avg_processing_time:.0f}ms" if total_queries > 0 else "0ms"]
        ]
        
        stats_table = Table(stats_data, colWidths=[2*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        # Recent queries
        if queries:
            story.append(Paragraph("Recent Queries", self.heading_style))
            
            for i, query in enumerate(queries[:10], 1):
                story.append(Paragraph(f"Query {i}: {query.query_text[:100]}...", self.body_style))
                story.append(Paragraph(f"Response: {query.response_text[:200]}...", self.body_style))
                story.append(Paragraph(f"Confidence: {query.confidence_score}% | Time: {query.processing_time}ms | Date: {query.created_at.strftime('%Y-%m-%d')}", self.caption_style))
                story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        logger.info(f"User report PDF generated: {output_path}")
        return output_path
    
    def export_to_markdown(self, query_data: Dict, include_sources: bool = True) -> str:
        """Export query and response to markdown format"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md_content = f"""# RAG Query Report

**Generated:** {timestamp}

## Question
{query_data['query']}

## Answer
{query_data['response']}

## Analysis
- **Grounding Confidence:** {(1 - query_data.get('hallucination_confidence', 0)):.2f}
- **Hallucination Detected:** {'Yes' if query_data.get('is_hallucination') else 'No'}
- **Processing Time:** {query_data.get('processing_time', 0):.2f}s
- **Query Time:** {query_data.get('time', timestamp)}

"""
        
        if include_sources:
            sources = query_data.get('source_docs', [])
            if sources:
                md_content += "## Source Documents\n\n"
                for i, doc in enumerate(sources, 1):
                    md_content += f"### Source {i}\n"
                    md_content += f"- **File:** {doc.metadata.get('filename', 'Unknown')}\n"
                    md_content += f"- **Page:** {doc.metadata.get('page_number', 'N/A')}\n"
                    md_content += f"- **Content:** {doc.page_content[:500]}...\n\n"
        
        return md_content
    
    def export_to_json(self, query_data: Dict, include_metadata: bool = True) -> str:
        """Export query and response to JSON format"""
        export_data = {
            "query": query_data['query'],
            "response": query_data['response'],
            "analysis": {
                "grounding_confidence": 1 - query_data.get('hallucination_confidence', 0),
                "is_hallucination": query_data.get('is_hallucination', False),
                "processing_time": query_data.get('processing_time', 0),
                "timestamp": query_data.get('time', datetime.now().isoformat())
            }
        }
        
        if include_metadata:
            sources = query_data.get('source_docs', [])
            export_data["sources"] = [
                {
                    "filename": doc.metadata.get('filename', 'Unknown'),
                    "page_number": doc.metadata.get('page_number', 0),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                for doc in sources
            ]
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def export_to_csv(self, queries: List[Dict], output_path: str = None) -> str:
        """Export multiple queries to CSV format"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"queries_export_{timestamp}.csv"
        
        data = []
        for query in queries:
            data.append({
                'Query': query['query'],
                'Response': query['response'],
                'Grounding_Confidence': 1 - query.get('hallucination_confidence', 0),
                'Is_Hallucination': query.get('is_hallucination', False),
                'Processing_Time': query.get('processing_time', 0),
                'Timestamp': query.get('time', ''),
                'Source_Count': len(query.get('source_docs', []))
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"CSV export generated: {output_path}")
        return output_path
    
    def export_document_changes_report(self, document_id: int, days: int = 30, output_path: str = None) -> str:
        """Export document change history to PDF"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"document_changes_{document_id}_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Get document and changes
        with self.db_manager.get_session() as session:
            document = session.query(self.db_manager.Document).filter(self.db_manager.Document.id == document_id).first()
            if not document:
                raise ValueError("Document not found")
            
            changes = self.db_manager.get_document_changes(document_id, days)
        
        # Title
        title = Paragraph(f"Document Change History - {document.filename}", self.title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Document info
        story.append(Paragraph("Document Information", self.heading_style))
        doc_info = [
            ['Field', 'Value'],
            ['Filename', document.filename],
            ['File Type', document.file_type],
            ['Current Version', str(document.version)],
            ['Status', document.status],
            ['Created', document.created_at.strftime('%Y-%m-%d %H:%M')],
            ['Last Updated', document.updated_at.strftime('%Y-%m-%d %H:%M')]
        ]
        
        doc_table = Table(doc_info, colWidths=[1.5*inch, 3.5*inch])
        doc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(doc_table)
        story.append(Spacer(1, 20))
        
        # Changes history
        if changes:
            story.append(Paragraph(f"Change History (Last {days} days)", self.heading_style))
            
            changes_data = [['Date', 'Type', 'Details']]
            for change in changes:
                changes_data.append([
                    change.created_at.strftime('%Y-%m-%d %H:%M'),
                    change.change_type.title(),
                    json.dumps(change.change_details) if change.change_details else 'N/A'
                ])
            
            changes_table = Table(changes_data, colWidths=[1.5*inch, 1*inch, 2.5*inch])
            changes_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9)
            ]))
            story.append(changes_table)
        else:
            story.append(Paragraph("No changes recorded in the specified time period.", self.body_style))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Document changes report generated: {output_path}")
        return output_path

# Global export manager instance
export_manager = None

def get_export_manager() -> ExportManager:
    """Get the global export manager instance"""
    global export_manager
    if export_manager is None:
        export_manager = ExportManager()
    return export_manager
