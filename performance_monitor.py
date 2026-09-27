#!/usr/bin/env python3

import os
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import json
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import redis
from celery import Celery
import structlog

logger = structlog.get_logger()

# Prometheus metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total number of queries processed')
QUERY_DURATION = Histogram('rag_query_duration_seconds', 'Query processing duration')
HALLUCINATION_COUNTER = Counter('rag_hallucinations_total', 'Total number of hallucinations detected')
DOCUMENT_PROCESSING_DURATION = Histogram('rag_document_processing_duration_seconds', 'Document processing duration')
SYSTEM_MEMORY_USAGE = Gauge('rag_system_memory_bytes', 'System memory usage in bytes')
SYSTEM_CPU_USAGE = Gauge('rag_system_cpu_percent', 'System CPU usage percentage')

class PerformanceMonitor:
    """Monitors and optimizes system performance"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = None
        self.metrics = defaultdict(deque)
        self.max_metrics_history = 1000
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Initialize Redis if available
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established for performance monitoring")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
    
    def start_monitoring(self):
        """Start background performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Update Prometheus gauges
                SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().used)
                SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
                
                # Store metrics in Redis if available
                if self.redis_client:
                    self._store_metrics_in_redis()
                
                time.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self):
        """Collect current system metrics"""
        timestamp = datetime.now()
        
        # Memory usage
        memory = psutil.virtual_memory()
        self._add_metric('memory_usage_percent', memory.percent, timestamp)
        self._add_metric('memory_available_mb', memory.available / (1024 * 1024), timestamp)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self._add_metric('cpu_usage_percent', cpu_percent, timestamp)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self._add_metric('disk_usage_percent', (disk.used / disk.total) * 100, timestamp)
        
        # Network I/O
        network = psutil.net_io_counters()
        self._add_metric('network_bytes_sent', network.bytes_sent, timestamp)
        self._add_metric('network_bytes_recv', network.bytes_recv, timestamp)
    
    def _add_metric(self, metric_name: str, value: float, timestamp: datetime):
        """Add a metric to the local cache"""
        self.metrics[metric_name].append((timestamp, value))
        
        # Keep only recent metrics
        if len(self.metrics[metric_name]) > self.max_metrics_history:
            self.metrics[metric_name].popleft()
    
    def _store_metrics_in_redis(self):
        """Store metrics in Redis for persistence"""
        if not self.redis_client:
            return
        
        try:
            metrics_data = {}
            for metric_name, values in self.metrics.items():
                if values:
                    latest_value = values[-1][1]
                    metrics_data[metric_name] = {
                        'value': latest_value,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Store in Redis with expiration (24 hours)
            self.redis_client.setex(
                'rag_performance_metrics',
                86400,  # 24 hours
                json.dumps(metrics_data)
            )
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
    
    def record_query(self, query_text: str, processing_time: float, 
                    is_hallucination: bool, confidence_score: float):
        """Record a query for performance analysis"""
        # Update Prometheus metrics
        QUERY_COUNTER.inc()
        QUERY_DURATION.observe(processing_time)
        
        if is_hallucination:
            HALLUCINATION_COUNTER.inc()
        
        # Store detailed metrics
        timestamp = datetime.now()
        query_metrics = {
            'query_length': len(query_text),
            'processing_time': processing_time,
            'is_hallucination': is_hallucination,
            'confidence_score': confidence_score,
            'timestamp': timestamp.isoformat()
        }
        
        self._add_metric('query_processing_time', processing_time, timestamp)
        self._add_metric('query_confidence_score', confidence_score, timestamp)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                self.redis_client.lpush('rag_query_metrics', json.dumps(query_metrics))
                self.redis_client.ltrim('rag_query_metrics', 0, 999)  # Keep last 1000 queries
            except Exception as e:
                logger.error(f"Error storing query metrics: {e}")
    
    def record_document_processing(self, filename: str, file_size: int, 
                                 processing_time: float, segments_created: int):
        """Record document processing metrics"""
        DOCUMENT_PROCESSING_DURATION.observe(processing_time)
        
        timestamp = datetime.now()
        doc_metrics = {
            'filename': filename,
            'file_size_mb': file_size / (1024 * 1024),
            'processing_time': processing_time,
            'segments_created': segments_created,
            'timestamp': timestamp.isoformat()
        }
        
        self._add_metric('document_processing_time', processing_time, timestamp)
        self._add_metric('document_segments_created', segments_created, timestamp)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                self.redis_client.lpush('rag_document_metrics', json.dumps(doc_metrics))
                self.redis_client.ltrim('rag_document_metrics', 0, 999)
            except Exception as e:
                logger.error(f"Error storing document metrics: {e}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        summary = {
            'system_metrics': {},
            'query_metrics': {},
            'document_metrics': {},
            'recommendations': []
        }
        
        # Calculate averages for system metrics
        for metric_name, values in self.metrics.items():
            recent_values = [v for t, v in values if t >= cutoff_time]
            if recent_values:
                summary['system_metrics'][metric_name] = {
                    'average': sum(recent_values) / len(recent_values),
                    'min': min(recent_values),
                    'max': max(recent_values),
                    'count': len(recent_values)
                }
        
        # Get query metrics from Redis if available
        if self.redis_client:
            try:
                # Query metrics
                query_metrics = []
                for i in range(min(100, self.redis_client.llen('rag_query_metrics'))):
                    metric_data = self.redis_client.lindex('rag_query_metrics', i)
                    if metric_data:
                        metric = json.loads(metric_data)
                        if datetime.fromisoformat(metric['timestamp']) >= cutoff_time:
                            query_metrics.append(metric)
                
                if query_metrics:
                    processing_times = [m['processing_time'] for m in query_metrics]
                    confidence_scores = [m['confidence_score'] for m in query_metrics]
                    hallucination_count = sum(1 for m in query_metrics if m['is_hallucination'])
                    
                    summary['query_metrics'] = {
                        'total_queries': len(query_metrics),
                        'avg_processing_time': sum(processing_times) / len(processing_times),
                        'avg_confidence_score': sum(confidence_scores) / len(confidence_scores),
                        'hallucination_rate': (hallucination_count / len(query_metrics)) * 100
                    }
                
                # Document metrics
                doc_metrics = []
                for i in range(min(100, self.redis_client.llen('rag_document_metrics'))):
                    metric_data = self.redis_client.lindex('rag_document_metrics', i)
                    if metric_data:
                        metric = json.loads(metric_data)
                        if datetime.fromisoformat(metric['timestamp']) >= cutoff_time:
                            doc_metrics.append(metric)
                
                if doc_metrics:
                    processing_times = [m['processing_time'] for m in doc_metrics]
                    segments_created = [m['segments_created'] for m in doc_metrics]
                    
                    summary['document_metrics'] = {
                        'total_documents': len(doc_metrics),
                        'avg_processing_time': sum(processing_times) / len(processing_times),
                        'total_segments': sum(segments_created),
                        'avg_segments_per_doc': sum(segments_created) / len(segments_created)
                    }
                    
            except Exception as e:
                logger.error(f"Error retrieving metrics from Redis: {e}")
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(summary)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics"""
        recommendations = []
        
        # Memory recommendations
        if 'memory_usage_percent' in summary['system_metrics']:
            avg_memory = summary['system_metrics']['memory_usage_percent']['average']
            if avg_memory > 80:
                recommendations.append("High memory usage detected. Consider increasing system memory or optimizing document processing.")
            elif avg_memory > 60:
                recommendations.append("Moderate memory usage. Monitor for potential optimization opportunities.")
        
        # CPU recommendations
        if 'cpu_usage_percent' in summary['system_metrics']:
            avg_cpu = summary['system_metrics']['cpu_usage_percent']['average']
            if avg_cpu > 80:
                recommendations.append("High CPU usage detected. Consider using more efficient models or parallel processing.")
        
        # Query performance recommendations
        if 'query_metrics' in summary and summary['query_metrics']:
            avg_query_time = summary['query_metrics']['avg_processing_time']
            if avg_query_time > 10:
                recommendations.append("Slow query processing detected. Consider optimizing retrieval or using faster models.")
            
            hallucination_rate = summary['query_metrics']['hallucination_rate']
            if hallucination_rate > 20:
                recommendations.append("High hallucination rate detected. Consider improving prompt engineering or using better models.")
        
        # Document processing recommendations
        if 'document_metrics' in summary and summary['document_metrics']:
            avg_doc_time = summary['document_metrics']['avg_processing_time']
            if avg_doc_time > 60:
                recommendations.append("Slow document processing detected. Consider using parallel processing or optimizing chunking strategy.")
        
        if not recommendations:
            recommendations.append("System performance is within normal parameters.")
        
        return recommendations
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest()

# Celery configuration for background tasks
celery_app = Celery('rag_tasks', broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'))

@celery_app.task
def process_document_background(file_path: str, file_type: str, user_id: int):
    """Background task for document processing"""
    from rag_core import RAGSystem
    from database import get_db_manager
    
    start_time = time.time()
    
    try:
        # Initialize RAG system
        rag_system = RAGSystem()
        rag_system.initialize_system()
        
        # Process document
        # This would need to be adapted based on your actual document processing logic
        processing_time = time.time() - start_time
        
        # Record metrics
        monitor = PerformanceMonitor()
        monitor.record_document_processing(
            filename=os.path.basename(file_path),
            file_size=os.path.getsize(file_path),
            processing_time=processing_time,
            segments_created=0  # This would be the actual count
        )
        
        return {
            'status': 'success',
            'processing_time': processing_time,
            'segments_created': 0
        }
        
    except Exception as e:
        logger.error(f"Error in background document processing: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

# Global performance monitor instance
performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global performance_monitor
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor()
        performance_monitor.start_monitoring()
    return performance_monitor
