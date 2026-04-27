"""
Elasticsearch tenant isolation OXM module

This module provides tenant isolation support for Elasticsearch, including:
- TenantAwareAsyncDocument: Tenant-aware asynchronous document class
- TenantAwareAliasDoc: Factory function for tenant-aware alias document class
- Configuration utility functions: Get tenant ES configuration, generate connection cache key, etc.
"""
