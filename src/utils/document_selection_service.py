"""
DocumentSelectionService - Manages 3-tier document selection system.

Implements the Document Selection requirements from the consolidate-to-postgres spec:
- System default (all enabled)
- User default override
- Conversation override
- Effective selection query
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import psycopg2
import psycopg2.extras

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentSelection:
    """Document selection state for a document."""
    
    document_id: int
    resource_hash: str
    display_name: str
    source_type: str
    
    # Selection state at each tier
    system_default: bool = True  # Always True (system default = enabled)
    user_default: Optional[bool] = None  # None = use system default
    conversation_override: Optional[bool] = None  # None = use user/system default
    
    # Computed effective state
    @property
    def enabled(self) -> bool:
        """Get effective enabled state using 3-tier precedence."""
        if self.conversation_override is not None:
            return self.conversation_override
        if self.user_default is not None:
            return self.user_default
        return self.system_default


class DocumentSelectionService:
    """
    Service for managing document selection in the 3-tier system.
    
    Tier precedence (highest to lowest):
    1. Conversation override (per-conversation setting)
    2. User default (user's global preference)
    3. System default (all documents enabled)
    
    Example:
        >>> service = DocumentSelectionService(pg_config)
        >>> # Disable a document globally for a user
        >>> service.set_user_default(user_id, doc_id, enabled=False)
        >>> # Re-enable for a specific conversation
        >>> service.set_conversation_override(conversation_id, doc_id, enabled=True)
        >>> # Get effective document set for search
        >>> enabled_ids = service.get_enabled_document_ids(user_id, conversation_id)
    """
    
    def __init__(self, pg_config: Optional[Dict[str, Any]] = None, *, connection_pool=None):
        """
        Initialize DocumentSelectionService.
        
        Args:
            pg_config: PostgreSQL connection parameters (fallback)
            connection_pool: ConnectionPool instance (preferred)
        """
        self._pool = connection_pool
        self._pg_config = pg_config
    
    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get a database connection."""
        if self._pool:
            return self._pool.get_connection_direct()
        elif self._pg_config:
            return psycopg2.connect(**self._pg_config)
        else:
            raise ValueError("No connection pool or pg_config provided")
    
    def _release_connection(self, conn) -> None:
        """Release connection back to pool or close it."""
        if self._pool:
            self._pool.release_connection(conn)
        else:
            conn.close()
    
    # =========================================================================
    # User Document Defaults
    # =========================================================================
    
    def get_user_defaults(
        self,
        user_id: str,
        *,
        include_enabled: bool = True,
    ) -> List[DocumentSelection]:
        """
        Get user's document default selections.
        
        Implements: GET user document defaults
        
        Args:
            user_id: The user's ID
            include_enabled: If True, include documents with no explicit setting
                           (showing system default = enabled)
                           
        Returns:
            List of DocumentSelection objects
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                if include_enabled:
                    # Return all non-deleted documents with their selection state
                    cursor.execute(
                        """
                        SELECT 
                            d.id as document_id,
                            d.resource_hash,
                            d.display_name,
                            d.source_type,
                            ud.enabled as user_default
                        FROM documents d
                        LEFT JOIN user_document_defaults ud 
                            ON d.id = ud.document_id AND ud.user_id = %s
                        WHERE d.is_deleted = FALSE
                        ORDER BY d.display_name
                        """,
                        (user_id,)
                    )
                else:
                    # Only return documents with explicit user settings
                    cursor.execute(
                        """
                        SELECT 
                            d.id as document_id,
                            d.resource_hash,
                            d.display_name,
                            d.source_type,
                            ud.enabled as user_default
                        FROM documents d
                        INNER JOIN user_document_defaults ud 
                            ON d.id = ud.document_id AND ud.user_id = %s
                        WHERE d.is_deleted = FALSE
                        ORDER BY d.display_name
                        """,
                        (user_id,)
                    )
                
                rows = cursor.fetchall()
                
                return [
                    DocumentSelection(
                        document_id=row["document_id"],
                        resource_hash=row["resource_hash"],
                        display_name=row["display_name"],
                        source_type=row["source_type"],
                        user_default=row["user_default"],
                    )
                    for row in rows
                ]
        finally:
            self._release_connection(conn)
    
    def set_user_default(
        self,
        user_id: str,
        document_id: int,
        enabled: bool,
    ) -> DocumentSelection:
        """
        Set user's default selection for a document.
        
        Implements: PUT user document default
        
        Args:
            user_id: The user's ID
            document_id: The document ID
            enabled: Whether the document should be enabled by default
            
        Returns:
            Updated DocumentSelection
            
        Raises:
            ValueError: If document not found
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Upsert user default
                cursor.execute(
                    """
                    INSERT INTO user_document_defaults (user_id, document_id, enabled)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, document_id) DO UPDATE SET
                        enabled = EXCLUDED.enabled,
                        updated_at = NOW()
                    """,
                    (user_id, document_id, enabled)
                )
                
                # Get document details
                cursor.execute(
                    """
                    SELECT id, resource_hash, display_name, source_type
                    FROM documents
                    WHERE id = %s AND is_deleted = FALSE
                    """,
                    (document_id,)
                )
                doc = cursor.fetchone()
                conn.commit()
                
                if doc is None:
                    raise ValueError(f"Document not found: {document_id}")
                
                logger.debug(f"Set user default: user={user_id}, doc={document_id}, enabled={enabled}")
                
                return DocumentSelection(
                    document_id=doc["id"],
                    resource_hash=doc["resource_hash"],
                    display_name=doc["display_name"],
                    source_type=doc["source_type"],
                    user_default=enabled,
                )
        finally:
            self._release_connection(conn)
    
    def clear_user_default(
        self,
        user_id: str,
        document_id: int,
    ) -> bool:
        """
        Clear user's default selection for a document (revert to system default).
        
        Args:
            user_id: The user's ID
            document_id: The document ID
            
        Returns:
            True if a record was deleted
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM user_document_defaults
                    WHERE user_id = %s AND document_id = %s
                    """,
                    (user_id, document_id)
                )
                conn.commit()
                return cursor.rowcount > 0
        finally:
            self._release_connection(conn)
    
    # =========================================================================
    # Conversation Document Overrides
    # =========================================================================
    
    def get_conversation_overrides(
        self,
        conversation_id: str,
    ) -> List[DocumentSelection]:
        """
        Get conversation-specific document overrides.
        
        Implements: GET conversation document overrides
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            List of DocumentSelection objects with overrides
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT 
                        d.id as document_id,
                        d.resource_hash,
                        d.display_name,
                        d.source_type,
                        co.enabled as conversation_override
                    FROM documents d
                    INNER JOIN conversation_document_overrides co 
                        ON d.id = co.document_id AND co.conversation_id = %s
                    WHERE d.is_deleted = FALSE
                    ORDER BY d.display_name
                    """,
                    (conversation_id,)
                )
                
                rows = cursor.fetchall()
                
                return [
                    DocumentSelection(
                        document_id=row["document_id"],
                        resource_hash=row["resource_hash"],
                        display_name=row["display_name"],
                        source_type=row["source_type"],
                        conversation_override=row["conversation_override"],
                    )
                    for row in rows
                ]
        finally:
            self._release_connection(conn)
    
    def set_conversation_override(
        self,
        conversation_id: str,
        document_id: int,
        enabled: bool,
    ) -> DocumentSelection:
        """
        Set conversation-specific document override.
        
        Implements: PUT conversation document override
        
        Args:
            conversation_id: The conversation ID
            document_id: The document ID
            enabled: Whether the document should be enabled for this conversation
            
        Returns:
            Updated DocumentSelection
            
        Raises:
            ValueError: If document not found
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Upsert conversation override
                cursor.execute(
                    """
                    INSERT INTO conversation_document_overrides (conversation_id, document_id, enabled)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (conversation_id, document_id) DO UPDATE SET
                        enabled = EXCLUDED.enabled,
                        updated_at = NOW()
                    """,
                    (conversation_id, document_id, enabled)
                )
                
                # Get document details
                cursor.execute(
                    """
                    SELECT id, resource_hash, display_name, source_type
                    FROM documents
                    WHERE id = %s AND is_deleted = FALSE
                    """,
                    (document_id,)
                )
                doc = cursor.fetchone()
                conn.commit()
                
                if doc is None:
                    raise ValueError(f"Document not found: {document_id}")
                
                logger.debug(
                    f"Set conversation override: conv={conversation_id}, doc={document_id}, enabled={enabled}"
                )
                
                return DocumentSelection(
                    document_id=doc["id"],
                    resource_hash=doc["resource_hash"],
                    display_name=doc["display_name"],
                    source_type=doc["source_type"],
                    conversation_override=enabled,
                )
        finally:
            self._release_connection(conn)
    
    def clear_conversation_override(
        self,
        conversation_id: str,
        document_id: int,
    ) -> bool:
        """
        Clear conversation-specific override (revert to user/system default).
        
        Args:
            conversation_id: The conversation ID
            document_id: The document ID
            
        Returns:
            True if a record was deleted
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM conversation_document_overrides
                    WHERE conversation_id = %s AND document_id = %s
                    """,
                    (conversation_id, document_id)
                )
                conn.commit()
                return cursor.rowcount > 0
        finally:
            self._release_connection(conn)
    
    # =========================================================================
    # Effective Document Selection (for search queries)
    # =========================================================================
    
    def get_enabled_document_ids(
        self,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Set[int]:
        """
        Get IDs of documents enabled for search using 3-tier precedence.
        
        Implements: Effective selection query
        
        The query uses: COALESCE(conversation_override, user_default, TRUE)
        
        Args:
            user_id: Optional user ID for user default lookup
            conversation_id: Optional conversation ID for override lookup
            
        Returns:
            Set of enabled document IDs
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT d.id
                    FROM documents d
                    LEFT JOIN user_document_defaults ud 
                        ON d.id = ud.document_id 
                        AND ud.user_id = %s
                    LEFT JOIN conversation_document_overrides co 
                        ON d.id = co.document_id 
                        AND co.conversation_id = %s
                    WHERE d.is_deleted = FALSE
                      AND COALESCE(co.enabled, ud.enabled, TRUE) = TRUE
                    """,
                    (user_id, conversation_id)
                )
                
                return {row[0] for row in cursor.fetchall()}
        finally:
            self._release_connection(conn)
    
    def get_enabled_resource_hashes(
        self,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Set[str]:
        """
        Get resource hashes of documents enabled for search.
        
        This is useful when filtering document_chunks by resource_hash
        in the PostgresVectorStore search.
        
        Args:
            user_id: Optional user ID for user default lookup
            conversation_id: Optional conversation ID for override lookup
            
        Returns:
            Set of enabled resource hashes
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT d.resource_hash
                    FROM documents d
                    LEFT JOIN user_document_defaults ud 
                        ON d.id = ud.document_id 
                        AND ud.user_id = %s
                    LEFT JOIN conversation_document_overrides co 
                        ON d.id = co.document_id 
                        AND co.conversation_id = %s
                    WHERE d.is_deleted = FALSE
                      AND COALESCE(co.enabled, ud.enabled, TRUE) = TRUE
                    """,
                    (user_id, conversation_id)
                )
                
                return {row[0] for row in cursor.fetchall()}
        finally:
            self._release_connection(conn)
    
    def get_full_selection_state(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> List[DocumentSelection]:
        """
        Get full selection state for all documents showing all tiers.
        
        Useful for UI that displays document selection controls.
        
        Args:
            user_id: The user's ID
            conversation_id: Optional conversation ID
            
        Returns:
            List of DocumentSelection objects with full tier information
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT 
                        d.id as document_id,
                        d.resource_hash,
                        d.display_name,
                        d.source_type,
                        ud.enabled as user_default,
                        co.enabled as conversation_override
                    FROM documents d
                    LEFT JOIN user_document_defaults ud 
                        ON d.id = ud.document_id AND ud.user_id = %s
                    LEFT JOIN conversation_document_overrides co 
                        ON d.id = co.document_id AND co.conversation_id = %s
                    WHERE d.is_deleted = FALSE
                    ORDER BY d.display_name
                    """,
                    (user_id, conversation_id)
                )
                
                rows = cursor.fetchall()
                
                return [
                    DocumentSelection(
                        document_id=row["document_id"],
                        resource_hash=row["resource_hash"],
                        display_name=row["display_name"],
                        source_type=row["source_type"],
                        user_default=row["user_default"],
                        conversation_override=row["conversation_override"],
                    )
                    for row in rows
                ]
        finally:
            self._release_connection(conn)
    
    # =========================================================================
    # Bulk Operations
    # =========================================================================
    
    def set_user_defaults_bulk(
        self,
        user_id: str,
        selections: Dict[int, bool],
    ) -> int:
        """
        Set multiple user defaults at once.
        
        Args:
            user_id: The user's ID
            selections: Dict of {document_id: enabled}
            
        Returns:
            Number of records upserted
        """
        if not selections:
            return 0
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                data = [
                    (user_id, doc_id, enabled)
                    for doc_id, enabled in selections.items()
                ]
                
                psycopg2.extras.execute_values(
                    cursor,
                    """
                    INSERT INTO user_document_defaults (user_id, document_id, enabled)
                    VALUES %s
                    ON CONFLICT (user_id, document_id) DO UPDATE SET
                        enabled = EXCLUDED.enabled,
                        updated_at = NOW()
                    """,
                    data,
                    template="(%s, %s, %s)",
                )
                conn.commit()
                
                logger.info(f"Bulk set {len(selections)} user defaults for {user_id}")
                return len(selections)
        finally:
            self._release_connection(conn)
    
    def set_conversation_overrides_bulk(
        self,
        conversation_id: str,
        selections: Dict[int, bool],
    ) -> int:
        """
        Set multiple conversation overrides at once.
        
        Args:
            conversation_id: The conversation ID
            selections: Dict of {document_id: enabled}
            
        Returns:
            Number of records upserted
        """
        if not selections:
            return 0
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                data = [
                    (conversation_id, doc_id, enabled)
                    for doc_id, enabled in selections.items()
                ]
                
                psycopg2.extras.execute_values(
                    cursor,
                    """
                    INSERT INTO conversation_document_overrides (conversation_id, document_id, enabled)
                    VALUES %s
                    ON CONFLICT (conversation_id, document_id) DO UPDATE SET
                        enabled = EXCLUDED.enabled,
                        updated_at = NOW()
                    """,
                    data,
                    template="(%s, %s, %s)",
                )
                conn.commit()
                
                logger.info(f"Bulk set {len(selections)} conversation overrides for {conversation_id}")
                return len(selections)
        finally:
            self._release_connection(conn)
