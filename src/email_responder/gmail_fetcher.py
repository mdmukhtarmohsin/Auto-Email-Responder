"""Gmail fetcher using MCP and Gmail API for batch email processing."""

import logging
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from fastmcp import FastMCP

from .config import config

logger = logging.getLogger(__name__)

# Gmail API scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.send'
]


@dataclass
class EmailData:
    """Email data structure."""
    id: str
    subject: str
    sender: str
    sender_email: str
    body: str
    timestamp: datetime
    thread_id: str
    labels: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'subject': self.subject,
            'sender': self.sender,
            'sender_email': self.sender_email,
            'body': self.body,
            'timestamp': self.timestamp.isoformat(),
            'thread_id': self.thread_id,
            'labels': self.labels
        }


class GmailFetcher:
    """Gmail fetcher with MCP and API integration."""
    
    def __init__(self):
        self.service = None
        self.credentials = None
        self.mcp_server = FastMCP("Gmail Email Responder")
        self._setup_gmail_service()
        self._setup_mcp_tools()
    
    def _setup_gmail_service(self) -> None:
        """Setup Gmail API service."""
        try:
            creds = None
            
            # Load existing token
            if os.path.exists(config.gmail_token_path):
                creds = Credentials.from_authorized_user_file(config.gmail_token_path, SCOPES)
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if os.path.exists(config.gmail_credentials_path):
                        flow = InstalledAppFlow.from_client_secrets_file(
                            config.gmail_credentials_path, SCOPES
                        )
                        creds = flow.run_local_server(port=0)
                    else:
                        logger.error(f"Gmail credentials file not found: {config.gmail_credentials_path}")
                        return
                
                # Save the credentials for the next run
                with open(config.gmail_token_path, 'w') as token:
                    token.write(creds.to_json())
            
            self.credentials = creds
            self.service = build('gmail', 'v1', credentials=creds)
            logger.info("Gmail service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Gmail service: {e}")
            self.service = None
    
    def _setup_mcp_tools(self) -> None:
        """Setup MCP tools for email operations."""
        
        @self.mcp_server.tool()
        def fetch_unread_emails(max_results: int = 10) -> List[Dict[str, Any]]:
            """Fetch unread emails from Gmail."""
            emails = self.fetch_unread_emails(max_results)
            return [email.to_dict() for email in emails]
        
        @self.mcp_server.tool()
        def mark_email_processed(email_id: str) -> bool:
            """Mark an email as processed by adding a label."""
            return self.mark_as_processed(email_id)
        
        @self.mcp_server.tool()
        def get_email_content(email_id: str) -> Optional[Dict[str, Any]]:
            """Get full email content by ID."""
            email = self.get_email_by_id(email_id)
            return email.to_dict() if email else None
    
    def _extract_email_body(self, payload: Dict[str, Any]) -> str:
        """Extract email body from Gmail API payload."""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        import base64
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
                elif part['mimeType'] == 'multipart/alternative':
                    body = self._extract_email_body(part)
                    if body:
                        break
        elif payload['mimeType'] == 'text/plain':
            if 'data' in payload['body']:
                import base64
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return body.strip()
    
    def _extract_headers(self, headers: List[Dict[str, str]]) -> Dict[str, str]:
        """Extract relevant headers from email."""
        header_dict = {}
        for header in headers:
            name = header.get('name', '').lower()
            value = header.get('value', '')
            if name in ['subject', 'from', 'date', 'to']:
                header_dict[name] = value
        return header_dict
    
    def fetch_unread_emails(self, max_results: Optional[int] = None) -> List[EmailData]:
        """Fetch unread emails from Gmail."""
        if not self.service:
            logger.error("Gmail service not initialized")
            return []
        
        max_results = max_results or config.max_emails_per_batch
        emails = []
        
        try:
            # Query for unread emails
            query = 'is:unread -label:auto-responder-processed'
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            logger.info(f"Found {len(messages)} unread emails")
            
            for message in messages:
                email_data = self.get_email_by_id(message['id'])
                if email_data:
                    emails.append(email_data)
            
        except HttpError as error:
            logger.error(f"An error occurred fetching emails: {error}")
        
        return emails
    
    def get_email_by_id(self, email_id: str) -> Optional[EmailData]:
        """Get email details by ID."""
        if not self.service:
            return None
        
        try:
            message = self.service.users().messages().get(
                userId='me',
                id=email_id,
                format='full'
            ).execute()
            
            headers = self._extract_headers(message['payload'].get('headers', []))
            body = self._extract_email_body(message['payload'])
            
            # Parse sender
            from_header = headers.get('from', '')
            sender_email = ''
            sender_name = from_header
            
            if '<' in from_header and '>' in from_header:
                sender_name = from_header.split('<')[0].strip().strip('"')
                sender_email = from_header.split('<')[1].split('>')[0].strip()
            else:
                sender_email = from_header
            
            # Parse timestamp
            timestamp = datetime.now()
            if 'date' in headers:
                try:
                    from email.utils import parsedate_to_datetime
                    timestamp = parsedate_to_datetime(headers['date'])
                except Exception:
                    pass
            
            email_data = EmailData(
                id=email_id,
                subject=headers.get('subject', ''),
                sender=sender_name,
                sender_email=sender_email,
                body=body,
                timestamp=timestamp,
                thread_id=message.get('threadId', ''),
                labels=message.get('labelIds', [])
            )
            
            return email_data
            
        except HttpError as error:
            logger.error(f"An error occurred getting email {email_id}: {error}")
            return None
    
    def mark_as_processed(self, email_id: str) -> bool:
        """Mark email as processed by adding a label."""
        if not self.service:
            return False
        
        try:
            # Create custom label if it doesn't exist
            label_name = 'auto-responder-processed'
            labels_result = self.service.users().labels().list(userId='me').execute()
            labels = labels_result.get('labels', [])
            
            label_id = None
            for label in labels:
                if label['name'] == label_name:
                    label_id = label['id']
                    break
            
            if not label_id:
                # Create the label
                label_object = {
                    'name': label_name,
                    'labelListVisibility': 'labelShow',
                    'messageListVisibility': 'show'
                }
                created_label = self.service.users().labels().create(
                    userId='me',
                    body=label_object
                ).execute()
                label_id = created_label['id']
            
            # Add label to message
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'addLabelIds': [label_id]}
            ).execute()
            
            logger.debug(f"Marked email {email_id} as processed")
            return True
            
        except HttpError as error:
            logger.error(f"An error occurred marking email as processed: {error}")
            return False
    
    def get_mcp_server(self) -> FastMCP:
        """Get the MCP server instance."""
        return self.mcp_server


# Global Gmail fetcher instance
gmail_fetcher = GmailFetcher() 