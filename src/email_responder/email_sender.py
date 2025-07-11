"""Email Sender using Gmail API with FastMCP integration."""

import logging
import base64
import json
import os
from typing import Optional, Dict, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastmcp import FastMCP
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .config import config

logger = logging.getLogger(__name__)


class EmailSender:
    """Handles sending email responses using Gmail API."""
    
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    
    def __init__(self):
        self.service = None
        self.mcp_server = None
        self._initialize_gmail_service()
        self._setup_mcp_server()
    
    def _initialize_gmail_service(self) -> None:
        """Initialize Gmail API service with authentication."""
        try:
            creds = None
            
            # Load existing token
            if os.path.exists(config.gmail_token_path):
                try:
                    creds = Credentials.from_authorized_user_file(
                        config.gmail_token_path, 
                        self.SCOPES
                    )
                except Exception as e:
                    logger.warning(f"Failed to load existing token: {e}")
            
            # Refresh or obtain new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        logger.info("Refreshed Gmail credentials")
                    except Exception as e:
                        logger.warning(f"Failed to refresh credentials: {e}")
                        creds = None
                
                if not creds:
                    if not os.path.exists(config.gmail_credentials_path):
                        logger.error(f"Gmail credentials file not found: {config.gmail_credentials_path}")
                        return
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        config.gmail_credentials_path, 
                        self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    logger.info("Obtained new Gmail credentials")
                
                # Save credentials for future use
                with open(config.gmail_token_path, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"Saved credentials to {config.gmail_token_path}")
            
            # Build Gmail service
            self.service = build('gmail', 'v1', credentials=creds)
            logger.info("Gmail sender service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gmail service: {e}")
    
    def _setup_mcp_server(self) -> None:
        """Setup FastMCP server for email sending operations."""
        try:
            self.mcp_server = FastMCP("Gmail Email Sender")
            
            @self.mcp_server.tool()
            def send_email_response(
                to_email: str,
                subject: str,
                body: str,
                in_reply_to: Optional[str] = None,
                thread_id: Optional[str] = None
            ) -> Dict[str, Any]:
                """Send an email response via Gmail API.
                
                Args:
                    to_email: Recipient email address
                    subject: Email subject line
                    body: Email body content
                    in_reply_to: Message ID this is replying to
                    thread_id: Gmail thread ID for conversation threading
                
                Returns:
                    Dict with success status and message details
                """
                return self._send_email_impl(
                    to_email=to_email,
                    subject=subject,
                    body=body,
                    in_reply_to=in_reply_to,
                    thread_id=thread_id
                )
            
            @self.mcp_server.tool()
            def get_sender_status() -> Dict[str, Any]:
                """Get the status of the email sender service.
                
                Returns:
                    Dict with service status and configuration
                """
                return {
                    "service_ready": self.service is not None,
                    "sender_email": config.gmail_email_address,
                    "scopes": self.SCOPES,
                    "credentials_configured": os.path.exists(config.gmail_credentials_path),
                    "token_exists": os.path.exists(config.gmail_token_path)
                }
            
            logger.info("FastMCP email sender server setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup MCP server: {e}")
    
    def _send_email_impl(
        self,
        to_email: str,
        subject: str,
        body: str,
        in_reply_to: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal implementation for sending emails."""
        
        if not self.service:
            error_msg = "Gmail service not initialized"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            # Create email message
            message = self._create_message(
                to_email=to_email,
                subject=subject,
                body=body,
                in_reply_to=in_reply_to
            )
            
            # Send email
            if thread_id:
                # Send as part of existing thread
                sent_message = self.service.users().messages().send(
                    userId='me',
                    body=message,
                    threadId=thread_id
                ).execute()
            else:
                # Send as new thread
                sent_message = self.service.users().messages().send(
                    userId='me',
                    body=message
                ).execute()
            
            logger.info(f"Email sent successfully to {to_email} - Message ID: {sent_message['id']}")
            
            return {
                "success": True,
                "message_id": sent_message['id'],
                "thread_id": sent_message.get('threadId'),
                "to_email": to_email,
                "subject": subject
            }
            
        except HttpError as e:
            error_msg = f"Gmail API error: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error sending email: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _create_message(
        self,
        to_email: str,
        subject: str,
        body: str,
        in_reply_to: Optional[str] = None
    ) -> Dict[str, str]:
        """Create email message in Gmail API format."""
        
        # Create MIME message
        message = MIMEMultipart('alternative')
        message['To'] = to_email
        message['From'] = config.gmail_email_address
        message['Subject'] = subject
        
        if in_reply_to:
            message['In-Reply-To'] = in_reply_to
            message['References'] = in_reply_to
        
        # Add HTML and plain text versions
        text_part = MIMEText(body, 'plain', 'utf-8')
        html_body = self._convert_to_html(body)
        html_part = MIMEText(html_body, 'html', 'utf-8')
        
        message.attach(text_part)
        message.attach(html_part)
        
        # Encode message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        return {'raw': raw_message}
    
    def _convert_to_html(self, text_body: str) -> str:
        """Convert plain text email to HTML format."""
        
        # Simple text to HTML conversion
        html_body = text_body.replace('\n\n', '</p><p>')
        html_body = html_body.replace('\n', '<br>')
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                p {{
                    margin-bottom: 15px;
                }}
                .signature {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <p>{html_body}</p>
        </body>
        </html>
        """
        
        return html_template
    
    def send_response_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        original_message_id: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> bool:
        """Send an email response (public interface).
        
        Args:
            to_email: Recipient email address
            subject: Email subject (will be prefixed with "Re:" if not already)
            body: Email body content
            original_message_id: ID of original message being replied to
            thread_id: Gmail thread ID for conversation threading
        
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        
        # Format subject for reply
        if not subject.lower().startswith('re:'):
            subject = f"Re: {subject}"
        
        result = self._send_email_impl(
            to_email=to_email,
            subject=subject,
            body=body,
            in_reply_to=original_message_id,
            thread_id=thread_id
        )
        
        return result.get('success', False)
    
    def test_connection(self) -> bool:
        """Test if the Gmail sender is working properly."""
        if not self.service:
            return False
        
        try:
            # Try to get user profile to test connection
            profile = self.service.users().getProfile(userId='me').execute()
            logger.info(f"Gmail sender test successful - Email: {profile.get('emailAddress')}")
            return True
        except Exception as e:
            logger.error(f"Gmail sender test failed: {e}")
            return False
    
    def get_mcp_server(self) -> Optional[FastMCP]:
        """Get the FastMCP server instance for external use."""
        return self.mcp_server


# Global email sender instance
email_sender = EmailSender() 