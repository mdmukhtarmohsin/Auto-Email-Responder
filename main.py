#!/usr/bin/env python3
"""
AI-Powered Email Responder - Main Application
============================================

Automated email response system using Gemini 2.5, LangChain, LangGraph, and Gmail API.

Usage:
    python3 main.py                    # Run once
    python3 main.py --daemon            # Run continuously  
    python3 main.py --status            # Check system status
    python3 main.py --refresh-policies  # Refresh policy index
    python3 main.py --test              # Test all components
"""

import logging
import time
import argparse
import signal
import sys
from datetime import datetime
from typing import Dict, Any

from src.email_responder.config import config
from src.email_responder.workflow import email_workflow
from src.email_responder.gmail_fetcher import gmail_fetcher
from src.email_responder.retriever_chain import policy_retriever
from src.email_responder.llm_response_chain import llm_response_chain
from src.email_responder.email_sender import email_sender
from src.email_responder.cache_manager import cache_manager

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/email_responder.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class EmailResponderApp:
    """Main application class for the Email Responder system."""
    
    def __init__(self):
        self.running = False
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run_once(self) -> Dict[str, Any]:
        """Process emails once and return results."""
        logger.info("Starting single email processing run")
        
        try:
            # Process emails using the workflow
            result = email_workflow.process_emails(
                thread_id=f"single_run_{int(time.time())}"
            )
            
            logger.info(f"Processing complete: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Error during email processing: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def run_daemon(self):
        """Run continuously as a daemon process."""
        logger.info(f"Starting email responder daemon (interval: {config.processing_interval_minutes} minutes)")
        
        self.running = True
        last_processing_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                interval_seconds = config.processing_interval_minutes * 60
                
                # Check if it's time to process emails
                if current_time - last_processing_time >= interval_seconds:
                    logger.info("Starting scheduled email processing")
                    
                    result = email_workflow.process_emails(
                        thread_id=f"daemon_{int(current_time)}"
                    )
                    
                    if result.get("success"):
                        logger.info(f"Daemon processing successful: {result.get('successful_responses', 0)} responses sent")
                    else:
                        logger.error(f"Daemon processing failed: {result.get('error', 'Unknown error')}")
                    
                    last_processing_time = current_time
                else:
                    # Sleep for a short time before checking again
                    time.sleep(30)
                    
            except KeyboardInterrupt:
                logger.info("Daemon interrupted by user")
                break
            except Exception as e:
                logger.error(f"Daemon error: {e}")
                time.sleep(60)  # Wait before retrying
        
        logger.info("Email responder daemon stopped")
    
    def check_status(self) -> Dict[str, Any]:
        """Check the status of all system components."""
        logger.info("Checking system status")
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "gemini_model": config.gemini_model,
                "gmail_email": config.gmail_email_address,
                "processing_interval": config.processing_interval_minutes,
                "max_emails_per_batch": config.max_emails_per_batch,
                "response_tone": config.response_tone
            },
            "components": email_workflow.get_workflow_status(),
            "cache_stats": cache_manager.get_cache_stats(),
            "policy_stats": policy_retriever.get_index_stats()
        }
        
        return status
    
    def refresh_policies(self) -> bool:
        """Refresh the policy knowledge base index."""
        logger.info("Refreshing policy knowledge base")
        
        try:
            success = policy_retriever.refresh_index()
            if success:
                logger.info("Policy index refreshed successfully")
            else:
                logger.error("Failed to refresh policy index")
            return success
        except Exception as e:
            logger.error(f"Error refreshing policies: {e}")
            return False
    
    def test_components(self) -> Dict[str, bool]:
        """Test all system components."""
        logger.info("Testing all system components")
        
        test_results = {}
        
        try:
            # Test Gmail fetcher
            test_results["gmail_fetcher"] = gmail_fetcher.test_connection()
            logger.info(f"Gmail fetcher test: {'PASS' if test_results['gmail_fetcher'] else 'FAIL'}")
            
            # Test LLM response chain
            test_results["llm_response"] = llm_response_chain.test_connection()
            logger.info(f"LLM response test: {'PASS' if test_results['llm_response'] else 'FAIL'}")
            
            # Test email sender
            test_results["email_sender"] = email_sender.test_connection()
            logger.info(f"Email sender test: {'PASS' if test_results['email_sender'] else 'FAIL'}")
            
            # Test cache manager
            test_results["cache_manager"] = cache_manager.test_connection()
            logger.info(f"Cache manager test: {'PASS' if test_results['cache_manager'] else 'FAIL'}")
            
            # Test policy retriever
            policy_stats = policy_retriever.get_index_stats()
            test_results["policy_retriever"] = policy_stats.get("vectorstore_exists", False)
            logger.info(f"Policy retriever test: {'PASS' if test_results['policy_retriever'] else 'FAIL'}")
            
            # Overall system test
            all_pass = all(test_results.values())
            test_results["overall"] = all_pass
            logger.info(f"Overall system test: {'PASS' if all_pass else 'FAIL'}")
            
        except Exception as e:
            logger.error(f"Error during component testing: {e}")
            test_results["error"] = str(e)
        
        return test_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Email Responder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--daemon", 
        action="store_true", 
        help="Run continuously as daemon process"
    )
    parser.add_argument(
        "--status", 
        action="store_true", 
        help="Check system status and component health"
    )
    parser.add_argument(
        "--refresh-policies", 
        action="store_true", 
        help="Refresh policy knowledge base index"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test all system components"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set log level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Initialize application
    app = EmailResponderApp()
    
    try:
        if args.status:
            # Check status
            status = app.check_status()
            print("\n=== EMAIL RESPONDER SYSTEM STATUS ===")
            print(f"Timestamp: {status['timestamp']}")
            print(f"Gemini Model: {status['configuration']['gemini_model']}")
            print(f"Gmail Email: {status['configuration']['gmail_email']}")
            print(f"Processing Interval: {status['configuration']['processing_interval']} minutes")
            print("\nComponent Status:")
            for component, ready in status['components'].items():
                status_icon = "✓" if ready else "✗"
                print(f"  {status_icon} {component}: {'Ready' if ready else 'Not Ready'}")
            
            # Print policy stats
            policy_stats = status['policy_stats']
            print(f"\nPolicy Index:")
            print(f"  Vectorstore: {'Exists' if policy_stats.get('vectorstore_exists') else 'Missing'}")
            print(f"  Documents: {policy_stats.get('document_count', 'Unknown')}")
            
        elif args.refresh_policies:
            # Refresh policies
            success = app.refresh_policies()
            if success:
                print("✓ Policy knowledge base refreshed successfully")
                sys.exit(0)
            else:
                print("✗ Failed to refresh policy knowledge base")
                sys.exit(1)
                
        elif args.test:
            # Test components
            results = app.test_components()
            print("\n=== COMPONENT TEST RESULTS ===")
            for component, passed in results.items():
                if component == "error":
                    continue
                status_icon = "✓" if passed else "✗"
                print(f"  {status_icon} {component}: {'PASS' if passed else 'FAIL'}")
            
            if "error" in results:
                print(f"\nError: {results['error']}")
                sys.exit(1)
            elif results.get("overall"):
                print("\n✓ All components are working correctly")
                sys.exit(0)
            else:
                print("\n✗ Some components failed tests")
                sys.exit(1)
                
        elif args.daemon:
            # Run as daemon
            app.run_daemon()
            
        else:
            # Run once
            result = app.run_once()
            if result.get("success"):
                print(f"✓ Processing complete: {result.get('successful_responses', 0)} responses sent")
                if result.get('processing_log'):
                    print("\nProcessing Log:")
                    for log_entry in result['processing_log']:
                        print(f"  {log_entry}")
            else:
                print(f"✗ Processing failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 